# MXFP4/MXFP8 伪量化与算法适配方案（v0.23）

## 一、目标和非目标

目标是在只支持 INT4/INT8、但不能执行原生 MXFP4/MXFP8 kernel 的 Ascend
设备上验证以下内容：

- OCP MXFP4（E2M1 element + E8M0 block scale）数值误差；
- OCP MXFP8（E4M3 element + E8M0 block scale）数值误差；
- RTN、OmniQuant、RHT、FlatQuant、AutoRound 对误差和模型精度的影响；
- Qwen3.5 dense Linear 和 Qwen3.5 MoE grouped matmul 的一致插入点。

非目标：

- 不生成真正 packed FP4/FP8 checkpoint；
- 不调用原生 MX quant matmul/grouped matmul；
- 不宣称 fake 路径的吞吐、显存或带宽等于真实 MX 硬件；
- 不在 vLLM serving 进程中执行校准训练。

## 二、为什么算法校准与 serving 必须分层

vLLM/vllm-ascend 是推理运行时。OmniQuant、FlatQuant、AutoRound 都需要
校准数据、反向传播或 block-wise 优化。把这些过程塞入模型加载会导致：

1. serving 启动依赖训练数据和优化器；
2. TP/EP 环境中重复校准，产物不确定；
3. 无法区分算法收益和运行时实现误差；
4. checkpoint 不可复现。

因此采用两阶段契约：

```text
离线算法工具
  -> 变换/优化 FP checkpoint
  -> 写入算法元数据与 QDQ 误差
  -> 保存 FP16/BF16 validation checkpoint

vLLM/vllm-ascend
  -> 校验 fake_mx_weight_state
  -> 加载算法产物
  -> 在线执行必要的 activation transform
  -> 共用 fake MX QDQ
  -> 普通 FP GEMM/GMM
```

## 三、共用 MX QDQ 实现

文件：`vllm_ascend/quantization/fake_mx.py`

### `fake_mx_quantize`

功能：

1. 沿最后一维按 `group_size` 分块，默认 32；
2. 计算每块有限值 amax；
3. 用 E8M0 power-of-two scale；
4. scale 的指数向上取整，保证最大值能落入 element range；
5. MXFP4 用 E2M1、MXFP8 用有限 E4M3；
6. 使用 round-to-nearest-even；
7. 立即反量化回原 FP dtype；
8. 支持标量或逐 block `clip_ratio`；
9. NaN 保留，Inf 饱和。

该实现与 Intel AutoRound 中 `mx_fp_rceil` 的 scale 语义对齐；AutoRound
默认 `quant_mx` 还提供另一种 exponent 选择，生成本项目 checkpoint 时必须
显式使用 OCP-compatible rceil，不能混用。

### `randomized_hadamard_transform`

功能：

- 对最后一维按 power-of-two group 做 normalized Walsh-Hadamard；
- 先乘 checkpoint/seed 对应的 Rademacher sign；
- 用普通 Torch butterfly 算子，不依赖 fast-hadamard NPU kernel；
- 输入输出 dtype 和 shape 不变；
- 同一 sign 必须用于离线 weight rotation 和在线 activation rotation。

## 四、算法一：RTN baseline

### 原始方法

RTN（round-to-nearest）不使用校准训练。权重和激活根据当前 block amax
直接映射到目标格式，是所有优化算法的基线。

### 本项目实现

scheme：

- `W4A4_MXFP4_FAKE`
- `W8A8_MXFP8_FAKE`

Linear：

- `process_weights_after_loading` 对 weight QDQ 一次；
- `apply` 对 activation QDQ；
- `F.linear`。

MoE：

- w13/w2 加载后各 QDQ 一次；
- expert FC1 输入 QDQ；
- activation 后、FC2 前再次 QDQ；
- ordinary floating-point GMM。

### 插入理由

权重是静态参数，只需在加载后写入一次误差。激活随 token 改变，必须在每次
forward、且紧邻矩阵乘之前 QDQ。MoE FC2 的输入是 SwiGLU 结果，它与 FC1
输入具有不同分布，不能复用 FC1 的 QDQ 结果。

## 五、算法二：OmniQuant

### 原始实现

论文与官方实现：

- Paper: <https://arxiv.org/abs/2308.13137>
- Code: <https://github.com/OpenGVLab/OmniQuant>
- 本次对照 commit: `feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14`

核心组件：

- LWC（Learnable Weight Clipping）：优化 weight 上下 clipping bound；
- LET（Learnable Equivalent Transformation）：学习等价 scale/shift，把
  activation outlier 的难度转移到 weight；
- 逐 decoder block 最小化 FP output 与 fake-quant output 的重建误差。

官方实现中的关键函数：

- `quantize/omniquant.py:omniquant`
- `models/transformation.py:smooth_ln_fcs_inplace`
- `models/transformation.py:smooth_fc_fc_inplace`
- `models/transformation.py:smooth_q_k_inplace`
- `quantize/quantizer.py:UniformAffineQuantizer`
- `smooth_and_quant_temporary`：训练时临时等价变换；
- `smooth_and_quant_inplace`：收敛后把 LET 折叠进 Norm/Linear。

以 Norm -> Linear 为例，等价折叠为：

```text
norm.weight <- norm.weight / scale
norm.bias   <- (norm.bias - shift) / scale
fc.weight   <- fc.weight * scale
fc.bias     <- fc.bias + original_fc_weight @ shift
```

### 本项目 checkpoint 契约

配置：

```json
{
  "default_quant_type": "W4A4_MXFP4_OMNIQUANT_FAKE",
  "fake_mx_weight_state": "prequantized_qdq",
  "group_size": 32
}
```

权重要求：

1. LET 已经离线折叠进 Norm、q/k/v/o、MLP 或 expert 权重；
2. LWC 已经应用；
3. 使用本项目相同 MX QDQ 规则把 weight 误差写入；
4. 保存为 FP16/BF16 tensor，而不是 packed MX；
5. 未量化的 router、embedding、lm_head 明确标记 `FLOAT`。

runtime scheme：

- `W4A4_MXFP4_OMNIQUANT_FAKE`
- `W8A8_MXFP8_OMNIQUANT_FAKE`
- Linear 和 MoE 都支持；
- `process_weights_after_loading` 检查 `prequantized_qdq` 后跳过 weight QDQ；
- forward 仍对 activation QDQ。

### 为什么不在 runtime 重放 LET

LET 同时修改 Norm 和多个相邻 Linear。只在单个 Linear 前乘 scale/shift 会
遗漏 bias、q/k 成对变换和 residual 边界，无法保证函数等价。官方实现本身
也在保存前调用 inplace folding，因此 runtime 应消费折叠后的 checkpoint。

## 六、算法三：RHT

### 原始方法与开源对应

RHT 指 randomized Hadamard transform。它使用
`R = H D / sqrt(n)` 将 outlier 能量分散到各通道，其中 `D` 是随机 ±1
对角阵。参考实现：

- AutoRound transform framework:
  <https://github.com/intel/auto-round/tree/main/auto_round/algorithms/transforms/hadamard>
- SpinQuant: <https://github.com/facebookresearch/SpinQuant>
- 本次 AutoRound 对照 commit:
  `60b813cb3bad82d683de72cc29a4b6315b2a1156`

AutoRound 关键文件：

- `algorithms/transforms/hadamard/inplace/apply.py`
- `algorithms/transforms/hadamard/inplace/hooks.py`
- `algorithms/transforms/hadamard/utils/math.py`
- `experimental/qmodules/mx.py`

其基本模式是：

```text
W_rotated = W @ R
x_runtime = x @ R
y = x_runtime @ W_rotated.T = x @ W.T
```

因 `R R.T = I`，变换前后 FP 函数等价；量化后，旋转通常减小 block outlier。

### 本项目 checkpoint 契约

```json
{
  "default_quant_type": "W4A4_MXFP4_RHT_FAKE",
  "fake_mx_weight_state": "rht_rotated_fp",
  "group_size": 32,
  "rht_group_size": 32,
  "rht_seed": 2026
}
```

离线侧：

1. 用相同 seed 生成 Rademacher signs；
2. 每 32 个输入通道构造 normalized RHT；
3. 将每个 Linear weight 的输入维做 `W @ R`；
4. MoE w13 的 hidden 维、w2 的 intermediate 维分别旋转；
5. 保存未 QDQ 的 rotated FP weight。

运行时：

- 加载后对 rotated weight 执行一次 MX QDQ；
- Linear 输入先 RHT，再 MX QDQ；
- MoE FC1 输入用 w13 signs；
- MoE SwiGLU 后用 w2 signs，再进入 FC2；
- TP RowParallel 使用 global sign 序列中当前 rank 的输入切片。

### 插入理由

RHT 必须紧邻被旋转 weight 的输入。放在 Decoder block 入口只覆盖
qkv/gate_up，无法覆盖 o_proj/down_proj；放进 MX QDQ 内部又无法表达不同
Linear 的旋转维度。Linear method 和 MoE GMM1/GMM2 边界是最小且完备的点。

## 七、算法四：Hadamard Learning

Hadamard Learning 对齐 AMCT-Q 的 `LearnableHadamard`：训练一个可逆
`K x K` 矩阵 `T`，activation 分块执行 `x @ T`，weight 分块执行
`W @ inverse(T).T`。它与使用固定正交矩阵的 RHT 是两个独立 scheme：

- `W4A4_MXFP4_HADAMARD_LEARNING_FAKE`；
- `W8A8_MXFP8_HADAMARD_LEARNING_FAKE`；
- checkpoint marker：`hadamard_learning_transformed_fp`；
- Linear 参数：`transform_weight[K,K]`；
- routed MoE 参数：`w13_transform_weight[E,K,K]` 和
  `w2_transform_weight[E,K,K]`。

35B routed MoE 的矩阵是 per-expert，FC1 变换必须放在 dispatch 后、GMM1
前；FC2 变换放在 SwiGLU 后、GMM2 前。详细训练调用链、AMCT-Q 导出映射和
逐文件修改见
`qwen3_5_hadamard_learning_fake_mx_v023.md`。

## 八、算法五：FlatQuant

### 原始实现

- Paper: <https://arxiv.org/abs/2410.09426>
- Code: <https://github.com/ruikangliu/FlatQuant>
- 本次对照 commit: `9d88ffcb7d2c6bda59fb5c44dad36adc101aadb1`

FlatQuant 学习快速 affine transform，使 weight 和 activation 的 loss
landscape 更平坦。官方实现用 Kronecker 分解降低大矩阵变换成本。

关键函数：

- `flatquant/flat_linear.py:FlatQuantizedLinear._train_forward`
- `FlatQuantizedLinear.reparameterize`
- `flatquant/trans_utils.py:kronecker_matmul`
- vLLM 参考：
  `vllm_custom/model_executor/layers/quantization/utils/flatquant_utils.py`

训练时：

```text
weight -> inverse/transpose transform -> clipping -> weight fake quant
x      -> forward transform -> activation fake quant
output -> optional output transform
```

收敛后 `reparameterize` 把 weight-side transform 和 clipping 写回 weight，
runtime 只保留 activation transform。

### v0.23 原生 Ascend 对应

文件：
`vllm_ascend/quantization/methods/w4a4_mxfp4_flatquant.py`

关键函数：

- `get_decompose_dim`
- `AscendW4A4MXFP4FlatQuantDynamicLinearMethod.get_weight`
- `get_pertensor_param`
- `apply`
- `process_weights_after_loading`

原生 `apply` 调用 `torch_npu.npu_kronecker_quant` 和
`torch_npu.npu_quant_matmul`。fake 实现保持相同 checkpoint 参数和 TP
矩阵切分，但把两步替换为普通 `torch.matmul` + `fake_mx_quantize` +
`F.linear`。

### 本项目契约

```json
{
  "default_quant_type": "W4A4_MXFP4_FLATQUANT_FAKE",
  "fake_mx_weight_state": "flatquant_transformed_fp",
  "group_size": 32,
  "max_supported_tp": 4
}
```

每个 Linear 提供：

- transformed floating-point `weight`
- `left_trans`
- `right_trans`
- `clip_ratio`

运行时：

```text
x.reshape(-1, left_dim, right_dim)
  -> left_trans @ x @ right_trans
  -> block clip
  -> MX fake QDQ
  -> F.linear(weight_qdq)
```

当前只支持 Linear，与 v0.23 原生 Ascend FlatQuant 能力保持一致。routed
MoE 需要逐 expert transform 和路由后的矩阵选择，当前不声称支持。Qwen3.5
35B-A3B 的 shared expert 可走 Linear FlatQuant；routed experts 必须选择 RTN、
RHT、OmniQuant 或 AutoRound fake scheme。

## 九、算法六：AutoRound

### 原始实现

- Code: <https://github.com/intel/auto-round>
- 文档:
  <https://github.com/intel/auto-round/blob/main/docs/step_by_step.md>
- 本次对照 commit:
  `60b813cb3bad82d683de72cc29a4b6315b2a1156`

AutoRound 用少量校准数据优化 rounding 和 clipping，而不是只做 RTN。
当前开源实现已提供 MXFP4/MXFP8 和 Hadamard transform。

MX 关键文件：

- `auto_round/data_type/mxfp.py:quant_mx`
- `auto_round/data_type/mxfp.py:quant_mx_rceil`
- `auto_round/algorithms/quantization/sign_round/quantizer.py`
- `auto_round/experimental/qmodules/mx.py`

`quant_mx` 中的主要可优化量：

- `v`：element rounding 前的偏移；
- `max_scale`：block amax multiplier / clipping；
- best-MSE checkpoint：保存迭代中重建误差最小的参数。

### 本项目 checkpoint 契约

```json
{
  "default_quant_type": "W4A4_MXFP4_AUTOROUND_FAKE",
  "fake_mx_weight_state": "prequantized_qdq",
  "group_size": 32
}
```

离线导出要求：

1. AutoRound 优化使用与目标一致的 MXFP4/MXFP8；
2. E8M0 scale 使用 OCP-compatible rceil；
3. 将最终 packed/quantized weight 反量化回 FP16/BF16；
4. 保存反量化 weight，保留其 rounding/clipping 误差；
5. runtime 不再次量化 weight；
6. activation 在 runtime 继续动态 fake MX QDQ。

选择“预 QDQ weight”而不是在 runtime 加载 `v`，原因是 `v` 可按 element
保存，体积接近原始 weight，而且最终推理只需要优化后的离散值。反量化权重
能最精确地复现算法结果，也符合“只模拟误差、不需要真实压缩”的目标。

## 十、逐文件修改清单

### 新增

- `vllm_ascend/quantization/fake_mx.py`
  - `_round_e2m1`
  - `_round_e4m3fn`
  - `fake_mx_quantize`
  - `randomized_hadamard_transform`
- `vllm_ascend/quantization/methods/fake_mx.py`
  - RTN Linear/MoE
  - OmniQuant Linear/MoE
  - RHT Linear/MoE
  - FlatQuant Linear
  - AutoRound Linear/MoE
- `tests/ut/quantization/test_fake_mx.py`
- `tests/ut/quantization/test_fake_mx_registry.py`
- `examples/quantization/fake_mx/*`

### 修改

- `vllm_ascend/quantization/methods/__init__.py`
  - import fake schemes，触发 registry 装饰器
- `vllm_ascend/quantization/modelslim_config.py`
  - `FAKE_MX_QUANT_TYPES`
  - `_get_fake_mx_quant_type`
  - `get_linear_quant_type`
  - `is_layer_skipped_ascend`
  - default 和 glob override
- `vllm_ascend/quantization/method_adapters.py`
  - 保留 v0.23 的 `layer_type` 传递，使 FlatQuant RowParallel 正确建矩阵
- `vllm_ascend/ops/fused_moe/moe_stage_params.py`
  - `MoEQuantParams.fake_mx_*`
- `vllm_ascend/ops/fused_moe/moe_runtime_args.py`
  - `build_fused_experts_input`
  - fake algorithm/RHT 元数据传递
- `vllm_ascend/ops/fused_moe/moe_mlp.py`
  - `unquant_apply_mlp`
  - `unified_apply_mlp`
  - SwiGLU 后的 RHT 和 QDQ
- `tests/ut/ops/test_moe_runtime_args.py`
- `tests/ut/quantization/test_modelslim_config.py`

## 十一、配置和产物防误用

`fake_mx_weight_state` 是强校验字段：

| 值 | 允许的 scheme | weight 内容 |
|---|---|---|
| 不设置 | RTN fake | 原始 FP weight |
| `flatquant_transformed_fp` | FlatQuant fake | 变换后、未 QDQ FP weight |
| `rht_rotated_fp` | RHT fake | RHT 后、未 QDQ FP weight |
| `hadamard_learning_transformed_fp` | Hadamard Learning fake | `inverse(T).T` 变换后、未 QDQ FP weight及学习矩阵 |
| `prequantized_qdq` | OmniQuant/AutoRound fake | 已写入 QDQ 误差的 FP weight |

该检查防止两类高风险错误：

- 将原始 FP checkpoint 用 OmniQuant/AutoRound scheme 加载，却没有任何
  weight 量化误差；
- 将已经 QDQ 的 weight 再做一次 RTN QDQ，得到不可解释的二次误差。

## 十二、测试计划

1. 纯 CPU/Torch 数值测试：
   - E2M1/E4M3 level；
   - ties-to-even；
   - E8M0 scale；
   - padding；
   - tensor clip ratio；
   - RHT norm preservation；
   - NaN/Inf；
2. registry/config 测试：
   - 全部 scheme 的 Linear/MoE 注册；
   - packed Qwen projection；
   - exact > glob > default；
   - `FLOAT` skip；
   - weight state mismatch；
3. MoE contract 测试：
   - fake metadata 从 scheme 到 GMM2；
   - `QuantType.NONE`；
   - RHT w2 signs；
   - LoRA/SwiGLU limit 字段不丢失；
4. Ascend 远端测试：
   - vLLM v0.23.0 + vllm-ascend v0.23.0rc1；
   - CANN 环境；
   - 单卡小 tensor Linear/GMM；
   - Qwen3.5-9B PPL/生成一致性；
   - Qwen3.5-35B-A3B MoE top-k、logits 和生成一致性；
5. 结果报告必须把“数值精度”与“真实性能/显存”分栏，fake 路径只报告前者。
