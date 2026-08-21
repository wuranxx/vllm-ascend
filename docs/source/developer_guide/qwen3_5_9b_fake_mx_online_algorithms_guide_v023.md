# Qwen3.5-9B Fake-MX 量化验证指南

适用版本：vLLM v0.23.0、vllm-ascend v0.23.0rc1。

当前方案使用原始 BF16/FP16 模型，在浮点 Tensor 中模拟 MXFP4/MXFP8 的
Quantize-Dequantize 误差，用于验证 RTN、RHT、LHT 和 FlatQuant 的模型精度。
它不生成低比特 packed weight，也不代表原生 MX kernel 的性能。

## 1. 执行模型

```text
原始 BF16/FP16 weight
  -> 可选算法变换
  -> Fake-MX weight QDQ（模型加载后执行一次）
  -> 以 BF16/FP16 保存带量化误差的数值

每次 Linear forward
  -> 可选 activation 变换
  -> Fake-MX activation QDQ
  -> 普通 F.linear
```

公共 QDQ 支持两种格式：

| 配置 | 模拟格式 | element | shared scale |
|---|---|---|---|
| `W4A4_MXFP4_*` | MXFP4 | E2M1 | 每组一个 E8M0 scale |
| `W8A8_MXFP8_*` | MXFP8 | E4M3 | 每组一个 E8M0 scale |

默认 `group_size=32`，最后一维必须能被 group size 整除。

## 2. 代码结构与调用链

| 文件 | 职责 |
|---|---|
| `vllm_ascend/quantization/modelslim_config.py` | 解析配置并按 module prefix 选择 scheme |
| `vllm_ascend/quantization/methods/registry.py` | 注册和创建 quant scheme |
| `vllm_ascend/ops/linear.py` | vLLM Linear 到 Ascend quant method 的适配 |
| `vllm_ascend/quantization/methods/fake_mx.py` | 算法变换、weight QDQ、activation QDQ |
| `vllm_ascend/quantization/fake_mx.py` | MXFP4/MXFP8 数值模拟和非 Linear 量化开关 |
| `vllm_ascend/quantization/kernels/fake_mx.py` | 可选融合 QDQ kernel 适配入口 |

真实调用链：

```text
quant_model_description.json
  -> ModelSlimConfig.get_quant_method(layer, prefix)
  -> create_scheme_for_layer(...)
  -> _get_fake_mx_quant_type(...)
  -> AscendLinearMethod(Fake-MX scheme)

模型权重加载完成
  -> scheme.process_weights_after_loading(layer)
  -> algorithm weight transform
  -> fake_mx_quantize(weight)

Linear forward
  -> scheme.apply(layer, x, bias)
  -> transform_activation(layer, x)
  -> fake_mx_quantize(x)
  -> F.linear(...)
```

## 3. Qwen3.5-9B 量化节点

### 3.1 Attention Linear

| 模型逻辑节点 | vLLM 物理 Linear | 配置 pattern |
|---|---|---|
| Full Attention Q/K/V | `self_attn.qkv_proj` | `*self_attn.qkv_proj*` |
| Full Attention output | `self_attn.o_proj` | `*self_attn.o_proj*` |
| GDN Q/K/V/Z、B/A | `linear_attn.in_proj_qkvz`、`in_proj_ba` | `*linear_attn.in_proj*` |
| GDN output | `linear_attn.out_proj` | `*linear_attn.out_proj*` |

### 3.2 Dense MLP

| 节点 | 配置 pattern |
|---|---|
| Fused gate/up projection | `*mlp.gate_up_proj*` |
| Down projection | `*mlp.down_proj*` |

`module_quant_overrides` 按 JSON 插入顺序匹配，第一个命中的 pattern 生效。
配置应从具体 pattern 写到宽 pattern，并以 `"*": "FLOAT"` 收尾。

### 3.3 Attention/GDN 非 Linear 开关

`fake_mx_quant_targets` 只控制无法由 Linear/MoE scheme 表达的额外节点，目前仅支持：

| target | 插入点 | 建议 |
|---|---|---|
| `attn-cache` | Norm/RoPE 后的 Q/K/V、fused attention/cache 前 | 单独消融 |

Attention/GDN/MLP/MoE Linear 的范围、算法和位宽完全由
`module_quant_overrides` 决定，不再使用 `attn-linear` target。`attn-cache` 只控制
额外的标准 Attention Q/K/V 非 Linear 激活边界；GDN core 保持浮点。

`attn-cache` 当前会 QDQ Q/K/V，但 fused Attention 不暴露 softmax 后的 P，
因此不包含 P/V MatMul 前的 P QDQ。

对应插入代码：

- `vllm_ascend/patch/worker/patch_qwen3_5.py`：Attention Q/K/V。

## 4. 支持的算法

### 4.1 RTN

| 项目 | 配置 |
|---|---|
| MXFP4 | `W4A4_MXFP4_FAKE` |
| MXFP8 | `W8A8_MXFP8_FAKE` |
| 参数文件 | 不需要 |

weight 和 activation 直接执行 Fake-MX QDQ，是其他算法的同范围基线。

### 4.2 RHT

| 项目 | 配置 |
|---|---|
| MXFP4 | `W4A4_MXFP4_RHT_FAKE` |
| MXFP8 | `W8A8_MXFP8_RHT_FAKE` |
| 参数 | `rht_seed`、`rht_group_size` |
| PTQ | 不需要 |

RHT 根据 seed 生成确定性随机 sign，对原始 weight 和 activation 执行匹配的
Randomized Hadamard Transform，再分别 QDQ。

### 4.3 LHT / Hadamard Learning

| 项目 | 配置 |
|---|---|
| MXFP4 | `W4A4_MXFP4_HADAMARD_LEARNING_FAKE` |
| MXFP8 | `W8A8_MXFP8_HADAMARD_LEARNING_FAKE` |
| 参数文件 | `lht_params_path`，必填 |
| 矩阵大小 | `hadamard_learning_matrix_size`，默认 128 |

AMCT PTQ 学习每个目标 Linear 的变换矩阵 `T`：

```text
activation: X' = X @ T
weight:     W' = W @ inverse(T).T
```

safetensors 中每个启用节点需要：

```text
<physical-layer-prefix>.transform_weight
```

### 4.4 FlatQuant

| 项目 | 配置 |
|---|---|
| MXFP4 | `W4A4_MXFP4_FLATQUANT_FAKE` |
| MXFP8 | `W8A8_MXFP8_FLATQUANT_FAKE` |
| 参数文件 | `flatquant_params_path`，必填 |
| 矩阵大小 | `flatquant_matrix_size`，默认 128 |
| diagonal scale | `flatquant_use_diag`，默认 true |
| 最大 TP | `max_supported_tp`，默认 4 |

当前实现采用：

```text
activation: X' = L.T @ X @ R
            X' = X' * diag

weight:     W' = inverse(L) @ W @ inverse(R).T
            W' = W' / diag
```

每个启用节点的 safetensors 参数：

```text
<physical-layer-prefix>.left_trans
<physical-layer-prefix>.right_trans
<physical-layer-prefix>.diag_scale
```

当 `flatquant_use_diag=false` 时可以省略 `diag_scale`。

## 5. 配置模板

### 5.1 Attention Linear uniform W4A4 RTN

```json
{
  "default_quant_type": "W4A4_MXFP4_FAKE",
  "fake_mx_quant_targets": [],
  "group_size": 32,
  "module_quant_overrides": {
    "*self_attn.qkv_proj*": "W4A4_MXFP4_FAKE",
    "*self_attn.o_proj*": "W4A4_MXFP4_FAKE",
    "*linear_attn.in_proj*": "W4A4_MXFP4_FAKE",
    "*linear_attn.out_proj*": "W4A4_MXFP4_FAKE",
    "*": "FLOAT"
  }
}
```

### 5.2 MLP-only W4A4 RTN

```json
{
  "default_quant_type": "W4A4_MXFP4_FAKE",
  "fake_mx_quant_targets": [],
  "group_size": 32,
  "module_quant_overrides": {
    "*mlp.gate_up_proj*": "W4A4_MXFP4_FAKE",
    "*mlp.down_proj*": "W4A4_MXFP4_FAKE",
    "*": "FLOAT"
  }
}
```

### 5.3 Attention Linear mixed RHT

```json
{
  "default_quant_type": "W8A8_MXFP8_RHT_FAKE",
  "fake_mx_quant_targets": [],
  "group_size": 32,
  "rht_group_size": 32,
  "rht_seed": 2026,
  "module_quant_overrides": {
    "*self_attn.qkv_proj*": "W4A4_MXFP4_RHT_FAKE",
    "*self_attn.o_proj*": "W8A8_MXFP8_RHT_FAKE",
    "*linear_attn.in_proj*": "W4A4_MXFP4_RHT_FAKE",
    "*linear_attn.out_proj*": "W8A8_MXFP8_RHT_FAKE",
    "*": "FLOAT"
  }
}
```

AMCT uniform W4A4 对纳入范围内的 projection 全部使用 W4A4。mixed 配置是
输入 projection W4A4、敏感 output projection W8A8 的独立精度策略。

完整配置位于：

```text
examples/quantization/fake_mx/w4a4_eval/configs/
```

## 6. 启动

1. 保持模型目录中的原始 BF16/FP16 safetensors 不变；
2. 将实验 JSON 复制为模型目录下的 `quant_model_description.json`；
3. LHT/FlatQuant 配置正确的参数文件路径；
4. 启动服务：

```bash
export PYTHONPATH=/path/to/vllm-ascend:/path/to/vllm:${PYTHONPATH}
ASCEND_RT_VISIBLE_DEVICES=0 vllm serve /path/to/Qwen3.5-9B \
  --quantization ascend \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len 16384 \
  --mamba-ssm-cache-dtype bfloat16
```

## 7. 验证要求

- 检查日志中的 scheme 和物理 module prefix 是否命中预期配置；
- LHT/FlatQuant 参数缺失、key 缺失或 shape 不匹配必须直接失败；
- 对 LHT/FlatQuant 先验证关闭 QDQ 时的正逆变换等价性；
- 每种算法都使用相同 scope 和位宽的 RTN 作为对照；
- 固定模型、数据集、prompt、seed、sampling、TP、上下文长度和 `max_tokens`；
- prefill/decode 使用同一 Linear scheme；Attention/GDN core 的执行分支不同；
- MLP-only 分数接近 BF16 不代表没有生效，应以 scheme 日志和同范围 A/B 判断。

## 8. 结果口径

当前可引用的 MATH-500 结果：

| 配置 | 分数 | 说明 |
|---|---:|---|
| BF16 | 93.8% | 浮点基线 |
| Attention Linear RTN W4A4 | 77.4% | uniform W4A4 |
| MLP-only RTN W4A4 | 92.6% | Attention 保持 FLOAT |
| RHT mixed | 91.6% | 固定随机 RHT，无 PTQ |
| FlatQuant mixed | 92.6% | FlatQuant PTQ |
| LHT mixed | 93.0% | `learnable_had` PTQ |

mixed 结果包含 output projection W8A8 保护。进行算法归因时，必须使用相同
mixed 位宽和相同量化范围的 RTN 基线。

## 9. 当前范围

- 当前重点是 Qwen3.5-9B Dense Linear；
- `attn-cache` 作为独立精度消融，默认不启用；GDN core 保持浮点；
- Qwen3.5-35B MoE 的算法参数映射和端到端验证另行实现；
- 离线预变换/预量化 weight 自动加载不在当前流程；
- 融合 QDQ kernel 只能替换 `fake_mx_quantize()` 的执行，不改变量化节点。

Kernel 替换和新算法接入见
`fake_mx_kernel_and_algorithm_extension_guide_v023.md`。
