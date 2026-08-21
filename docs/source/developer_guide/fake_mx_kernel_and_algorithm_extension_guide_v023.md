# Fake-MX Kernel 替换与新算法接入指南

适用版本：vLLM v0.23.0、vllm-ascend v0.23.0rc1。

本文说明两类扩展：用融合算子替换 Fake-MX QDQ reference 实现，以及增加新的
Linear 量化算法。扩展应保持量化节点和配置层不变。

## 1. 扩展边界

```text
模型/算子插入点
  -> fake_mx_quantize(tensor, format, group_size, ...)
  -> reference 或 kernel backend
  -> 与输入相同 shape/device/dtype 的反量化 Tensor
```

Kernel 只负责一个 Tensor 的 MX QDQ。Linear、Attention、GDN 和 MoE 不应直接
导入外部 kernel。算法只负责 QDQ 前的变换/裁剪参数，不应复制 MX 编码逻辑。

## 2. 当前 Kernel 接口

### 2.1 公共分发

文件：`vllm_ascend/quantization/fake_mx.py`

`fake_mx_quantize()` 根据 `fake_mx_backend` 分发：

| backend | 行为 |
|---|---|
| `reference` | 使用 PyTorch golden 实现 |
| `kernel` | 必须使用融合 kernel，不支持时立即报错 |
| `auto` | kernel 支持时使用，否则回退 reference |

配置示例：

```json
{
  "fake_mx_backend": "kernel",
  "group_size": 32
}
```

### 2.2 Kernel 适配层

文件：`vllm_ascend/quantization/kernels/fake_mx.py`

需要实现/调整的函数：

| 函数 | 职责 |
|---|---|
| `_load_external_fake_mx_kernel()` | 导入外部单入口算子 |
| `fake_mx_kernel_support_reason()` | 检查 device、dtype、shape、格式、group、clip 能力 |
| `fake_mx_quantize_kernel()` | 参数转换、调用算子、校验输出契约 |

外部入口建议保持：

```python
def fake_mx_qdq(
    tensor,
    mx_format: str,
    group_size: int,
    clip_ratio: float = 1.0,
):
    ...
```

返回值必须满足：

- shape 与输入完全一致；
- device 与输入一致；
- dtype 与输入一致；
- 输出为反量化后的浮点值，不是 packed FP4/FP8；
- 不原地修改输入，除非接口明确允许且调用方已验证生命周期。

## 3. Kernel 数值契约

reference 位于 `vllm_ascend/quantization/fake_mx.py`，是对拍标准：

- 沿最后一维按 `group_size` 分块；
- MXFP4 element 使用 E2M1；
- MXFP8 element 使用有限 E4M3；
- 每组使用 E8M0 power-of-two shared scale；
- element rounding 使用 half-away-from-zero；
- mantissa 超过 1.75 时处理指数 carry；
- 最小 shared exponent 为 -127；
- 正确处理 0、NaN、Inf、饱和和 subnormal；
- 最后一维不可整除 group size 时行为与 reference 一致。

Kernel 支持检查应 fail-fast，不允许因为输入不支持而返回未经 QDQ 的原值。

## 4. Kernel 替换步骤

1. 在 `_load_external_fake_mx_kernel()` 中导入实际函数；
2. 将算子支持矩阵写入 `fake_mx_kernel_support_reason()`；
3. 在 `fake_mx_quantize_kernel()` 中适配 format enum、group size 和 clip 参数；
4. 对随机、边界和异常值 Tensor 与 reference 对拍；
5. 使用 `fake_mx_backend=kernel` 运行，确保没有 reference 回退；
6. 分别验证 weight QDQ、prefill activation、decode activation；
7. 再运行模型数据集精度与性能测试。

对拍工具：

```bash
python examples/quantization/fake_mx/validation/compare_fake_mx_kernel.py \
  --kernel-entry my_package.fake_mx:fake_mx_qdq \
  --format mxfp4 \
  --group-size 32
```

至少覆盖：BF16/FP16、MXFP4/MXFP8、连续/非连续输入、小值、最大值、全零、
NaN/Inf、不同 batch/token 数和多次重复调用。

## 5. Kernel 替换不应修改的文件

纯 QDQ kernel 替换通常不需要修改：

- `vllm_ascend/quantization/methods/fake_mx.py`；
- `vllm_ascend/patch/worker/patch_qwen3_5.py`；
- `vllm_ascend/ops/gdn.py`；
- Qwen3.5 模型 forward；
- 各算法 JSON。

这些位置已经统一调用 `fake_mx_quantize()` wrapper。

## 6. 新 Linear 算法接入

算法适配类放在 `vllm_ascend/quantization/methods/fake_mx.py`。推荐继承
`_AscendFakeMXLinearMethod`。

### 6.1 最小实现

```python
class _AscendNewAlgorithmFakeMXLinearMethod(_AscendFakeMXLinearMethod):
    algorithm = "new_algorithm"

    def __init__(self):
        super().__init__()
        config = _quant_description()
        self.params_path = config.get("new_algorithm_params_path")

    def get_pertensor_param(self, params_dtype, **kwargs):
        return {"transform": ...}

    def process_weights_after_loading(self, layer):
        # 加载参数；对原始 weight 执行匹配变换
        # 最后调用 super() 完成一次性 Fake-MX weight QDQ
        super().process_weights_after_loading(layer)

    def transform_activation(self, layer, x):
        # 返回 QDQ 前的 activation
        return transformed_x
```

分别注册 MXFP4/MXFP8：

```python
@register_scheme("W4A4_MXFP4_NEW_ALGO_FAKE", "linear")
class AscendW4A4NewAlgo(_AscendNewAlgorithmFakeMXLinearMethod):
    mx_format = "mxfp4"

@register_scheme("W8A8_MXFP8_NEW_ALGO_FAKE", "linear")
class AscendW8A8NewAlgo(_AscendNewAlgorithmFakeMXLinearMethod):
    mx_format = "mxfp8"
```

并将两个名称加入 `modelslim_config.py::FAKE_MX_QUANT_TYPES`。

### 6.2 参数加载

复用以下公共函数：

| 函数 | 用途 |
|---|---|
| `_resolve_model_artifact()` | 解析绝对路径或相对模型目录路径 |
| `_load_transform_params()` | 缓存加载 safetensors |
| `_layer_prefix_candidates()` | 映射 AMCT logical key 与 vLLM physical prefix |
| `_copy_transform_param()` | key、shape、dtype/device 校验 |
| `_inverse_fp32()` | 用 FP32 solve 计算逆矩阵 |

参数不完整时必须报错，不使用 identity/random 参数继续评测。

### 6.3 算法数学要求

如果算法使用可逆变换，必须先验证关闭 QDQ 时：

```text
F.linear(transform_x(x), transform_w(W)) ≈ F.linear(x, W)
```

然后再验证：

- weight QDQ 只执行一次；
- activation QDQ 每次 forward 执行；
- TP rank 的 transform/sign 切分一致；
- fused qkv/qkvz/ba 的 logical shard 使用一致变换；
- 算法切换只改 scheme 和参数文件，不改模型 forward。

## 7. 新量化节点接入

如果节点是普通 Linear，只需配置 `module_quant_overrides`。

如果目标在融合算子内部：

1. 在最靠近数学 operand 的 vllm-ascend backend 位置插入 hook；
2. 通过显式 target 开关控制；
3. 调用 `maybe_fake_mx_quantize_activations()` 或 `fake_mx_quantize()`；
4. 默认关闭新节点，先做单点消融；
5. 不把算法判断散落进模型 forward。

现有示例：

- Attention Q/K/V：`patch/worker/patch_qwen3_5.py`；
- GDN core 输入：`ops/gdn.py`；
- Linear：`quantization/methods/fake_mx.py`。

## 8. 验收清单

- reference/kernel 单 Tensor 对拍通过；
- `kernel` 模式无静默回退；
- 新 scheme 能被 ModelSlimConfig 解析；
- 原始 BF16 权重加载后只变换/QDQ 一次；
- 关闭 QDQ 的算法等价性通过；
- TP=1 和目标 TP 均通过；
- prefill/decode 均走到 kernel；
- 相同 scope/bit policy 的 RTN 与算法结果可比较；
- kernel 性能和模型精度分别报告。
