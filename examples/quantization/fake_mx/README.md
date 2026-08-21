# Fake MXFP4/MXFP8 validation

This path emulates MX quantization error on devices without native MXFP4 or
MXFP8 support. It never creates packed MX tensors. The default `reference`
backend uses ordinary floating-point operators, while an optional fused QDQ
kernel can replace that numerical implementation without moving quantization
boundaries:

1. Load the original FP16/BF16 checkpoint.
2. Round weights to the selected MX grid once in
   `process_weights_after_loading`.
3. Round activations to the selected MX grid on every forward.
4. Execute the ordinary floating-point linear or grouped-matmul kernel.

The stored parameter and activation dtypes remain FP16/BF16. The values include
the error from E2M1/E4M3 element rounding and one E8M0 power-of-two scale per
group.

## Configuration

Copy one of the sample files to the model directory as
`quant_model_description.json`, then run:

```bash
vllm serve /path/to/qwen3.5-model --quantization ascend
```

Supported scheme names:

- `W4A4_MXFP4_FAKE`: fake MXFP4 weights and activations.
- `W8A8_MXFP8_FAKE`: fake MXFP8 weights and activations.
- `W4A4_MXFP4_FLATQUANT_FAKE`: FlatQuant transform followed by fake MXFP4.
- `W8A8_MXFP8_FLATQUANT_FAKE`: FlatQuant transform followed by fake MXFP8.
- `W4A4_MXFP4_OMNIQUANT_FAKE` / `W8A8_MXFP8_OMNIQUANT_FAKE`.
- `W4A4_MXFP4_RHT_FAKE` / `W8A8_MXFP8_RHT_FAKE`.
- `W4A4_MXFP4_HADAMARD_LEARNING_FAKE` /
  `W8A8_MXFP8_HADAMARD_LEARNING_FAKE`: AMCT-Q learnable block transform
  followed by fake MX QDQ.
- `W4A4_MXFP4_AUTOROUND_FAKE` / `W8A8_MXFP8_AUTOROUND_FAKE`.

`default_quant_type` applies to modules without an explicit `*.weight` entry.
`module_quant_overrides` is evaluated in JSON insertion order; the first glob
matching the vLLM module prefix wins. An explicit per-weight entry has the
highest priority and can use `FLOAT` to skip a module.

### QDQ execution backend

`fake_mx_backend` controls how every enabled Fake MX node computes the same
quantize-dequantize contract. It is independent of `fake_mx_quant_targets` and
module precision overrides:

- `reference` (default): use the AMCT-compatible PyTorch golden implementation.
- `kernel`: require the optional fused QDQ kernel and fail immediately if it
  cannot execute the request.
- `auto`: use the fused kernel when supported, otherwise log once and fall back
  to `reference`.

```json
{
  "fake_mx_backend": "reference",
  "fake_mx_quant_targets": []
}
```

The optional operator adapter is
`vllm_ascend/quantization/kernels/fake_mx.py`. When the external kernel is
delivered, update only `_load_external_fake_mx_kernel()` to import its actual
single-input/single-output QDQ entry point and add finalized capability checks
to `fake_mx_kernel_support_reason()`. Existing Linear, Attention, GDN, and MoE
insertion points continue to call the stable `fake_mx_quantize()` wrapper.

### AMCT-compatible attention targets

`fake_mx_quant_targets` independently controls Qwen3.5 non-Linear attention
injection boundaries. It defaults to `[]` and currently accepts only
`"attn-cache"`:

- `attn-cache`: additionally fake-QDQ normalized/RoPE-applied Q/K/V immediately
  before the fused attention/cache boundary. This matches AMCT's Q/K/V operand
  placement, but the fused vLLM attention backend does not expose AMCT's
  post-softmax probability (P) fake-QDQ point.

Attention/GDN/MLP/MoE Linear operands and weights are selected exclusively by
`module_quant_overrides`; they do not require a target entry.

The MX element/shared-exponent math follows AMCT-Q: 32-element blocks by
default, shared exponent carry at mantissa `> 1.75`, minimum E8M0 exponent
`-127`, and half-away-from-zero element rounding. The last tensor dimension
must be divisible by `group_size`, as required by AMCT's `unflatten` contract.

The Qwen direct-conversion samples use W4A4 MXFP4 for attention/GDN
projections, Dense MLP, shared experts, and routed experts. Embeddings, the
visual tower, router gates, and LM head remain floating point. Full-attention
Q/K/V remain floating point unless `attn-cache` is explicitly enabled. The
projected GDN `mixed_qkv` remains floating point, matching AMCT `attn-linear`.

## FlatQuant checkpoint contract

Fake FlatQuant is a linear-only validation path. Each enabled linear layer must
provide floating-point, FlatQuant-transformed `weight` plus `left_trans`,
`right_trans`, and `clip_ratio` tensors from calibration. The runtime applies:

```text
x -> left_trans @ reshape(x) @ right_trans
  -> block clipping
  -> fake MX QDQ
  -> floating-point GEMM with the fake-MX-QDQ transformed weight
```

Do not select a FlatQuant fake scheme for a plain pretrained checkpoint that
does not contain these transform parameters. Routed MoE FlatQuant is not
implemented because the current Ascend FlatQuant contract is linear-only.

## Algorithm checkpoint contracts

The algorithm examples are not drop-in configs for an untouched pretrained
checkpoint. Each algorithm expects specific calibration artifacts:

- **FlatQuant**: `flatquant_params_path` must point to a safetensors file
  containing `left_trans`, `right_trans`, `clip_ratio`, and optional
  `diag_scale` per enabled layer. The runtime loads these, applies the inverse
  transform to the weight, and applies the forward transform to activations.
- **RHT**: No external params needed. The runtime generates random signs from
  `rht_seed` and rotates both weight and activations at load/forward time.
- **Hadamard Learning (LHT)**: `lht_params_path` must point to a safetensors
  file containing `transform_weight` (K×K matrix) per enabled layer. The
  runtime loads the matrix, applies `inv(T).T` to the weight, and applies
  `x @ T` to activations.
- **OmniQuant / AutoRound**: `fake_mx_weight_state: "prequantized_qdq"` marks
  a checkpoint that already contains the final MX QDQ error. The runtime
  deliberately skips a second weight QDQ.

See
`docs/source/developer_guide/fake_mx_algorithm_adaptation_v023.md` for the
offline/runtime boundary and exact insertion points.

## Scope and limitations

- This validates numerical accuracy, perplexity, and task metrics, not MX
  kernel performance or packed-checkpoint memory use.
- Router logits are computed from the original activation. Expert FC1 input,
  expert weights, and the post-activation FC2 input receive fake-MX error.
- Fake-MX routed MoE requires the split dispatch/GMM1/SwiGLU/GMM2/combine
  path. Monolithic FusedMC2 is rejected because it has no GMM1/GMM2
  intermediate QDQ insertion point.
- The implementation uses only ordinary floating-point tensors and arithmetic,
  so it does not require `torch.float4`, `torch.float8`, or E8M0 dtypes.
- `group_size` defaults to 32, matching OCP MX formats.

See
`docs/source/developer_guide/qwen3_5_a4w4_mxfp_hadamard_flatquant_validation_v023.md`
for Qwen3.5 attention, prefill/decode, Hadamard, FlatQuant, and MoE details.
Hadamard Learning 的训练语义、导出映射和逐 expert 插入点见
`docs/source/developer_guide/qwen3_5_hadamard_learning_fake_mx_v023.md`。
