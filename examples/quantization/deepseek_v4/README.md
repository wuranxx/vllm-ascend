# DeepSeek-V4-Flash W8A8 Quantization

Convert DeepSeek-V4-Flash native FP8 weights to W8A8 quantized weights in ModelSlim format, loadable by vllm-ascend on Ascend NPU without `--quantization` flag.

## Prerequisites

- **Hardware**: Ascend NPU (Atlas 800I A2 / Atlas A3 Inference series)
- **Software**:
  - PyTorch + torch_npu (matching versions, both must be installed)
  - vllm-ascend (for loading the quantized weights)
- **Input**: DeepSeek-V4-Flash FP8 HF weights (from [deepseek-ai/DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash))

## Usage

```bash
python examples/quantization/deepseek_v4/quantize.py \
    --input_fp8_hf_path /path/to/DeepSeek-V4-Flash \
    --output_hf_path /path/to/DeepSeek-V4-Flash-w8a8
```

By default, the script uses `quantize_config.yaml` next to it. To use a custom config:

```bash
python examples/quantization/deepseek_v4/quantize.py \
    --input_fp8_hf_path /path/to/DeepSeek-V4-Flash \
    --output_hf_path /path/to/DeepSeek-V4-Flash-w8a8 \
    --config /path/to/custom_config.yaml
```

## Output Format

The output directory contains ModelSlim-format quantized weights:

| File | Description |
|------|-------------|
| `quant_model_weights-{N:05d}-of-{M:05d}.safetensors` | Sharded quantized weights (~4GB per shard) |
| `quant_model_weights.safetensors.index.json` | Shard index with `weight_map` and `total_size` |
| `quant_model_description.json` | Per-weight quant type (`W8A8_DYNAMIC` or `FLOAT`) |
| `config.json` | Model config with `quantization_config` removed |
| `*.py`, `*.json`, `*.jinja` | Copied from input (tokenizer, modeling code, etc.) |

### How vllm-ascend Loads the Output

1. `config.json` has no `quant_method` field
2. vllm-ascend's `AscendModelSlimConfig.override_quantization_method()` detects this on NPU and auto-selects the ascend path
3. `maybe_update_config()` loads `quant_model_description.json` to determine per-layer quantization scheme
4. No `--quantization` flag needed when serving:

```bash
vllm serve /path/to/DeepSeek-V4-Flash-w8a8
```

## Quantization Config (YAML)

The `quantize_config.yaml` declares which layers to quantize via `include`/`exclude` glob patterns:

- **`*attn*`** (exclude `wq_a`, `wkv`, `wo_a`, compressor, indexer) — quantize `wq_b`, `wo_b`
- **`*ffn*`** (exclude `*gate`) — quantize expert and shared_expert weights
- **`*e_proj`, `*h_proj`** — quantize MTP projection layers

Each quantized weight produces three tensors:
- `.weight` (int8) — quantized weight
- `.weight_scale` (float32) — per-channel scale
- `.weight_offset` (float32) — per-channel offset (zeros for symmetric quant)

## How It Works

```
FP8 weight + .scale ──decode_fp8──▶ BF16 weight
                                          │
                              YAML match + dim==2?
                                    ├── Yes ── weight_quant_sym_perchannel ──▶ int8 + scale + offset (W8A8_DYNAMIC)
                                    └── No  ── keep original (FLOAT)
```

Quantization logic references [msmodelslim](https://gitee.com/ascend/msit/tree/master/msmodelslim):
- FP8 dequant: `decode_fp8` / `decode_fp4` (128×128 block)
- W8A8 weight quant: per-channel symmetric (`clamp([-128, 127])`)
- Output sharding: 4GB per shard (BufferedSafetensorsWriter)
