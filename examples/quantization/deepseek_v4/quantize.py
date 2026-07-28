"""DeepSeek-V4-Flash W8A8 quantization script.

Converts native FP8 weights to W8A8 quantized weights in ModelSlim format,
loadable by vllm-ascend without --quantization flag (auto-detected on NPU).

Quantization logic references msmodelslim:
- FP8 dequant: msmodelslim/model/deepseek_v4/convert_fp8_to_bf16.py
- W8A8 weight quant: msmodelslim/pytorch/llm_ptq/llm_ptq_tools/flat_quant/components/quantizer.py
- Output format: msmodelslim/pytorch/llm_ptq/llm_ptq_tools/save/

Script form (argparse, shard reading, memory management) references:
- cann-recipes-infer/models/deepseek_v4/utils/convert_model.py
"""

import argparse
import fnmatch
import json
import os
import shutil
from glob import glob

import torch
import torch_npu  # noqa: F401  required for tensor.npu()
import yaml
from safetensors.torch import load_file, save_file
from tqdm import tqdm

# W8A8 symmetric per-channel quantization constants
NUM_BITS = 8
Q_MAX = 2 ** (NUM_BITS - 1) - 1  # 127
Q_MIN = -(2 ** (NUM_BITS - 1))  # -128
# Epsilon to prevent division by zero when the input is all zeros
QUANT_EPSILON = 1e-5

# FP8 block size for dequantization (DeepSeek-V4 uses 128x128 blocks)
FP8_BLOCK_SIZE = 128
# MXFP4 block size: 32 FP4 elements (packed into 16 uint8) share 1 block scale
MXFP4_BLOCK_SIZE = 32

# Output shard size (GB), matches YAML save.ascendv1_saver.part_file_size
OUTPUT_SHARD_GB = 4
ONE_GB_BYTES = 1073741824

# Max number of input safetensors shards to keep in memory simultaneously.
# FP8 weight and its scale may land in adjacent shards, so a window of 2 is
# sufficient for the common case; get_tensor() may pull in extra scale shards
# which are evicted by the while-loop after each iteration.
MAX_CACHED_SHARDS = 2

# Quant type strings for quant_model_description.json
QUANT_TYPE_W8A8_DYNAMIC = "W8A8_DYNAMIC"
QUANT_TYPE_FLOAT = "FLOAT"


def decode_fp8(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize FP8 e4m3 weight to bfloat16.

    References: msmodelslim/model/deepseek_v4/convert_fp8_to_bf16.py:decode_fp8
    """
    weight = weight.unflatten(0, (-1, FP8_BLOCK_SIZE)).unflatten(-1, (-1, FP8_BLOCK_SIZE)).float()
    weight = weight * scale[:, None, :, None].float()
    return weight.flatten(2, 3).flatten(0, 1).bfloat16()


def decode_fp4(packed_fp4_data: torch.Tensor, block_scales: torch.Tensor) -> torch.Tensor:
    """Dequantize MXFP4 packed weight to bfloat16.

    References: msmodelslim/model/deepseek_v4/convert_fp8_to_bf16.py:decode_fp4
    """
    lut = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        device=packed_fp4_data.device,
        dtype=torch.float32,
    )

    uint8 = packed_fp4_data.view(torch.uint8)
    low = uint8 & 0x0F
    high = (uint8 >> 4) & 0x0F
    indices = torch.stack([low, high], dim=-1).flatten(-2)

    sign = 1.0 - 2.0 * ((indices >> 3) & 1).float()
    abs_idx = indices & 0x07
    values = sign * lut[abs_idx.long()]

    # MXFP4: 32 FP4 elements share 1 block scale, so repeat_interleave(32)
    # expands block_scales to match the unpacked weight dimension
    scales_expanded = block_scales.to(torch.float32).repeat_interleave(MXFP4_BLOCK_SIZE, dim=-1)
    return (values * scales_expanded).to(torch.bfloat16)


def weight_quant_sym_perchannel(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-channel symmetric W8A8 weight quantization.

    Returns (quant_weight[int8], weight_scale[fp32], weight_offset[fp32 zeros]).

    References: msmodelslim WeightQuantizer.find_params(sym=True) + sym_quant()
    """
    x = tensor.npu().flatten(1) if tensor.dim() > 2 else tensor.npu()

    tmp = torch.zeros(x.shape[0], device=x.device)
    xmin = torch.minimum(x.min(1)[0], tmp)
    xmax = torch.maximum(x.max(1)[0], tmp)

    # symmetric: take max of abs(xmin) and xmax, clamp to avoid div-by-zero
    xmax = torch.maximum(torch.abs(xmin), xmax).clamp(min=QUANT_EPSILON)
    scale = (xmax / Q_MAX).unsqueeze(-1)  # (M, 1) for broadcasting with x (M, N)
    zero = torch.zeros_like(scale)

    quant_weight = torch.clamp(torch.round(x / scale), Q_MIN, Q_MAX).to(torch.int8)
    return quant_weight.cpu(), scale.cpu().to(torch.float32), zero.cpu().to(torch.float32)


def match_yaml_rules(weight_name: str, yaml_rules: list) -> bool:
    """Check if weight_name matches any linear_quant rule in YAML config.

    A rule matches if weight_name matches any include pattern AND does not
    match any exclude pattern. Only `linear_quant` type rules are considered.
    """
    for rule in yaml_rules:
        if rule.get("type") != "linear_quant":
            continue
        includes = rule.get("include", [])
        excludes = rule.get("exclude", [])
        if any(fnmatch.fnmatch(weight_name, pat) for pat in includes):
            if not any(fnmatch.fnmatch(weight_name, pat) for pat in excludes):
                return True
    return False


class BufferedSafetensorsWriter:
    """Buffered safetensors writer with 4GB shard flushing.

    Simplified from msmodelslim BufferedSafetensorsWriter.
    Accumulates tensors in memory; flushes to a new shard when exceeding
    OUTPUT_SHARD_GB. On close, renames shards to {N:05d}-of-{M:05d} format
    and writes the index.json.
    """

    def __init__(self, save_directory: str, save_prefix: str = "quant_model_weights"):
        self.save_directory = save_directory
        self.save_prefix = save_prefix
        self.max_size = OUTPUT_SHARD_GB * ONE_GB_BYTES
        self.wait_save_keys: dict[str, torch.Tensor] = {}
        self.saved_keys_map: dict[str, str] = {}
        self.quant_description: dict[str, str] = {
            "version": "1.0.0",
            "model_quant_type": QUANT_TYPE_W8A8_DYNAMIC,
            "group_size": 0,
            "metadata": {},
        }
        self.total_size = 0
        self._wait_save_size = 0
        self._save_count = 0

    def write(self, name: str, tensor: torch.Tensor, quant_type: str) -> None:
        """Write a tensor and record its quant type in description."""
        tensor = tensor.detach().cpu().contiguous()
        tensor_size = tensor.numel() * tensor.element_size()

        if self._wait_save_size + tensor_size >= self.max_size:
            self._flush()

        self.wait_save_keys[name] = tensor
        self.quant_description[name] = quant_type
        self.total_size += tensor_size
        self._wait_save_size += tensor_size

    def _flush(self) -> None:
        if not self.wait_save_keys:
            return
        self._save_count += 1
        file_name = f"{self.save_prefix}-{self._save_count:05d}-of-00000.safetensors"
        file_path = os.path.join(self.save_directory, file_name)
        save_file(self.wait_save_keys, file_path, metadata={"format": "pt"})
        self.saved_keys_map.update({k: file_name for k in self.wait_save_keys})
        self.wait_save_keys.clear()
        self._wait_save_size = 0

    def close(self) -> None:
        """Flush remaining, rename shards, write index.json and description."""
        self._flush()

        # Rename shards: -of-00000 -> -of-{M:05d}
        for i in range(self._save_count):
            shard_idx = i + 1
            src_name = f"{self.save_prefix}-{shard_idx:05d}-of-00000.safetensors"
            src = os.path.join(self.save_directory, src_name)
            dst_name = f"{self.save_prefix}-{shard_idx:05d}-of-{self._save_count:05d}.safetensors"
            shutil.move(src, os.path.join(self.save_directory, dst_name))
            # Update weight_map: replace src_name with dst_name for keys in this shard
            for key in self.saved_keys_map:
                if self.saved_keys_map[key] == src_name:
                    self.saved_keys_map[key] = dst_name

        # Write index.json
        index = {
            "metadata": {"total_size": self.total_size},
            "weight_map": self.saved_keys_map,
        }
        index_path = os.path.join(self.save_directory, f"{self.save_prefix}.safetensors.index.json")
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

        # Write quant_model_description.json
        desc_path = os.path.join(self.save_directory, "quant_model_description.json")
        with open(desc_path, "w") as f:
            json.dump(self.quant_description, f, indent=2)


def load_yaml_rules(config_path: str) -> list:
    """Load linear_quant rules from YAML config."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("spec", {}).get("process", [])


def main(input_fp8_hf_path: str, output_hf_path: str, config_path: str) -> None:
    """Convert FP8 weights to W8A8 quantized weights in ModelSlim format.

    Args:
        input_fp8_hf_path: Path to HF directory containing FP8 safetensors.
        output_hf_path: Path to output directory for quantized weights.
        config_path: Path to YAML quantization config.
    """
    assert torch.npu.is_available(), "NPU is required for W8A8 quantization. Ensure torch_npu is installed."

    os.makedirs(output_hf_path, exist_ok=True)

    # Read model index and config
    with open(os.path.join(input_fp8_hf_path, "model.safetensors.index.json")) as f:
        model_index = json.load(f)
    with open(os.path.join(input_fp8_hf_path, "config.json")) as f:
        config = json.load(f)

    weight_map = model_index["weight_map"]
    yaml_rules = load_yaml_rules(config_path)

    # Remove quantization_config so vllm-ascend auto-detects via override_quantization_method
    config.pop("quantization_config", None)

    writer = BufferedSafetensorsWriter(output_hf_path)

    # Cache for loaded safetensors files (keep last 2 to limit memory)
    loaded_files: dict[str, dict[str, torch.Tensor]] = {}

    def get_tensor(name: str) -> torch.Tensor:
        file_name = weight_map[name]
        if file_name not in loaded_files:
            loaded_files[file_name] = load_file(os.path.join(input_fp8_hf_path, file_name), device="cpu")
        return loaded_files[file_name][name]

    safetensor_files = sorted(glob(os.path.join(input_fp8_hf_path, "*.safetensors")))

    for safetensor_file in tqdm(safetensor_files, desc="Quantizing shards"):
        file_name = os.path.basename(safetensor_file)
        current_state_dict = load_file(safetensor_file, device="cpu")
        loaded_files[file_name] = current_state_dict

        for weight_name, weight in current_state_dict.items():
            # Skip FP8 scale files - consumed during dequantization
            if weight_name.endswith(".scale"):
                continue

            # Dequantize FP8 weights to BF16 first
            if weight.element_size() == 1:
                scale_name = weight_name.replace(".weight", ".scale")
                try:
                    scale_inv = get_tensor(scale_name)
                    if weight.dtype == torch.float8_e4m3fn:
                        weight = decode_fp8(weight, scale_inv)
                    elif weight.dtype in (torch.int8, torch.uint8):
                        # MXFP4 packed as int8/uint8
                        weight = decode_fp4(weight, scale_inv)
                    else:
                        raise ValueError(
                            f"Unexpected 1-byte dtype {weight.dtype} for {weight_name}; "
                            "expected float8_e4m3fn (FP8) or int8/uint8 (MXFP4 packed)."
                        )
                except KeyError:
                    print(f"Warning: Missing scale tensor for {weight_name}, keeping original")
                    writer.write(weight_name, weight, QUANT_TYPE_FLOAT)
                    continue

            # Determine quantization: YAML match + 2D tensor (Linear weights)
            base_name = weight_name.rsplit(".", 1)[0] if weight_name.endswith(".weight") else None
            should_quantize = base_name is not None and weight.dim() == 2 and match_yaml_rules(base_name, yaml_rules)

            if should_quantize:
                quant_weight, weight_scale, weight_offset = weight_quant_sym_perchannel(weight)
                writer.write(weight_name, quant_weight, QUANT_TYPE_W8A8_DYNAMIC)
                writer.write(
                    weight_name.replace(".weight", ".weight_scale"),
                    weight_scale,
                    QUANT_TYPE_W8A8_DYNAMIC,
                )
                writer.write(
                    weight_name.replace(".weight", ".weight_offset"),
                    weight_offset,
                    QUANT_TYPE_W8A8_DYNAMIC,
                )
            else:
                writer.write(weight_name, weight, QUANT_TYPE_FLOAT)

            # MTP shared weights: vLLM loads MTP as a separate model instance that
            # only accepts weights with "mtp.0." prefix. The embedding and head
            # are shared with the main model in training (tie_weight), but at
            # deployment each model instance needs its own copy in the safetensors
            # file. FP8 path auto-renames these in vLLM; W8A8 path does not, so
            # we duplicate them here (matches msmodelslim warp_mtp_model behavior).
            # clone() is required because safetensors rejects tensors sharing the
            # same storage in the same shard.
            if weight_name == "embed.weight":
                writer.write("mtp.0.emb.tok_emb.weight", weight.clone(), QUANT_TYPE_FLOAT)
            elif weight_name == "head.weight":
                writer.write("mtp.0.head.weight", weight.clone(), QUANT_TYPE_FLOAT)

        # Memory management: keep only the most recently used shards.
        # Use while (not if) because get_tensor() may pull in multiple scale
        # shards beyond the current one, so we evict until we're back at the limit.
        while len(loaded_files) > MAX_CACHED_SHARDS:
            oldest = next(iter(loaded_files))
            del loaded_files[oldest]

    writer.close()

    # Post-quantization sanity checks (design.md risk 1: YAML glob matching)
    # Only count weight entries (exclude top-level metadata like model_quant_type)
    w8a8_weight_suffixes = (".weight", ".weight_scale", ".weight_offset")
    w8a8_count = sum(
        1
        for k, v in writer.quant_description.items()
        if v == QUANT_TYPE_W8A8_DYNAMIC and k.endswith(w8a8_weight_suffixes)
    )
    assert w8a8_count > 0, "No weights were quantized — check YAML config include patterns"
    assert w8a8_count % 3 == 0, (
        f"W8A8_DYNAMIC weight entry count ({w8a8_count}) must be a multiple of 3 "
        "(weight + weight_scale + weight_offset per quantized layer)"
    )

    # Write config.json (without quantization_config)
    with open(os.path.join(output_hf_path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Copy non-safetensors files (tokenizer, modeling code, etc.)
    # Skip config.json (already written above) and model.safetensors.index.json
    # (stale — references input FP8 shard names that don't exist in the output;
    # vllm's filter_duplicate_safetensors_files would use it to filter out all
    # quant_model_weights-*.safetensors, causing "Cannot find any model weights").
    # The script writes quant_model_weights.safetensors.index.json via writer.close(),
    # and vllm falls back to globbing *.safetensors when model.safetensors.index.json
    # is absent.
    skip_files = {"config.json", "model.safetensors.index.json"}
    copy_extensions = (".py", ".json", ".jinja")
    for root, _, files in os.walk(input_fp8_hf_path):
        for file in files:
            if file in skip_files:
                continue
            # .gitattributes is a hidden file with no extension, copy by name
            if file.endswith(copy_extensions) or file == ".gitattributes":
                src = os.path.join(root, file)
                rel_dir = os.path.relpath(root, input_fp8_hf_path)
                dst_dir = os.path.join(output_hf_path, rel_dir)
                os.makedirs(dst_dir, exist_ok=True)
                shutil.copy2(src, os.path.join(dst_dir, file))

    print(f"\nQuantization complete. Output saved to: {output_hf_path}")
    print(f"  - {writer._save_count} safetensors shards")
    print(f"  - quant_model_description.json ({len(writer.quant_description)} entries)")
    print("  - config.json (quantization_config removed)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize DeepSeek-V4-Flash FP8 weights to W8A8 for vllm-ascend.")
    parser.add_argument(
        "--input_fp8_hf_path",
        type=str,
        required=True,
        help="Path to HF directory containing DeepSeek-V4-Flash FP8 weights.",
    )
    parser.add_argument(
        "--output_hf_path",
        type=str,
        required=True,
        help="Path to output directory for W8A8 quantized weights.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML quantization config. Defaults to quantize_config.yaml next to this script.",
    )
    args = parser.parse_args()

    if args.config is None:
        args.config = os.path.join(os.path.dirname(__file__), "quantize_config.yaml")

    main(args.input_fp8_hf_path, args.output_hf_path, args.config)
