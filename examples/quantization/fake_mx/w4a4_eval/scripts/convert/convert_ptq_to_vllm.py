#!/usr/bin/env python3
"""将 AMCT PTQ .pt 参数转换为 vllm-ascend fake_mx safetensors 格式。

支持两种算法和两种量化目标：
  --algo flatquant --target attn-linear  -> attn 层 FlatQuant 参数
  --algo flatquant --target mlp           -> mlp 层 FlatQuant 参数
  --algo lht --target attn-linear         -> attn 层 LHT 参数
  --algo lht --target mlp                 -> mlp 层 LHT 参数

用法:
  python convert_ptq_to_vllm.py --algo flatquant --target attn-linear \
    --ptq_dir /data/ptq_output_flatquant_amct/ptq_params/qwen3_5/attn-linear \
    --model_dir /data/models/Qwen3.5-9B \
    --output /data/flatquant_attn_params.safetensors

  python convert_ptq_to_vllm.py --algo flatquant --target mlp \
    --ptq_dir /data/ptq_output_flatquant_mlp/ptq_params/qwen3_5/mlp \
    --model_dir /data/models/Qwen3.5-9B \
    --output /data/flatquant_mlp_params.safetensors

  # 合并 attn + mlp 参数
  python convert_ptq_to_vllm.py --merge \
    /data/flatquant_attn_params.safetensors \
    /data/flatquant_mlp_params.safetensors \
    /data/flatquant_attn_mlp_params.safetensors
"""

import argparse
import json
import os

import torch
from safetensors.torch import load_file, save_file

NUM_LAYERS = 32


def load_ptq_params(ptq_dir, layer_idx, target):
    """加载一个层的 PTQ 参数。"""
    result = {}
    if target == "attn-linear":
        for unit_name in ["linear_attn", "self_attn"]:
            path = os.path.join(ptq_dir, f"layer_{layer_idx}_{unit_name}.pt")
            if os.path.exists(path):
                result[unit_name] = torch.load(path, map_location="cpu")
    elif target == "mlp":
        path = os.path.join(ptq_dir, f"layer_{layer_idx}_mlp.pt")
        if os.path.exists(path):
            result["mlp"] = torch.load(path, map_location="cpu")
    return result


def load_original_weights(model_dir, target):
    """加载模型原始权重中与 target 相关的部分。"""
    from safetensors import safe_open

    weights = {}
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        shard_files = set(index["weight_map"].values())
    else:
        shard_files = [f for f in os.listdir(model_dir) if f.endswith(".safetensors")]

    if target == "mlp":
        filter_keys = [".mlp."]
    elif target == "attn-linear":
        filter_keys = [".self_attn.", ".linear_attn."]
    else:
        filter_keys = [""]
    for shard in shard_files:
        path = os.path.join(model_dir, shard)
        if not os.path.exists(path):
            continue
        with safe_open(path, framework="pt") as st:
            for key in st.keys():
                if any(fk in key for fk in filter_keys) and ".weight" in key:
                    weights[key] = st.get_tensor(key)
    return weights


def apply_flatquant_transform(weight, left, right, diag_scale=None):
    """对权重应用 FlatQuant 逆变换: W' = inv(left) @ W @ inv(right).T / diag_scale。"""
    original_shape = weight.shape
    in_features = original_shape[-1]
    left_dim = left.shape[0]
    right_dim = right.shape[0]
    assert left_dim * right_dim == in_features, f"{left_dim}*{right_dim} != {in_features}"

    weight_f32 = weight.to(torch.float32).reshape(-1, left_dim, right_dim)
    inv_left = torch.linalg.solve(left.to(torch.float32), torch.eye(left_dim, dtype=torch.float32))
    inv_right_t = torch.linalg.solve(right.t().to(torch.float32), torch.eye(right_dim, dtype=torch.float32))
    transformed = torch.matmul(inv_left, weight_f32)
    transformed = torch.matmul(transformed, inv_right_t)
    if diag_scale is not None:
        diag = diag_scale.to(torch.float32).reshape(left_dim, right_dim)
        transformed = transformed / diag.unsqueeze(0).clamp(min=1e-8)
    return transformed.reshape(original_shape)


def convert_flatquant_attn(ptq_dir, model_dir, output_path):
    """转换 attn 层 FlatQuant 参数。"""
    original_weights = load_original_weights(model_dir, "attn-linear")
    output_tensors = {}

    for layer_idx in range(NUM_LAYERS):
        ptq = load_ptq_params(ptq_dir, layer_idx, "attn-linear")
        if not ptq:
            continue
        for unit_name, params in ptq.items():
            input_left = params["input_transform.transform.linear_left.weight"]
            input_right = params["input_transform.transform.linear_right.weight"]
            input_diag = params.get("input_transform.transform.diag_scale")
            out_left = params["out_transform.transform.linear_left.weight"]
            out_right = params["out_transform.transform.linear_right.weight"]
            out_diag = params.get("out_transform.transform.diag_scale")

            if unit_name == "self_attn":
                for proj, tf_left, tf_right, tf_diag in [
                    ("qkv_proj", input_left, input_right, input_diag),
                    ("o_proj", out_left, out_right, out_diag),
                ]:
                    _process_flatquant_proj(
                        original_weights, output_tensors, layer_idx, f"self_attn.{proj}", tf_left, tf_right, tf_diag
                    )
            elif unit_name == "linear_attn":
                for proj in ["in_proj_qkvz", "in_proj_ba"]:
                    _process_flatquant_proj(
                        original_weights,
                        output_tensors,
                        layer_idx,
                        f"linear_attn.{proj}",
                        input_left,
                        input_right,
                        input_diag,
                    )
                _process_flatquant_proj(
                    original_weights, output_tensors, layer_idx, "linear_attn.out_proj", out_left, out_right, out_diag
                )

    save_file(output_tensors, output_path)
    print(f"Saved {len(output_tensors)} tensors to {output_path}")


def convert_flatquant_mlp(ptq_dir, model_dir, output_path):
    """转换 mlp 层 FlatQuant 参数。"""
    original_weights = load_original_weights(model_dir, "mlp")
    output_tensors = {}

    for layer_idx in range(NUM_LAYERS):
        ptq = load_ptq_params(ptq_dir, layer_idx, "mlp")
        if not ptq:
            continue
        params = _flatten_ptq_params(ptq["mlp"])
        input_left = params["input_transform.transform.linear_left.weight"]
        input_right = params["input_transform.transform.linear_right.weight"]
        input_diag = params.get("input_transform.transform.diag_scale")
        hidden_left = params["hidden_transform.transform.linear_left.weight"]
        hidden_right = params["hidden_transform.transform.linear_right.weight"]
        hidden_diag = params.get("hidden_transform.transform.diag_scale")

        for proj, tf_left, tf_right, tf_diag in [
            (
                "gate_proj",
                input_left.clone(),
                input_right.clone(),
                input_diag.clone() if input_diag is not None else None,
            ),
            (
                "up_proj",
                input_left.clone(),
                input_right.clone(),
                input_diag.clone() if input_diag is not None else None,
            ),
            (
                "down_proj",
                hidden_left.clone(),
                hidden_right.clone(),
                hidden_diag.clone() if hidden_diag is not None else None,
            ),
        ]:
            _process_flatquant_proj(
                original_weights, output_tensors, layer_idx, f"mlp.{proj}", tf_left, tf_right, tf_diag
            )

    save_file(output_tensors, output_path)
    print(f"Saved {len(output_tensors)} tensors to {output_path}")


def _process_flatquant_proj(original_weights, output_tensors, layer_idx, proj_path, left, right, diag):
    """Output FlatQuant transform matrices in vllm-ascend fused-projection naming.

    vllm-ascend loads left_trans/right_trans/diag_scale at runtime and applies
    the weight inverse transform itself, so we do NOT output pre-transformed
    .weight tensors here.
    """
    prefix = f"model.language_model.layers.{layer_idx}.{proj_path}"
    output_tensors[f"{prefix}.left_trans"] = left.to(torch.float32).clone()
    output_tensors[f"{prefix}.right_trans"] = right.to(torch.float32).clone()
    if diag is not None:
        output_tensors[f"{prefix}.diag_scale"] = diag.to(torch.float32).clone()


def _extract_lht_transform_weight(params, transform_name="input_transform"):
    """Extract LHT transform_weight from AMCT params, supporting multiple key formats."""
    key = f"{transform_name}.transform.linear.weight"
    if key in params:
        return params[key]
    if f"{transform_name}.transform_weight" in params:
        return params[f"{transform_name}.transform_weight"]
    if transform_name in params and isinstance(params[transform_name], dict):
        sub = params[transform_name]
        if "transform_weight" in sub:
            return sub["transform_weight"]
        if "transform" in sub and isinstance(sub["transform"], dict) and "linear" in sub["transform"]:
            return sub["transform"]["linear"]["weight"]
    return None


def convert_lht_attn(ptq_dir, output_path):
    """转换 attn 层 LHT 参数。"""
    output_tensors = {}
    for layer_idx in range(NUM_LAYERS):
        ptq = load_ptq_params(ptq_dir, layer_idx, "attn-linear")
        if not ptq:
            continue
        for unit_name, params in ptq.items():
            input_tw = _extract_lht_transform_weight(params, "input_transform")
            out_tw = _extract_lht_transform_weight(params, "out_transform")
            if input_tw is None or out_tw is None:
                continue
            if unit_name == "self_attn":
                prefix = f"model.language_model.layers.{layer_idx}.self_attn"
                output_tensors[f"{prefix}.qkv_proj.transform_weight"] = input_tw.to(torch.float32)
                output_tensors[f"{prefix}.o_proj.transform_weight"] = out_tw.to(torch.float32)
            elif unit_name == "linear_attn":
                prefix = f"model.language_model.layers.{layer_idx}.linear_attn"
                output_tensors[f"{prefix}.in_proj_qkvz.transform_weight"] = input_tw.to(torch.float32)
                output_tensors[f"{prefix}.in_proj_ba.transform_weight"] = input_tw.to(torch.float32).clone()
                output_tensors[f"{prefix}.out_proj.transform_weight"] = out_tw.to(torch.float32)
    save_file(output_tensors, output_path)
    print(f"Saved {len(output_tensors)} tensors to {output_path}")


def convert_lht_mlp(ptq_dir, output_path):
    """转换 mlp 层 LHT 参数。"""
    output_tensors = {}
    for layer_idx in range(NUM_LAYERS):
        ptq = load_ptq_params(ptq_dir, layer_idx, "mlp")
        if not ptq:
            continue
        params = ptq["mlp"]
        input_tw = _extract_lht_transform_weight(params, "input_transform")
        hidden_tw = _extract_lht_transform_weight(params, "hidden_transform")
        if input_tw is None or hidden_tw is None:
            continue
        for proj, tw in [
            ("gate_proj", input_tw.clone()),
            ("up_proj", input_tw.clone()),
            ("down_proj", hidden_tw.clone()),
        ]:
            prefix = f"model.language_model.layers.{layer_idx}.mlp.{proj}"
            output_tensors[f"{prefix}.transform_weight"] = tw.to(torch.float32)
    save_file(output_tensors, output_path)
    print(f"Saved {len(output_tensors)} tensors to {output_path}")


def merge_safetensors(paths, output_path):
    """合并多个 safetensors 文件。"""
    merged = {}
    for p in paths:
        data = load_file(p)
        merged.update(data)
        print(f"  Loaded {len(data)} tensors from {p}")
    save_file(merged, output_path)
    print(f"Saved {len(merged)} tensors to {output_path}")


def convert_omniquant_attn(ptq_dir, output_path):
    """转换 attn 层 OmniQuant 参数。输出 log_scale per projection."""
    output_tensors = {}
    for layer_idx in range(NUM_LAYERS):
        ptq = load_ptq_params(ptq_dir, layer_idx, "attn-linear")
        if not ptq:
            continue
        for unit_name, params in ptq.items():
            input_scale = params["input_transform.log_scale"]
            out_scale = params["out_transform.log_scale"]
            if unit_name == "self_attn":
                prefix = f"model.language_model.layers.{layer_idx}.self_attn"
                output_tensors[f"{prefix}.qkv_proj.log_scale"] = input_scale.to(torch.float32)
                output_tensors[f"{prefix}.o_proj.log_scale"] = out_scale.to(torch.float32)
            elif unit_name == "linear_attn":
                prefix = f"model.language_model.layers.{layer_idx}.linear_attn"
                output_tensors[f"{prefix}.in_proj_qkvz.log_scale"] = input_scale.to(torch.float32)
                output_tensors[f"{prefix}.in_proj_ba.log_scale"] = input_scale.to(torch.float32).clone()
                output_tensors[f"{prefix}.out_proj.log_scale"] = out_scale.to(torch.float32)
    save_file(output_tensors, output_path)
    print(f"Saved {len(output_tensors)} tensors to {output_path}")


def convert_omniquant_mlp(ptq_dir, output_path):
    """转换 mlp 层 OmniQuant 参数。"""
    output_tensors = {}
    for layer_idx in range(NUM_LAYERS):
        ptq = load_ptq_params(ptq_dir, layer_idx, "mlp")
        if not ptq:
            continue
        params = ptq["mlp"]
        flat = _flatten_ptq_params(params)
        input_scale = flat["input_transform.log_scale"]
        hidden_scale = flat["hidden_transform.log_scale"]
        for proj, scale in [
            ("gate_proj", input_scale.clone()),
            ("up_proj", input_scale.clone()),
            ("down_proj", hidden_scale.clone()),
        ]:
            prefix = f"model.language_model.layers.{layer_idx}.mlp.{proj}"
            output_tensors[f"{prefix}.log_scale"] = scale.to(torch.float32)
    save_file(output_tensors, output_path)
    print(f"Saved {len(output_tensors)} tensors to {output_path}")


def _flatten_ptq_params(params, prefix=""):
    """Recursively flatten nested dict PTQ params into dotted keys."""
    flat = {}
    for k, v in params.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(_flatten_ptq_params(v, full_key))
        elif isinstance(v, torch.Tensor):
            flat[full_key] = v
    return flat


_MERGE_MAPS = {
    "self_attn": {"qkv_proj": ["q_proj", "k_proj", "v_proj"]},
    "linear_attn": {
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    },
}

_ACTIVATION_RENAME = {
    "self_attn": {"inp": "qkv_proj"},
    "linear_attn": {"inp": "in_proj_qkvz", "o_proj": "out_proj"},
}


def _reverse_merge(merge_maps):
    """Build merged_name → [sub_names] lookup from all unit merge maps."""
    reverse = {}
    for unit_map in merge_maps.values():
        for merged_name, split_names in unit_map.items():
            reverse[merged_name] = split_names
    return reverse


_MERGE_REVERSE = _reverse_merge(_MERGE_MAPS)


def _get_weight_shape(original_weights, layer_idx, unit_name, proj_name):
    """Look up weight [out, in] shape, handling merged projections."""
    sub_names = _MERGE_REVERSE.get(proj_name, [proj_name])
    total_out = 0
    input_size = None
    for sn in sub_names:
        key = f"model.language_model.layers.{layer_idx}.{unit_name}.{sn}.weight"
        if key not in original_weights:
            return None
        w = original_weights[key]
        total_out += w.shape[0]
        input_size = w.shape[1]
    if input_size is None:
        return None
    return (total_out, input_size)


def _post_process_projections(proj_params, unit_name, is_activation):
    """Merge split projections or rename activation projections to vllm names."""
    if is_activation:
        rename_map = _ACTIVATION_RENAME.get(unit_name, {})
        return {rename_map.get(k, k): v for k, v in proj_params.items()}
    else:
        merge_map = _MERGE_MAPS.get(unit_name, {})
        if not merge_map:
            return proj_params
        merged = {}
        consumed = set()
        for merged_name, split_names in merge_map.items():
            if all(n in proj_params for n in split_names):
                merged[merged_name] = {}
                for param_name in proj_params[split_names[0]]:
                    tensors = [proj_params[n][param_name] for n in split_names]
                    merged[merged_name][param_name] = torch.cat(tensors, dim=0)
                consumed.update(split_names)
        for proj_name, params in proj_params.items():
            if proj_name not in consumed:
                merged[proj_name] = params
        return merged


def _convert_per_projection_attn(ptq_dir, output_path, algo_name, param_names, model_dir=None):
    """Generic converter for per-projection algorithms (autoround/lwc/lac).

    Supports two AMCT key formats:
    - weight type: {proj}.weight_quantizer.algorithms.{algo}.{param}
    - activation type: {proj}_afq.algorithms.{algo}.{param}

    For linear_attn, merges split projections (in_proj_qkv + in_proj_z → in_proj_qkvz)
    to match vllm's merged module names. Reshapes 'value' to weight shape if model_dir given.
    """
    original_weights = load_original_weights(model_dir, "attn-linear") if model_dir else {}
    output_tensors = {}
    for layer_idx in range(NUM_LAYERS):
        ptq = load_ptq_params(ptq_dir, layer_idx, "attn-linear")
        if not ptq:
            continue
        for unit_name, params in ptq.items():
            prefix = f"model.language_model.layers.{layer_idx}.{unit_name}"
            flat_params = _flatten_ptq_params(params)
            proj_params = {}
            is_activation = False
            for ptq_key, value in flat_params.items():
                if not isinstance(value, torch.Tensor):
                    continue
                parts = ptq_key.split(".")
                proj_name = None
                param_name = None
                if (
                    len(parts) >= 5
                    and parts[1] == "weight_quantizer"
                    and parts[2] == "algorithms"
                    and parts[3] == algo_name
                ):
                    proj_name = parts[0]
                    param_name = ".".join(parts[4:])
                elif (
                    len(parts) >= 4 and parts[0].endswith("_afq") and parts[1] == "algorithms" and parts[2] == algo_name
                ):
                    proj_name = parts[0].removesuffix("_afq")
                    param_name = ".".join(parts[3:])
                    is_activation = True
                elif (
                    len(parts) >= 4
                    and parts[0].endswith("_quant")
                    and parts[1] == "algorithms"
                    and parts[2] == algo_name
                ):
                    proj_name = parts[0].removesuffix("_quant")
                    param_name = ".".join(parts[3:])
                    is_activation = True
                if proj_name and param_name and param_name in param_names:
                    proj_params.setdefault(proj_name, {})[param_name] = value.clone()
            final_params = _post_process_projections(proj_params, unit_name, is_activation)
            for proj_name, pmap in final_params.items():
                for param_name, value in pmap.items():
                    if param_name == "value" and original_weights:
                        shape = _get_weight_shape(original_weights, layer_idx, unit_name, proj_name)
                        if shape and value.numel() == shape[0] * shape[1]:
                            value = value.reshape(shape)
                    out_key = f"{prefix}.{proj_name}.{param_name}"
                    output_tensors[out_key] = value
    save_file(output_tensors, output_path)
    print(f"Saved {len(output_tensors)} tensors to {output_path}")


def _convert_per_projection_mlp(ptq_dir, output_path, algo_name, param_names, model_dir=None):
    """Generic converter for per-projection weight algorithms (mlp target)."""
    original_weights = load_original_weights(model_dir, "mlp") if model_dir else {}
    output_tensors = {}
    for layer_idx in range(NUM_LAYERS):
        ptq = load_ptq_params(ptq_dir, layer_idx, "mlp")
        if not ptq:
            continue
        for unit_name, params in ptq.items():
            prefix = f"model.language_model.layers.{layer_idx}.{unit_name}"
            flat_params = _flatten_ptq_params(params)
            for ptq_key, value in flat_params.items():
                parts = ptq_key.split(".")
                if (
                    len(parts) >= 5
                    and parts[1] == "weight_quantizer"
                    and parts[2] == "algorithms"
                    and parts[3] == algo_name
                ):
                    proj_name = parts[0]
                    param_name = ".".join(parts[4:])
                    if param_name in param_names:
                        out_value = value.clone()
                        if param_name == "value" and original_weights:
                            weight_key = f"{prefix}.{proj_name}.weight"
                            if weight_key in original_weights:
                                w = original_weights[weight_key]
                                if out_value.numel() == w.numel():
                                    out_value = out_value.reshape(w.shape)
                        out_key = f"{prefix}.{proj_name}.{param_name}"
                        output_tensors[out_key] = out_value
    save_file(output_tensors, output_path)
    print(f"Saved {len(output_tensors)} tensors to {output_path}")


def convert_autoround_attn(ptq_dir, output_path, model_dir=None):
    """转换 attn 层 AutoRound 参数。per-projection value/min_scale/max_scale."""
    _convert_per_projection_attn(ptq_dir, output_path, "autoround", {"value", "min_scale", "max_scale"}, model_dir)


def convert_autoround_mlp(ptq_dir, output_path, model_dir=None):
    """转换 mlp 层 AutoRound 参数。"""
    _convert_per_projection_mlp(ptq_dir, output_path, "autoround", {"value", "min_scale", "max_scale"}, model_dir)


def convert_lwc_attn(ptq_dir, output_path, model_dir=None):
    """转换 attn 层 LWC 参数。per-projection clip_factor_min/max."""
    _convert_per_projection_attn(ptq_dir, output_path, "lwc", {"clip_factor_min", "clip_factor_max"}, model_dir)


def convert_lwc_mlp(ptq_dir, output_path):
    """转换 mlp 层 LWC 参数。"""
    _convert_per_projection_mlp(ptq_dir, output_path, "lwc", {"clip_factor_min", "clip_factor_max"})


def convert_lac_attn(ptq_dir, output_path, model_dir=None):
    """转换 attn 层 LAC 参数。per-projection clip_factor + maxval/minval."""
    _convert_per_projection_attn(ptq_dir, output_path, "lac", {"clip_factor_min", "clip_factor_max", "maxval", "minval"}, model_dir)


def convert_lac_mlp(ptq_dir, output_path):
    """转换 mlp 层 LAC 参数。

    LAC mlp uses input_quant (→ gate_proj + up_proj) and hidden_quant (→ down_proj).
    Each projection gets its own copy of clip_factor/maxval/minval.
    """
    output_tensors = {}
    for layer_idx in range(NUM_LAYERS):
        ptq = load_ptq_params(ptq_dir, layer_idx, "mlp")
        if not ptq:
            continue
        params = ptq["mlp"]
        flat = _flatten_ptq_params(params)
        quant_map = {
            "input_quant": ["gate_proj", "up_proj"],
            "hidden_quant": ["down_proj"],
        }
        for quant_key, proj_list in quant_map.items():
            for param_name in ["clip_factor_min", "clip_factor_max", "maxval", "minval"]:
                full_key = f"{quant_key}.algorithms.lac.{param_name}"
                if full_key not in flat:
                    continue
                value = flat[full_key]
                for proj in proj_list:
                    prefix = f"model.language_model.layers.{layer_idx}.mlp.{proj}"
                    output_tensors[f"{prefix}.{param_name}"] = value.clone()
    save_file(output_tensors, output_path)
    print(f"Saved {len(output_tensors)} tensors to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert AMCT PTQ params to vllm-ascend safetensors")
    parser.add_argument(
        "--algo",
        choices=["flatquant", "lht", "omniquant", "autoround", "lwc", "lac"],
        help="Algorithm name",
    )
    parser.add_argument("--target", choices=["attn-linear", "mlp"], help="Quant target")
    parser.add_argument("--ptq_dir", help="PTQ .pt params directory")
    parser.add_argument("--model_dir", help="Model directory (for FlatQuant weight transform)")
    parser.add_argument("--output", help="Output safetensors path")
    parser.add_argument("--merge", nargs=3, metavar=("INPUT1", "INPUT2", "OUTPUT"), help="Merge two safetensors files")
    args = parser.parse_args()

    if args.merge:
        merge_safetensors([args.merge[0], args.merge[1]], args.merge[2])
        return

    if not all([args.algo, args.target, args.ptq_dir, args.output]):
        parser.error("--algo, --target, --ptq_dir, --output are required (unless --merge)")

    if args.algo == "flatquant":
        if not args.model_dir:
            parser.error("--model_dir is required for FlatQuant (needs original weights for transform)")
        if args.target == "attn-linear":
            convert_flatquant_attn(args.ptq_dir, args.model_dir, args.output)
        elif args.target == "mlp":
            convert_flatquant_mlp(args.ptq_dir, args.model_dir, args.output)
    elif args.algo == "lht":
        if args.target == "attn-linear":
            convert_lht_attn(args.ptq_dir, args.output)
        elif args.target == "mlp":
            convert_lht_mlp(args.ptq_dir, args.output)
    elif args.algo == "omniquant":
        if args.target == "attn-linear":
            convert_omniquant_attn(args.ptq_dir, args.output)
        elif args.target == "mlp":
            convert_omniquant_mlp(args.ptq_dir, args.output)
    elif args.algo == "autoround":
        if args.target == "attn-linear":
            convert_autoround_attn(args.ptq_dir, args.output, args.model_dir)
        elif args.target == "mlp":
            convert_autoround_mlp(args.ptq_dir, args.output, args.model_dir)
    elif args.algo == "lwc":
        if args.target == "attn-linear":
            convert_lwc_attn(args.ptq_dir, args.output, args.model_dir)
        elif args.target == "mlp":
            convert_lwc_mlp(args.ptq_dir, args.output)
    elif args.algo == "lac":
        if args.target == "attn-linear":
            convert_lac_attn(args.ptq_dir, args.output, args.model_dir)
        elif args.target == "mlp":
            convert_lac_mlp(args.ptq_dir, args.output)


if __name__ == "__main__":
    main()
