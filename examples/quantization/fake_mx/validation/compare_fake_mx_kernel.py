"""Compare a standardized external fake-MX kernel with the reference path."""

import argparse
import importlib
import json
from typing import Callable

import torch

from vllm_ascend.quantization.fake_mx import fake_mx_quantize


def _load_entry(spec: str) -> Callable:
    module_name, separator, function_name = spec.partition(":")
    if not separator:
        raise ValueError("--kernel-entry must use module:function syntax")
    function = getattr(importlib.import_module(module_name), function_name)
    if not callable(function):
        raise TypeError(f"{spec} is not callable")
    return function


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel-entry", required=True, help="Python module:function")
    parser.add_argument("--format", choices=("mxfp4", "mxfp8"), required=True)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--shape", type=int, nargs="+", default=(17, 4096))
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--clip-ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("an available NPU is required")
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(args.seed)
    tensor = torch.randn(*args.shape, dtype=dtype, device="npu")
    kernel = _load_entry(args.kernel_entry)

    expected = fake_mx_quantize(tensor, args.format, args.group_size, args.clip_ratio)
    actual = kernel(tensor, args.format, args.group_size, args.clip_ratio)
    if actual.shape != tensor.shape or actual.dtype != tensor.dtype or actual.device != tensor.device:
        raise RuntimeError("kernel output shape, dtype, or device does not match its input")

    difference = (actual.float() - expected.float()).abs()
    report = {
        "format": args.format,
        "group_size": args.group_size,
        "shape": list(args.shape),
        "dtype": args.dtype,
        "clip_ratio": args.clip_ratio,
        "max_absolute_error": difference.max().item(),
        "mean_absolute_error": difference.mean().item(),
        "different_elements": torch.count_nonzero(difference).item(),
        "total_elements": difference.numel(),
    }
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
