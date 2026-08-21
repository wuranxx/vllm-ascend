#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Adapter seam for an optional single-input/single-output Fake MX QDQ kernel.

When the external kernel is delivered, only ``_load_external_fake_mx_kernel``
needs to import its real entry point. The rest of vLLM-Ascend calls the stable
``fake_mx_quantize`` wrapper in ``vllm_ascend.quantization.fake_mx``.
"""

import importlib
from collections.abc import Callable
from functools import lru_cache
from typing import Any

import torch

_EXTERNAL_KERNEL_MODULE = "vllm_ascend.quantization.kernels._fake_mx_kernel"
_EXTERNAL_KERNEL_ENTRY = "fake_mx_quantize"


@lru_cache(maxsize=1)
def _load_external_fake_mx_kernel() -> tuple[Callable[..., torch.Tensor] | None, str | None]:
    """Lazily resolve the optional extension; installation requires restart."""
    try:
        module = importlib.import_module(_EXTERNAL_KERNEL_MODULE)
        kernel = getattr(module, _EXTERNAL_KERNEL_ENTRY)
    except (ImportError, AttributeError) as error:
        return None, f"cannot load {_EXTERNAL_KERNEL_MODULE}.{_EXTERNAL_KERNEL_ENTRY}: {error}"
    if not callable(kernel):
        return None, f"{_EXTERNAL_KERNEL_MODULE}.{_EXTERNAL_KERNEL_ENTRY} is not callable"
    return kernel, None


def fake_mx_kernel_support_reason(
    tensor: torch.Tensor,
    mx_format: str,
    group_size: int,
    clip_ratio: float | torch.Tensor,
) -> str | None:
    """Return ``None`` when the adapter can execute the request.

    Shape/format validation shared with the reference path happens in the
    public wrapper. Add hardware-specific dtype, layout, format, or group-size
    capability checks here when the external kernel contract is finalized.
    """
    del tensor, mx_format, group_size, clip_ratio
    _, load_error = _load_external_fake_mx_kernel()
    return load_error


def fake_mx_quantize_kernel(
    tensor: torch.Tensor,
    mx_format: str,
    group_size: int,
    clip_ratio: float | torch.Tensor,
) -> torch.Tensor:
    """Invoke the external QDQ kernel and enforce the public tensor contract."""
    kernel, load_error = _load_external_fake_mx_kernel()
    if kernel is None:
        raise RuntimeError(load_error or "Fake MX fused QDQ kernel is unavailable.")

    result: Any = kernel(tensor, mx_format, group_size, clip_ratio)
    if not isinstance(result, torch.Tensor):
        raise TypeError(f"Fake MX fused QDQ kernel must return a Tensor, got {type(result).__name__}.")
    if result.shape != tensor.shape:
        raise ValueError(f"Fake MX fused QDQ kernel changed shape from {tuple(tensor.shape)} to {tuple(result.shape)}.")
    if result.dtype != tensor.dtype:
        raise TypeError(f"Fake MX fused QDQ kernel changed dtype from {tensor.dtype} to {result.dtype}.")
    return result


__all__ = ["fake_mx_kernel_support_reason", "fake_mx_quantize_kernel"]
