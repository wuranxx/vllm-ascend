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
"""Device-independent transforms and fake quantization for MX validation.

The returned tensor keeps the input floating-point dtype.  Values are rounded
as if they were stored as an MX element plus an E8M0 power-of-two scale, then
immediately dequantized.  No native FP4/FP8 dtype or MX kernel is required.
"""

import math
from collections.abc import Mapping
from typing import Any, Literal, cast

import torch
from vllm.config import get_current_vllm_config_or_none
from vllm.logger import logger

FakeMXBackend = Literal["reference", "kernel", "auto"]
FakeMXFormat = Literal["mxfp4", "mxfp8"]
FAKE_MX_BACKENDS = frozenset({"reference", "kernel", "auto"})


def get_fake_mx_backend(quant_description: Mapping[str, Any] | None = None) -> FakeMXBackend:
    """Resolve the process model's QDQ backend at the stable wrapper boundary."""
    if quant_description is None:
        vllm_config = get_current_vllm_config_or_none()
        quant_config = getattr(vllm_config, "quant_config", None)
        quant_description = getattr(quant_config, "quant_description", {})
    backend = quant_description.get("fake_mx_backend", "reference")
    if not isinstance(backend, str) or backend not in FAKE_MX_BACKENDS:
        raise ValueError(f"fake_mx_backend must be one of {sorted(FAKE_MX_BACKENDS)}, got {backend!r}.")
    return cast(FakeMXBackend, backend)


def learned_hadamard_transform(
    tensor: torch.Tensor,
    transform_weight: torch.Tensor,
) -> torch.Tensor:
    """Apply AMCT-Q's learned blockwise Hadamard-like transform.

    AMCT-Q learns one invertible ``K x K`` matrix ``T`` and evaluates an
    activation as ``reshape(x, -1, K) @ T``.  The offline converter must pair
    this with ``reshape(weight, -1, K) @ inv(T).T`` before weight fake-QDQ.
    ``transform_weight`` remains floating point; this helper only simulates
    the transform and does not require a native Hadamard/MX operator.
    """
    if not tensor.is_floating_point():
        raise TypeError(f"Hadamard Learning requires a floating tensor, got {tensor.dtype}.")
    if not transform_weight.is_floating_point():
        raise TypeError(f"Hadamard Learning transform_weight must be floating point, got {transform_weight.dtype}.")
    if transform_weight.ndim != 2 or transform_weight.shape[0] != transform_weight.shape[1]:
        raise ValueError(
            "Hadamard Learning transform_weight must be a square 2-D matrix, "
            f"got shape {tuple(transform_weight.shape)}."
        )
    matrix_size = transform_weight.shape[0]
    if tensor.shape[-1] % matrix_size:
        raise ValueError(
            f"Hadamard Learning input dimension ({tensor.shape[-1]}) must be divisible by matrix_size ({matrix_size})."
        )

    original_shape = tensor.shape
    transformed = (
        tensor.to(torch.float32)
        .reshape(-1, matrix_size)
        .matmul(transform_weight.to(device=tensor.device, dtype=torch.float32))
    )
    return transformed.reshape(original_shape).to(tensor.dtype)


def hadamard_transform(
    tensor: torch.Tensor,
    matrix_size: int = 128,
) -> torch.Tensor:
    """Apply a deterministic normalized Hadamard transform blockwise.

    Uses the same butterfly structure as the Fast Walsh-Hadamard Transform
    (FWHT) without a random sign diagonal.  The result is identical to
    multiplying by ``scipy.linalg.hadamard(matrix_size) / sqrt(matrix_size)``
    but requires no external dependency.  AMCT-final uses the same
    deterministic Hadamard matrix, so vllm-ascend and AMCT stay in sync.
    """
    if not tensor.is_floating_point():
        raise TypeError(f"RHT requires a floating tensor, got {tensor.dtype}.")
    if matrix_size <= 0 or matrix_size & (matrix_size - 1):
        raise ValueError(f"RHT matrix_size must be a positive power of two, got {matrix_size}.")
    if tensor.shape[-1] % matrix_size:
        raise ValueError(
            f"RHT input dimension ({tensor.shape[-1]}) must be divisible by matrix_size ({matrix_size})."
        )

    original_shape = tensor.shape
    original_dtype = tensor.dtype
    blocked = tensor.to(torch.float32).reshape(*original_shape[:-1], -1, matrix_size)

    step = 1
    while step < matrix_size:
        butterfly = blocked.reshape(*blocked.shape[:-1], matrix_size // (2 * step), 2, step)
        left = butterfly[..., 0, :]
        right = butterfly[..., 1, :]
        blocked = torch.cat((left + right, left - right), dim=-1).reshape(*blocked.shape)
        step *= 2

    return (blocked / math.sqrt(matrix_size)).reshape(original_shape).to(original_dtype)


def randomized_hadamard_transform(
    tensor: torch.Tensor,
    signs: torch.Tensor,
    group_size: int = 32,
) -> torch.Tensor:
    """Apply a normalized randomized Walsh-Hadamard transform by group.

    ``signs`` is the random Rademacher diagonal in ``H @ D``.  The same signs
    must be used by the offline weight rotation.  The transform is evaluated
    with ordinary floating-point tensor operations and therefore works on
    devices without a native Hadamard or MX kernel.
    """
    if not tensor.is_floating_point():
        raise TypeError(f"RHT requires a floating tensor, got {tensor.dtype}.")
    if group_size <= 0 or group_size & (group_size - 1):
        raise ValueError(f"RHT group_size must be a positive power of two, got {group_size}.")
    if tensor.shape[-1] % group_size:
        raise ValueError(f"RHT input dimension ({tensor.shape[-1]}) must be divisible by group_size ({group_size}).")
    if signs.numel() not in (group_size, tensor.shape[-1]):
        raise ValueError(
            f"RHT signs must contain either group_size ({group_size}) or "
            f"the full input dimension ({tensor.shape[-1]}) values, got {signs.numel()}."
        )

    original_shape = tensor.shape
    original_dtype = tensor.dtype
    blocked = tensor.to(torch.float32).reshape(*original_shape[:-1], -1, group_size)
    if signs.numel() == group_size:
        sign_blocks = signs.reshape(1, group_size)
    else:
        sign_blocks = signs.reshape(-1, group_size)
    blocked = blocked * sign_blocks.to(device=tensor.device, dtype=torch.float32)

    step = 1
    while step < group_size:
        butterfly = blocked.reshape(*blocked.shape[:-1], group_size // (2 * step), 2, step)
        left = butterfly[..., 0, :]
        right = butterfly[..., 1, :]
        blocked = torch.cat((left + right, left - right), dim=-1).reshape(*blocked.shape)
        step *= 2

    return (blocked / math.sqrt(group_size)).reshape(original_shape).to(original_dtype)


def _amct_round(value: torch.Tensor) -> torch.Tensor:
    """AMCT ``round_ste`` forward value: round halves away from zero."""
    return torch.sign(value) * torch.floor(torch.abs(value) + 0.5)


def _amct_shared_exponent(blocked: torch.Tensor, element_emax: int) -> torch.Tensor:
    """Match AMCT-Q ``shared_exponents``/``round_to_decimal`` exactly."""
    amax = blocked.abs().amax(dim=-1, keepdim=True)
    nonzero_amax = amax + torch.finfo(torch.float32).tiny * (amax == 0).to(amax.dtype)
    exponent = torch.floor(torch.log2(nonzero_amax))
    mantissa = nonzero_amax / torch.exp2(exponent)
    exponent = torch.where(mantissa > 1.75, exponent + 1, exponent)
    return (exponent - element_emax).clamp(min=-127, max=1e10)


def _amct_quantize_elementwise(
    value: torch.Tensor,
    min_exponent: int,
    max_normal: float,
    shift: int,
    rounding_offset: torch.Tensor | None = None,
) -> torch.Tensor:
    """Match AMCT-Q ``quantize_elewise`` without its STE gradient wrapper."""
    private_exponent = torch.floor(torch.log2(value.abs() + (value == 0).to(value.dtype)))
    private_scale = torch.exp2(private_exponent.clamp(min=min_exponent))
    scaled = value / private_scale * shift
    if rounding_offset is not None:
        scaled = scaled + rounding_offset.to(device=scaled.device, dtype=scaled.dtype)
    rounded = _amct_round(scaled)
    return (rounded / shift * private_scale).clamp(min=-max_normal, max=max_normal)


def _validate_fake_mx_request(
    tensor: torch.Tensor,
    mx_format: FakeMXFormat,
    group_size: int,
    clip_ratio: float | torch.Tensor,
) -> None:
    if not tensor.is_floating_point():
        raise TypeError(f"fake MX quantization requires a floating tensor, got {tensor.dtype}.")
    if mx_format not in ("mxfp4", "mxfp8"):
        raise ValueError(f"Unsupported fake MX format: {mx_format}.")
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}.")
    if not isinstance(clip_ratio, torch.Tensor) and not 0.0 < clip_ratio <= 1.0:
        raise ValueError(f"clip_ratio must be in (0, 1], got {clip_ratio}.")
    if tensor.shape[-1] % group_size:
        raise ValueError(
            "AMCT-compatible fake MX requires the last dimension "
            f"({tensor.shape[-1]}) to be divisible by group_size ({group_size})."
        )


def _fake_mx_quantize_reference(
    tensor: torch.Tensor,
    mx_format: FakeMXFormat,
    group_size: int,
    clip_ratio: float | torch.Tensor,
    rounding_offset: torch.Tensor | None = None,
) -> torch.Tensor:
    """PyTorch golden implementation used for validation and fallback."""
    original_shape = tensor.shape
    original_dtype = tensor.dtype
    work = tensor.to(torch.float32)
    blocked = work.reshape(*original_shape[:-1], -1, group_size)
    amax = blocked.abs().amax(dim=-1, keepdim=True)
    if isinstance(clip_ratio, torch.Tensor):
        ratio = clip_ratio.to(device=tensor.device, dtype=torch.float32)
        if ratio.ndim == amax.ndim - 1:
            ratio = ratio.unsqueeze(-1)
        try:
            ratio = torch.broadcast_to(ratio, amax.shape)
        except RuntimeError as error:
            raise ValueError(
                f"clip_ratio shape {tuple(clip_ratio.shape)} is not broadcastable to MX blocks {tuple(amax.shape)}."
            ) from error
        quant_amax = amax * ratio
    else:
        quant_amax = amax * clip_ratio
    clipped = blocked.clamp(min=-quant_amax, max=quant_amax)
    element_emax, min_exponent, max_normal, shift = (2, 0, 6.0, 2) if mx_format == "mxfp4" else (8, -6, 448.0, 8)
    scale = torch.exp2(_amct_shared_exponent(clipped, element_emax))
    v = None
    if rounding_offset is not None:
        v = rounding_offset.to(device=tensor.device, dtype=torch.float32)
        if v.shape == work.shape:
            v = v.reshape(*original_shape[:-1], -1, group_size)
        elif v.ndim == clipped.ndim - 1:
            v = v.unsqueeze(-1)
        try:
            v = torch.broadcast_to(v, clipped.shape)
        except RuntimeError as error:
            raise ValueError(
                f"rounding_offset shape {tuple(rounding_offset.shape)}"
                f" is not broadcastable to MX blocks {tuple(clipped.shape)}."
            ) from error
    quantized = _amct_quantize_elementwise(clipped / scale, min_exponent, max_normal, shift, rounding_offset=v)
    result = (scale * quantized).reshape(original_shape)
    return result.to(original_dtype)


def fake_mx_quantize(
    tensor: torch.Tensor,
    mx_format: FakeMXFormat,
    group_size: int = 32,
    clip_ratio: float | torch.Tensor = 1.0,
    rounding_offset: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply MX QDQ with a stable reference/kernel dispatch boundary.

    ``reference`` preserves the AMCT-compatible PyTorch implementation.
    ``kernel`` requires the optional fused QDQ adapter. ``auto`` selects that
    kernel when available and otherwise falls back to the reference path.
    The returned tensor always keeps the input floating-point shape and dtype.

    ``rounding_offset`` adds a per-element offset before rounding, matching
    AMCT's ``quantize_elewise(x, v)`` → ``round_ste(x + v)``. Used by AutoRound.
    """
    backend = get_fake_mx_backend()
    _validate_fake_mx_request(tensor, mx_format, group_size, clip_ratio)
    if tensor.shape[-1] == 0:
        return tensor
    if backend == "reference":
        return _fake_mx_quantize_reference(tensor, mx_format, group_size, clip_ratio, rounding_offset)

    # Keep the optional extension out of import-time initialization so the
    # reference backend remains usable when no fused kernel is installed.
    from vllm_ascend.quantization.kernels.fake_mx import (
        fake_mx_kernel_support_reason,
        fake_mx_quantize_kernel,
    )

    unsupported_reason = fake_mx_kernel_support_reason(tensor, mx_format, group_size, clip_ratio)
    if unsupported_reason is None:
        return fake_mx_quantize_kernel(tensor, mx_format, group_size, clip_ratio)
    if backend == "kernel":
        raise RuntimeError(f"fake_mx_backend='kernel' cannot execute this request: {unsupported_reason}")

    logger.warning_once(
        "Fake MX fused QDQ kernel is unavailable for this request; falling back "
        f"to the reference backend. Reason: {unsupported_reason}"
    )
    return _fake_mx_quantize_reference(tensor, mx_format, group_size, clip_ratio)


__all__ = [
    "FakeMXBackend",
    "FakeMXFormat",
    "fake_mx_quantize",
    "get_fake_mx_backend",
    "hadamard_transform",
    "learned_hadamard_transform",
    "randomized_hadamard_transform",
]
