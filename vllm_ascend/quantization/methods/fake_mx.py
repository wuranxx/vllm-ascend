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
"""Fake MXFP4/MXFP8 schemes backed by ordinary floating-point NPU kernels."""

import math
import os
from collections.abc import Callable
from functools import cache
from typing import Any

import torch
import torch.nn.functional as F
from vllm.config import get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import logger
from vllm.model_executor.layers.linear import RowParallelLinear

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
from vllm_ascend.ops.fused_moe.experts_selector import select_experts
from vllm_ascend.ops.fused_moe.moe_runtime_args import build_fused_experts_input
from vllm_ascend.quantization.fake_mx import (
    FakeMXFormat,
    fake_mx_quantize,
    hadamard_transform,
    learned_hadamard_transform,
    randomized_hadamard_transform,
)
from vllm_ascend.utils import maybe_trans_nz

from .base import AscendLinearScheme, AscendMoEScheme, QuantType, get_moe_num_logical_experts
from .registry import register_scheme

MAX_FLATQUANT_TRANSFORM_DIM = 256
DEFAULT_TRANSFORM_MATRIX_SIZE = 128


def _quant_description() -> dict[str, Any]:
    """Return the active fake-MX experiment configuration."""
    return get_current_vllm_config().quant_config.quant_description


def _resolve_model_artifact(path: str) -> str:
    """Resolve an algorithm artifact relative to the loaded model."""
    if os.path.isabs(path):
        return path
    model_path = get_current_vllm_config().model_config.model
    return os.path.join(model_path, path)


@cache
def _load_safetensors(resolved_path: str) -> dict[str, torch.Tensor]:
    if not os.path.isfile(resolved_path):
        raise FileNotFoundError(f"Fake-MX transform artifact not found: {resolved_path}")

    from safetensors.torch import load_file

    params = load_file(resolved_path)
    logger.info("Loaded %d fake-MX transform params from %s", len(params), resolved_path)
    return params


def _load_transform_params(path: str) -> dict[str, torch.Tensor]:
    """Load one immutable transform artifact once per worker process."""
    return _load_safetensors(_resolve_model_artifact(path))


def _layer_prefix_candidates(layer: torch.nn.Module) -> tuple[str, ...]:
    """Map vLLM layer prefixes to the prefixes emitted by AMCT exporters."""
    prefix = getattr(layer, "prefix", "") or ""
    logical_prefixes = [prefix]
    if prefix.endswith(".gate_up_proj"):
        # vLLM physically fuses the logical gate/up projections. AMCT exports
        # the same input transform under both logical names; accept either.
        logical_prefixes.extend(
            [
                prefix.removesuffix("gate_up_proj") + "gate_proj",
                prefix.removesuffix("gate_up_proj") + "up_proj",
            ]
        )
    candidates = []
    for candidate in logical_prefixes:
        candidates.extend(
            [
                candidate,
                f"model.{candidate}",
                candidate.replace("language_model.model.", "model.language_model."),
            ]
        )
    return tuple(dict.fromkeys(candidates))


def _find_transform_param(
    params: dict[str, torch.Tensor],
    layer: torch.nn.Module,
    suffix: str,
) -> tuple[str, torch.Tensor] | None:
    for prefix in _layer_prefix_candidates(layer):
        key = f"{prefix}.{suffix}"
        if key in params:
            return key, params[key]
    return None


def _copy_transform_param(
    target: torch.Tensor,
    params: dict[str, torch.Tensor],
    layer: torch.nn.Module,
    suffix: str,
    *,
    required: bool = True,
) -> str | None:
    match = _find_transform_param(params, layer, suffix)
    if match is None:
        if required:
            prefix = getattr(layer, "prefix", "") or "<unknown>"
            raise KeyError(f"Missing {suffix!r} transform parameter for layer {prefix!r}.")
        return None

    key, value = match
    if target.shape != value.shape:
        raise ValueError(f"Transform parameter {key!r} has shape {tuple(value.shape)}, expected {tuple(target.shape)}.")
    target.copy_(value.to(device=target.device, dtype=target.dtype))
    return key


def _inverse_fp32(matrix: torch.Tensor, *, transpose: bool = False) -> torch.Tensor:
    """Compute a stable inverse while retaining the tested matrix orientation."""
    source = matrix.t() if transpose else matrix
    source = source.to(torch.float32)
    identity = torch.eye(source.shape[0], device=source.device, dtype=torch.float32)
    return torch.linalg.solve(source, identity)


# ---- Pure transform functions shared by Linear scheme and patch sites ----


def transform_flatquant_weight(
    weight: torch.Tensor,
    left_trans: torch.Tensor,
    right_trans: torch.Tensor,
    diag_scale: torch.Tensor | None,
    left_dim: int,
    right_dim: int,
) -> torch.Tensor:
    """FlatQuant weight inverse transform: W' = inv(left) @ (W / diag) @ inv(right).T.

    Returns the transformed weight in float32 with the original shape.
    """
    inv_left = _inverse_fp32(left_trans)
    inv_right_t = _inverse_fp32(right_trans, transpose=True)
    original_shape = weight.shape
    weight_blocked = weight.to(torch.float32).reshape(-1, left_dim, right_dim)
    if diag_scale is not None:
        diag = diag_scale.to(torch.float32).reshape(left_dim, right_dim)
        if not torch.isfinite(diag).all():
            raise ValueError("FlatQuant diag_scale contains NaN or Inf.")
        if torch.any(diag.abs() < 1e-8):
            raise ValueError("FlatQuant diag_scale contains near-zero values.")
        weight_blocked = weight_blocked / diag.unsqueeze(0)
    rotated = torch.matmul(inv_left, weight_blocked)
    rotated = torch.matmul(rotated, inv_right_t)
    return rotated.reshape(original_shape)


def transform_flatquant_activation(
    x: torch.Tensor,
    left_trans: torch.Tensor,
    right_trans: torch.Tensor,
    diag_scale: torch.Tensor | None,
    left_dim: int,
    right_dim: int,
) -> torch.Tensor:
    """FlatQuant activation forward transform: x' = left.T @ (reshape(x) * diag) @ right.

    Returns the transformed activation with the original input shape.
    """
    input_shape = x.shape
    reshaped = x.reshape(-1, left_dim, right_dim)
    if diag_scale is not None:
        reshaped = reshaped * diag_scale.to(x.dtype).reshape(1, left_dim, right_dim)
    transformed = torch.matmul(left_trans.to(x.dtype).transpose(0, 1), reshaped)
    transformed = torch.matmul(transformed, right_trans.to(x.dtype))
    return transformed.reshape(*input_shape)


def transform_lht_weight(
    weight: torch.Tensor,
    transform_weight: torch.Tensor,
    matrix_size: int,
) -> torch.Tensor:
    """LHT weight inverse transform: W' = W @ Q (block-wise).

    AMCT exports the orthogonal matrix Q actually used by forward.  Since
    inv(Q).T == Q for orthogonal matrices, the paired weight transform is
    also W @ Q — no explicit inverse needed.
    """
    original_shape = weight.shape
    weight_blocked = weight.to(torch.float32).reshape(-1, matrix_size)
    rotated = weight_blocked @ transform_weight.to(torch.float32)
    return rotated.reshape(original_shape)


def _get_decompose_dim(size: int, tp_size: int) -> tuple[int, int]:
    """Decompose a feature size into FlatQuant Kronecker dimensions."""
    left_candidate = math.isqrt(size)
    if left_candidate * left_candidate < size:
        left_candidate += 1

    while True:
        difference = left_candidate * left_candidate - size
        right_candidate = math.isqrt(difference)
        if right_candidate * right_candidate == difference:
            break
        left_candidate += 1

    left_dim = left_candidate - right_candidate
    right_dim = left_candidate + right_candidate
    if left_dim + right_dim > MAX_FLATQUANT_TRANSFORM_DIM:
        raise ValueError(
            f"FlatQuant left and right transform dimensions must not exceed {MAX_FLATQUANT_TRANSFORM_DIM} in total."
        )
    if left_dim * tp_size > MAX_FLATQUANT_TRANSFORM_DIM:
        return MAX_FLATQUANT_TRANSFORM_DIM, tp_size * size // MAX_FLATQUANT_TRANSFORM_DIM
    return left_dim, right_dim


def _validate_weight_state(algorithm: str, required: str | None, quant_description: dict[str, Any]) -> None:
    """Enforce fake_mx_weight_state for prequantized checkpoint schemes."""
    if required is None:
        return
    weight_state = quant_description.get("fake_mx_weight_state")
    if weight_state != required:
        raise ValueError(f"{algorithm} fake-MX requires fake_mx_weight_state={required!r}, got {weight_state!r}.")


class _AscendFakeMXLinearMethod(AscendLinearScheme):
    is_fake_mx = True
    mx_format: FakeMXFormat
    algorithm = "rtn"
    prequantized_weight = False
    required_weight_state: str | None = None

    def __init__(self):
        quant_description = _quant_description()
        self.group_size = int(quant_description.get("group_size", 32))
        _validate_weight_state(self.algorithm, self.required_weight_state, quant_description)

    @staticmethod
    def get_weight(input_size: int, output_size: int, params_dtype: torch.dtype) -> dict[str, Any]:
        return {"weight": torch.empty(output_size, input_size, dtype=params_dtype)}

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        tp_rank: int | None = 0,
    ) -> torch.Tensor:
        x = self.transform_activation(layer, x)
        quantized_x = fake_mx_quantize(x, self.mx_format, self.group_size)
        return F.linear(quantized_x, layer.weight, bias)

    def transform_activation(self, layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        return x

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "_fake_mx_weight_processed", False):
            return
        if not self.prequantized_weight:
            layer.weight.data.copy_(fake_mx_quantize(layer.weight.data, self.mx_format, self.group_size))
        layer._fake_mx_weight_processed = True


@register_scheme("W4A4_MXFP4_FAKE", "linear")
class AscendW4A4MXFP4FakeLinearMethod(_AscendFakeMXLinearMethod):
    """MXFP4 QDQ followed by an ordinary floating-point linear operation."""

    mx_format: FakeMXFormat = "mxfp4"


@register_scheme("W8A8_MXFP8_FAKE", "linear")
class AscendW8A8MXFP8FakeLinearMethod(_AscendFakeMXLinearMethod):
    """MXFP8 QDQ followed by an ordinary floating-point linear operation."""

    mx_format: FakeMXFormat = "mxfp8"


class _AscendPrequantizedWeightFakeMXLinearMethod(_AscendFakeMXLinearMethod):
    """Consume an FP checkpoint whose weight already contains QDQ error."""

    prequantized_weight = True
    required_weight_state = "prequantized_qdq"


class _AscendOmniQuantFakeMXLinearMethod(_AscendFakeMXLinearMethod):
    """OmniQuant: per-dimension log-scale transform.

    Loads ``log_scale`` from external params. Weight is scaled up by
    ``exp(log_scale)`` and activation is scaled down, preserving the linear
    transform output while reducing MX QDQ error.
    """

    algorithm = "omniquant"
    supports_pertensor_layer_type = True

    def __init__(self):
        super().__init__()
        quant_description = _quant_description()
        self.params_path = quant_description.get("omniquant_params_path")
        if not self.params_path:
            raise ValueError("OmniQuant requires omniquant_params_path.")
        self.input_size = 0

    def get_weight(self, input_size: int, output_size: int, params_dtype: torch.dtype) -> dict[str, Any]:
        self.input_size = input_size
        return super().get_weight(input_size, output_size, params_dtype)

    def get_pertensor_param(self, params_dtype: torch.dtype, **kwargs: Any) -> dict[str, Any]:
        return {"log_scale": torch.zeros(1, self.input_size, dtype=params_dtype)}

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if not getattr(layer, "_fake_mx_omniquant_processed", False):
            params = _load_transform_params(self.params_path)
            _copy_transform_param(layer.log_scale.data, params, layer, "log_scale")
            scale = torch.exp(layer.log_scale.data.to(torch.float32)).clamp(min=1e-4, max=1e4)
            layer.weight.data.copy_((layer.weight.data.to(torch.float32) * scale).to(layer.weight.data.dtype))
            layer._fake_mx_scale = scale.to(layer.weight.device)
            layer._fake_mx_omniquant_processed = True
        layer.log_scale = torch.nn.Parameter(layer.log_scale.data.contiguous(), requires_grad=False)
        super().process_weights_after_loading(layer)

    def transform_activation(self, layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        scale = getattr(layer, "_fake_mx_scale", None)
        if scale is not None:
            return (x.to(torch.float32) / scale.to(x.device)).to(x.dtype)
        return x


class _AscendAutoRoundFakeMXLinearMethod(_AscendFakeMXLinearMethod):
    """AutoRound: learnable weight clipping and per-element rounding offset.

    Loads ``value`` (rounding offset), ``min_scale``, ``max_scale`` from
    external params. Weight is clipped to a learned range, and the rounding
    offset is passed to ``fake_mx_quantize`` via ``rounding_offset=``.
    """

    algorithm = "autoround"
    supports_pertensor_layer_type = True

    def __init__(self):
        super().__init__()
        quant_description = _quant_description()
        self.params_path = quant_description.get("autoround_params_path")
        if not self.params_path:
            raise ValueError("AutoRound requires autoround_params_path.")
        self.group_size = int(quant_description.get("group_size", 32))
        self.input_size = 0
        self.output_size = 0

    def get_weight(self, input_size: int, output_size: int, params_dtype: torch.dtype) -> dict[str, Any]:
        self.input_size = input_size
        self.output_size = output_size
        return super().get_weight(input_size, output_size, params_dtype)

    def get_pertensor_param(self, params_dtype: torch.dtype, **kwargs: Any) -> dict[str, Any]:
        num_groups = (self.output_size * self.input_size) // self.group_size
        return {
            "value": torch.zeros(self.output_size, self.input_size, dtype=params_dtype),
            "min_scale": torch.ones(num_groups, dtype=torch.float32),
            "max_scale": torch.ones(num_groups, dtype=torch.float32),
        }

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        tp_rank: int | None = 0,
    ) -> torch.Tensor:
        x = self.transform_activation(layer, x)
        quantized_x = fake_mx_quantize(x, self.mx_format, self.group_size)
        return F.linear(quantized_x, layer.weight, bias)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if not getattr(layer, "_fake_mx_autoround_processed", False):
            params = _load_transform_params(self.params_path)
            _copy_transform_param(layer.value.data, params, layer, "value")
            _copy_transform_param(layer.min_scale.data, params, layer, "min_scale")
            _copy_transform_param(layer.max_scale.data, params, layer, "max_scale")

            weight = layer.weight.data.to(torch.float32)
            min_scale = torch.clamp(layer.min_scale.data.to(torch.float32), 0.0, 1.0)
            max_scale = torch.clamp(layer.max_scale.data.to(torch.float32), 0.0, 1.0)
            grouped = weight.reshape(-1, self.group_size)
            group_min = torch.clamp(grouped.amin(dim=-1, keepdim=True), max=0)
            group_max = torch.clamp(grouped.amax(dim=-1, keepdim=True), min=0)
            tuned_min = -(group_min.abs() * min_scale.unsqueeze(-1) if min_scale.ndim == 1 else min_scale)
            tuned_max = group_max * max_scale.unsqueeze(-1) if max_scale.ndim == 1 else max_scale
            max_abs = torch.maximum(tuned_min.abs(), tuned_max.abs()).clamp(min=1e-5)
            layer.weight.data.copy_(
                torch.clamp(grouped, min=-max_abs, max=max_abs).reshape(weight.shape).to(layer.weight.data.dtype)
            )
            layer._fake_mx_autoround_processed = True
        layer.value = torch.nn.Parameter(layer.value.data.contiguous(), requires_grad=False)
        layer.min_scale = torch.nn.Parameter(layer.min_scale.data.contiguous(), requires_grad=False)
        layer.max_scale = torch.nn.Parameter(layer.max_scale.data.contiguous(), requires_grad=False)
        if not getattr(layer, "_fake_mx_weight_processed", False):
            layer.weight.data.copy_(
                fake_mx_quantize(
                    layer.weight.data,
                    self.mx_format,
                    self.group_size,
                    rounding_offset=layer.value.data.to(layer.weight.device),
                )
            )
            layer._fake_mx_weight_processed = True


class _AscendLWCFakeMXLinearMethod(_AscendFakeMXLinearMethod):
    """LWC: learnable weight clipping.

    Loads ``clip_factor_min`` and ``clip_factor_max`` from external params.
    Weight is clipped per-group using learned scaling factors.
    """

    algorithm = "lwc"
    supports_pertensor_layer_type = True

    def __init__(self):
        super().__init__()
        quant_description = _quant_description()
        self.params_path = quant_description.get("lwc_params_path")
        if not self.params_path:
            raise ValueError("LWC requires lwc_params_path.")
        self.group_size = int(quant_description.get("group_size", 32))
        self.input_size = 0
        self.output_size = 0

    def get_weight(self, input_size: int, output_size: int, params_dtype: torch.dtype) -> dict[str, Any]:
        self.input_size = input_size
        self.output_size = output_size
        return super().get_weight(input_size, output_size, params_dtype)

    def get_pertensor_param(self, params_dtype: torch.dtype, **kwargs: Any) -> dict[str, Any]:
        clip_dim = (self.output_size * self.input_size) // self.group_size
        return {
            "clip_factor_min": torch.ones(clip_dim, 1, dtype=torch.float32) * 4.0,
            "clip_factor_max": torch.ones(clip_dim, 1, dtype=torch.float32) * 4.0,
        }

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if not getattr(layer, "_fake_mx_lwc_processed", False):
            params = _load_transform_params(self.params_path)
            _copy_transform_param(layer.clip_factor_min.data, params, layer, "clip_factor_min")
            _copy_transform_param(layer.clip_factor_max.data, params, layer, "clip_factor_max")

            weight = layer.weight.data.to(torch.float32)
            grouped = weight.reshape(-1, self.group_size)
            sigmoid_min = torch.sigmoid(layer.clip_factor_min.data.to(torch.float32))
            sigmoid_max = torch.sigmoid(layer.clip_factor_max.data.to(torch.float32))
            cur_min = grouped.min(dim=-1, keepdim=True)[0] * sigmoid_min
            cur_max = grouped.max(dim=-1, keepdim=True)[0] * sigmoid_max
            layer.weight.data.copy_(
                torch.clamp(grouped, min=cur_min, max=cur_max).reshape(weight.shape).to(layer.weight.data.dtype)
            )
            layer._fake_mx_lwc_processed = True
        layer.clip_factor_min = torch.nn.Parameter(layer.clip_factor_min.data.contiguous(), requires_grad=False)
        layer.clip_factor_max = torch.nn.Parameter(layer.clip_factor_max.data.contiguous(), requires_grad=False)
        super().process_weights_after_loading(layer)


class _AscendLACFakeMXLinearMethod(_AscendFakeMXLinearMethod):
    """LAC: learnable activation clipping.

    Loads ``clip_factor_min``, ``clip_factor_max``, ``maxval``, ``minval``
    from external params. Activation is clipped before QDQ.
    """

    algorithm = "lac"
    supports_pertensor_layer_type = True

    def __init__(self):
        super().__init__()
        quant_description = _quant_description()
        self.params_path = quant_description.get("lac_params_path")
        if not self.params_path:
            raise ValueError("LAC requires lac_params_path.")

    def get_pertensor_param(self, params_dtype: torch.dtype, **kwargs: Any) -> dict[str, Any]:
        return {
            "clip_factor_min": torch.ones(1, dtype=torch.float32) * 4.0,
            "clip_factor_max": torch.ones(1, dtype=torch.float32) * 4.0,
            "maxval": torch.zeros(1, dtype=torch.float32),
            "minval": torch.zeros(1, dtype=torch.float32),
        }

    def transform_activation(self, layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        if not getattr(layer, "_lac_has_params", False):
            return x
        clip_factor_min = layer.clip_factor_min.data.to(x.device, dtype=torch.float32)
        clip_factor_max = layer.clip_factor_max.data.to(x.device, dtype=torch.float32)
        maxval = layer.maxval.data.to(x.device, dtype=torch.float32)
        minval = layer.minval.data.to(x.device, dtype=torch.float32)
        if not getattr(layer, "_lac_range_computed", False):
            if maxval.item() == 0:
                maxval = x.to(torch.float32).amax().clamp(min=1e-5)
                layer.maxval.data.copy_(maxval.to(layer.maxval.dtype))
            if minval.item() == 0:
                minval = x.to(torch.float32).amin().clamp(max=-1e-5)
                layer.minval.data.copy_(minval.to(layer.minval.dtype))
            layer._lac_range_computed = True
        cur_max = maxval * torch.sigmoid(clip_factor_max)
        cur_min = minval * torch.sigmoid(clip_factor_min)
        return torch.clamp(x.to(torch.float32), min=cur_min, max=cur_max).to(x.dtype)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if not getattr(layer, "_fake_mx_lac_processed", False):
            params = _load_transform_params(self.params_path)
            loaded_min = _copy_transform_param(
                layer.clip_factor_min.data, params, layer, "clip_factor_min", required=False
            )
            loaded_max = _copy_transform_param(
                layer.clip_factor_max.data, params, layer, "clip_factor_max", required=False
            )
            _copy_transform_param(layer.maxval.data, params, layer, "maxval", required=False)
            _copy_transform_param(layer.minval.data, params, layer, "minval", required=False)
            layer._lac_has_params = loaded_min is not None or loaded_max is not None
            layer._fake_mx_lac_processed = True
        layer.clip_factor_min = torch.nn.Parameter(layer.clip_factor_min.data.contiguous(), requires_grad=False)
        layer.clip_factor_max = torch.nn.Parameter(layer.clip_factor_max.data.contiguous(), requires_grad=False)
        layer.maxval = torch.nn.Parameter(layer.maxval.data.contiguous(), requires_grad=False)
        layer.minval = torch.nn.Parameter(layer.minval.data.contiguous(), requires_grad=False)
        super().process_weights_after_loading(layer)


@register_scheme("W4A4_MXFP4_OMNIQUANT_FAKE", "linear")
class AscendW4A4MXFP4OmniQuantFakeLinearMethod(_AscendOmniQuantFakeMXLinearMethod):
    mx_format: FakeMXFormat = "mxfp4"


@register_scheme("W8A8_MXFP8_OMNIQUANT_FAKE", "linear")
class AscendW8A8MXFP8OmniQuantFakeLinearMethod(_AscendOmniQuantFakeMXLinearMethod):
    mx_format: FakeMXFormat = "mxfp8"


@register_scheme("W4A4_MXFP4_AUTOROUND_FAKE", "linear")
class AscendW4A4MXFP4AutoRoundFakeLinearMethod(_AscendAutoRoundFakeMXLinearMethod):
    mx_format: FakeMXFormat = "mxfp4"


@register_scheme("W8A8_MXFP8_AUTOROUND_FAKE", "linear")
class AscendW8A8MXFP8AutoRoundFakeLinearMethod(_AscendAutoRoundFakeMXLinearMethod):
    mx_format: FakeMXFormat = "mxfp8"


@register_scheme("W4A4_MXFP4_LWC_FAKE", "linear")
class AscendW4A4MXFP4LWCFakeLinearMethod(_AscendLWCFakeMXLinearMethod):
    mx_format: FakeMXFormat = "mxfp4"


@register_scheme("W8A8_MXFP8_LWC_FAKE", "linear")
class AscendW8A8MXFP8LWCFakeLinearMethod(_AscendLWCFakeMXLinearMethod):
    mx_format: FakeMXFormat = "mxfp8"


@register_scheme("W4A4_MXFP4_LAC_FAKE", "linear")
class AscendW4A4MXFP4LACFakeLinearMethod(_AscendLACFakeMXLinearMethod):
    mx_format: FakeMXFormat = "mxfp4"


@register_scheme("W8A8_MXFP8_LAC_FAKE", "linear")
class AscendW8A8MXFP8LACFakeLinearMethod(_AscendLACFakeMXLinearMethod):
    mx_format: FakeMXFormat = "mxfp8"


class _AscendRHTFakeMXLinearMethod(_AscendFakeMXLinearMethod):
    """RHT for Linear: accepts original BF16 checkpoint, rotates weight at load time.

    Uses a deterministic normalized Hadamard matrix (no random signs, no seed).
    The same butterfly structure as ``scipy.linalg.hadamard(n) / sqrt(n)`` is
    computed in-place via the Fast Walsh-Hadamard Transform (FWHT) without a
    sign diagonal, matching AMCT-final's ``_HadamardTransform``.
    """

    algorithm = "rht"

    def __init__(self):
        super().__init__()
        quant_description = _quant_description()
        self.rht_matrix_size = int(quant_description.get("rht_matrix_size",
                                    quant_description.get("rht_group_size", self.group_size)))

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if not getattr(layer, "_fake_mx_rht_weight_rotated", False):
            layer.weight.data.copy_(
                hadamard_transform(layer.weight.data, self.rht_matrix_size)
            )
            layer._fake_mx_rht_weight_rotated = True
        super().process_weights_after_loading(layer)

    def transform_activation(self, layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        return hadamard_transform(x, self.rht_matrix_size)


@register_scheme("W4A4_MXFP4_RHT_FAKE", "linear")
class AscendW4A4MXFP4RHTFakeLinearMethod(_AscendRHTFakeMXLinearMethod):
    mx_format: FakeMXFormat = "mxfp4"


@register_scheme("W8A8_MXFP8_RHT_FAKE", "linear")
class AscendW8A8MXFP8RHTFakeLinearMethod(_AscendRHTFakeMXLinearMethod):
    mx_format: FakeMXFormat = "mxfp8"


class _AscendHadamardLearningFakeMXLinearMethod(_AscendFakeMXLinearMethod):
    """LHT for Linear: accepts original BF16 checkpoint, loads external transform
    matrix and applies inverse transform to weight at load time."""

    algorithm = "hadamard_learning"
    supports_pertensor_layer_type = True

    def __init__(self):
        super().__init__()
        quant_description = _quant_description()
        self.matrix_size = int(quant_description.get("hadamard_learning_matrix_size", DEFAULT_TRANSFORM_MATRIX_SIZE))
        if self.matrix_size <= 0:
            raise ValueError("hadamard_learning_matrix_size must be positive.")
        self.params_path = quant_description.get("lht_params_path")
        if not self.params_path:
            raise ValueError("Hadamard Learning requires lht_params_path.")

    def get_pertensor_param(self, params_dtype: torch.dtype, **kwargs: Any) -> dict[str, Any]:
        return {
            "transform_weight": torch.eye(
                self.matrix_size,
                dtype=torch.float32,
            )
        }

    def transform_activation(self, layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        return learned_hadamard_transform(x, layer.transform_weight)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if layer.weight.shape[-1] % self.matrix_size:
            raise ValueError(
                f"Hadamard Learning input dimension ({layer.weight.shape[-1]}) must be "
                f"divisible by matrix_size ({self.matrix_size})."
            )
        if not getattr(layer, "_fake_mx_lht_weight_transformed", False):
            params = _load_transform_params(self.params_path)
            key = _copy_transform_param(layer.transform_weight.data, params, layer, "transform_weight")
            logger.debug("LHT: loaded %s", key)
            transformed_weight = transform_lht_weight(layer.weight.data, layer.transform_weight.data, self.matrix_size)
            layer.weight.data.copy_(transformed_weight.to(layer.weight.data.dtype))
            layer._fake_mx_lht_weight_transformed = True
        layer.transform_weight = torch.nn.Parameter(
            layer.transform_weight.data.contiguous(),
            requires_grad=False,
        )
        super().process_weights_after_loading(layer)


@register_scheme("W4A4_MXFP4_HADAMARD_LEARNING_FAKE", "linear")
class AscendW4A4MXFP4HadamardLearningFakeLinearMethod(_AscendHadamardLearningFakeMXLinearMethod):
    mx_format: FakeMXFormat = "mxfp4"


@register_scheme("W8A8_MXFP8_HADAMARD_LEARNING_FAKE", "linear")
class AscendW8A8MXFP8HadamardLearningFakeLinearMethod(_AscendHadamardLearningFakeMXLinearMethod):
    mx_format: FakeMXFormat = "mxfp8"


class _AscendFakeMXFlatQuantLinearMethod(_AscendFakeMXLinearMethod):
    """FlatQuant transform plus fake MX QDQ and floating-point GEMM.

    The checkpoint always supplies original BF16/FP16 ``weight``. Calibration
    supplies the activation transforms, whose inverse is applied to the weight
    online before fake-MX QDQ.
    """

    supports_pertensor_layer_type = True
    algorithm = "flatquant"

    def __init__(self):
        super().__init__()
        quant_description = _quant_description()
        self.max_supported_tp = int(quant_description.get("max_supported_tp", 4))
        self.tp_size = get_tensor_model_parallel_world_size()
        if self.tp_size > self.max_supported_tp:
            raise ValueError(
                f"Fake FlatQuant TP size ({self.tp_size}) exceeds max_supported_tp ({self.max_supported_tp})."
            )
        self.flatquant_params_path = quant_description.get("flatquant_params_path", None)
        if not self.flatquant_params_path:
            raise ValueError("FlatQuant requires flatquant_params_path.")
        self.matrix_size = int(quant_description.get("flatquant_matrix_size", DEFAULT_TRANSFORM_MATRIX_SIZE))
        if self.matrix_size <= 0:
            raise ValueError("flatquant_matrix_size must be positive.")
        self.use_diag_scale = bool(quant_description.get("flatquant_use_diag", True))
        self.input_size = 0

    def _load_external_params(self) -> dict[str, torch.Tensor]:
        return _load_transform_params(self.flatquant_params_path)

    def get_weight(self, input_size: int, output_size: int, params_dtype: torch.dtype) -> dict[str, Any]:
        self.input_size = input_size
        return super().get_weight(input_size, output_size, params_dtype)

    def get_pertensor_param(self, params_dtype: torch.dtype, **kwargs: Any) -> dict[str, Any]:
        layer_type = kwargs.get("layer_type")
        if self.input_size % self.matrix_size == 0:
            # AMCT decomposition: L = D // K, R = K
            if layer_type == "row":
                origin_size = self.input_size * self.tp_size
                left_trans_dim = origin_size // self.matrix_size
                right_trans_dim = self.matrix_size
            else:
                left_trans_dim = self.input_size // self.matrix_size
                right_trans_dim = self.matrix_size
        elif layer_type == "row":
            origin_size = self.input_size * self.tp_size
            _, right_trans_dim = _get_decompose_dim(origin_size // self.max_supported_tp, self.max_supported_tp)
            left_trans_dim = origin_size // right_trans_dim
        else:
            left_trans_dim, right_trans_dim = _get_decompose_dim(self.input_size, 1)
        return {
            "left_trans": torch.eye(left_trans_dim, dtype=torch.float32),
            "right_trans": torch.eye(right_trans_dim, dtype=torch.float32),
            "clip_ratio": torch.ones(1, dtype=torch.float32),
            "diag_scale": torch.ones(self.input_size, dtype=torch.float32),
        }

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        tp_rank: int | None = 0,
    ) -> torch.Tensor:
        left_dim = layer.left_trans.shape[0]
        right_dim = layer.right_trans.shape[0]
        if left_dim * right_dim != x.shape[-1]:
            raise ValueError(
                "FlatQuant transform matrices dimension mismatch: "
                f"left_dim({left_dim}) * right_dim({right_dim}) != in_features({x.shape[-1]})."
            )
        diag = layer.diag_scale if hasattr(layer, "diag_scale") and layer.diag_scale is not None else None
        transformed = transform_flatquant_activation(x, layer.left_trans, layer.right_trans, diag, left_dim, right_dim)
        quantized_x = fake_mx_quantize(transformed, self.mx_format, self.group_size, clip_ratio=layer.aclnn_clip_ratio)
        return F.linear(quantized_x, layer.weight, bias)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if not getattr(layer, "_fake_mx_flatquant_weight_transformed", False):
            ext_params = self._load_external_params()
            layer_prefix = getattr(layer, "prefix", "") or ""
            left_key = _copy_transform_param(layer.left_trans.data, ext_params, layer, "left_trans")
            _copy_transform_param(layer.right_trans.data, ext_params, layer, "right_trans")
            _copy_transform_param(layer.diag_scale.data, ext_params, layer, "diag_scale", required=self.use_diag_scale)
            layer.clip_ratio.data.fill_(1.0)
            logger.debug("FlatQuant: loaded %s for %s", left_key, layer_prefix)
            left_dim = layer.left_trans.data.shape[0]
            right_dim = layer.right_trans.data.shape[0]
            diag = layer.diag_scale.data if hasattr(layer, "diag_scale") and layer.diag_scale is not None else None
            transformed_weight = transform_flatquant_weight(
                layer.weight.data, layer.left_trans.data, layer.right_trans.data, diag, left_dim, right_dim
            )
            layer.weight.data.copy_(transformed_weight.to(layer.weight.data.dtype))
            layer._fake_mx_flatquant_weight_transformed = True
        if isinstance(layer, RowParallelLinear):
            left_dim = layer.left_trans.data.shape[0]
            left_block_size = left_dim // layer.tp_size
            layer.left_trans.data = layer.left_trans.data[
                layer.tp_rank * left_block_size : (layer.tp_rank + 1) * left_block_size,
                layer.tp_rank * left_block_size : (layer.tp_rank + 1) * left_block_size,
            ]

        layer.left_trans = torch.nn.Parameter(layer.left_trans.data.contiguous(), requires_grad=False)
        layer.right_trans = torch.nn.Parameter(layer.right_trans.data.contiguous(), requires_grad=False)
        layer.clip_ratio = torch.nn.Parameter(layer.clip_ratio.data.to(torch.float32), requires_grad=False)
        layer.aclnn_clip_ratio = float(layer.clip_ratio.item())
        if hasattr(layer, "diag_scale") and layer.diag_scale is not None:
            layer.diag_scale = torch.nn.Parameter(
                layer.diag_scale.data.to(torch.float32).contiguous(), requires_grad=False
            )
        super().process_weights_after_loading(layer)


@register_scheme("W4A4_MXFP4_FLATQUANT_FAKE", "linear")
class AscendW4A4MXFP4FakeFlatQuantLinearMethod(_AscendFakeMXFlatQuantLinearMethod):
    mx_format: FakeMXFormat = "mxfp4"


@register_scheme("W8A8_MXFP8_FLATQUANT_FAKE", "linear")
class AscendW8A8MXFP8FakeFlatQuantLinearMethod(_AscendFakeMXFlatQuantLinearMethod):
    mx_format: FakeMXFormat = "mxfp8"


class _AscendFakeMXFusedMoEMethod(AscendMoEScheme):
    mx_format: FakeMXFormat
    quant_type: QuantType = QuantType.NONE
    algorithm = "rtn"
    prequantized_weight = False
    required_weight_state: str | None = None

    def __init__(self):
        quant_description = _quant_description()
        self.group_size = int(quant_description.get("group_size", 32))
        self.rht_matrix_size = int(
            quant_description.get("rht_matrix_size",
                                  quant_description.get("rht_group_size", self.group_size))
        )
        self.hadamard_learning_matrix_size = int(
            quant_description.get("hadamard_learning_matrix_size", DEFAULT_TRANSFORM_MATRIX_SIZE)
        )
        _validate_weight_state(self.algorithm, self.required_weight_state, quant_description)
        self.dynamic_eplb = get_ascend_config().eplb_config.dynamic_eplb
        if self.dynamic_eplb:
            raise NotImplementedError("Fake-MX MoE validation does not support dynamic EPLB.")

    def get_weight(
        self,
        num_experts: int,
        intermediate_size_per_partition: int,
        hidden_sizes: int,
        params_dtype: torch.dtype,
    ) -> dict[str, Any]:
        weights = {
            "w13_weight": torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_sizes,
                dtype=params_dtype,
            ),
            "w2_weight": torch.empty(
                num_experts,
                hidden_sizes,
                intermediate_size_per_partition,
                dtype=params_dtype,
            ),
        }
        if self.algorithm == "hadamard_learning":
            matrix_size = self.hadamard_learning_matrix_size
            weights.update(
                {
                    "w13_transform_weight": torch.empty(
                        num_experts,
                        matrix_size,
                        matrix_size,
                        dtype=params_dtype,
                    ),
                    "w2_transform_weight": torch.empty(
                        num_experts,
                        matrix_size,
                        matrix_size,
                        dtype=params_dtype,
                    ),
                }
            )
        return weights

    @staticmethod
    def get_dynamic_quant_param(
        num_experts: int,
        intermediate_size_per_partition: int,
        hidden_sizes: int,
        params_dtype: torch.dtype,
    ) -> dict[str, Any]:
        return {}

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "_fake_mx_weight_processed", False):
            return
        if self.algorithm == "rht":
            pass
        elif self.algorithm == "hadamard_learning":
            matrix_size = self.hadamard_learning_matrix_size
            if layer.w13_weight.shape[-1] % matrix_size or layer.w2_weight.shape[-1] % matrix_size:
                raise ValueError(
                    f"Hadamard Learning MoE input dimensions must be divisible by matrix_size ({matrix_size})."
                )
            layer.w13_transform_weight = torch.nn.Parameter(
                layer.w13_transform_weight.data.contiguous(), requires_grad=False
            )
            layer.w2_transform_weight = torch.nn.Parameter(
                layer.w2_transform_weight.data.contiguous(), requires_grad=False
            )
        if not self.prequantized_weight:
            layer.w13_weight.data.copy_(fake_mx_quantize(layer.w13_weight.data, self.mx_format, self.group_size))
            layer.w2_weight.data.copy_(fake_mx_quantize(layer.w2_weight.data, self.mx_format, self.group_size))
        # Checkpoints are loaded as [experts, N, K], while the Ascend split
        # grouped-matmul path consumes [experts, K, N].  Keep fake QDQ above
        # in checkpoint layout so MX blocks are formed along the logical K
        # dimension, then match AscendUnquantizedFusedMoEMethod's runtime
        # layout before GMM1/GMM2 execution.
        w13_data = layer.w13_weight.data.transpose(1, 2).contiguous()
        w2_data = layer.w2_weight.data.transpose(1, 2).contiguous()
        layer.w13_weight = torch.nn.Parameter(maybe_trans_nz(w13_data), requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(maybe_trans_nz(w2_data), requires_grad=False)
        layer._fake_mx_weight_processed = True

    @staticmethod
    def _validate_execution_path() -> None:
        if _EXTRA_CTX.moe_comm_type == MoECommType.FUSED_MC2:
            raise NotImplementedError(
                "Fake-MX MoE requires a split dispatch -> GMM1 -> activation QDQ -> "
                "GMM2 -> combine path; FUSED_MC2 is monolithic and has no insertion "
                "point between GMM1 and GMM2. Disable fused MC2 for fake-MX validation."
            )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        is_prefill: bool = True,
        enable_force_load_balance: bool = False,
        log2phy: torch.Tensor | None = None,
        global_redundant_expert_num: int = 0,
        pertoken_scale: Any | None = None,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        mc2_mask: torch.Tensor | None = None,
        tid2eid: Any | None = None,
    ) -> torch.Tensor:
        self._validate_execution_path()
        num_shared_experts = getattr(layer, "n_shared_experts", 0) or 0
        num_logical_experts = get_moe_num_logical_experts(
            layer,
            num_experts,
            global_redundant_expert_num=global_redundant_expert_num,
            num_shared_experts=num_shared_experts,
        )
        if router_logits.shape[1] != num_logical_experts:
            raise AssertionError("Number of global experts mismatch (excluding redundancy)")

        topk_weights, topk_ids = select_experts(
            hidden_states=x,
            router_logits=router_logits,
            top_k=top_k,
            use_grouped_topk=use_grouped_topk,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
            num_experts=num_logical_experts,
            tid2eid=tid2eid,
        )
        if topk_weights is None or topk_ids is None:
            raise RuntimeError("topk_weights and topk_ids must be set before fused MoE execution.")
        if enable_force_load_balance:
            random_matrix = torch.rand(topk_ids.size(0), num_logical_experts, device=topk_ids.device)
            topk_ids = torch.argsort(random_matrix, dim=1)[:, : topk_ids.size(1)].to(topk_ids.dtype)

        topk_weights = topk_weights.to(x.dtype)
        if self.algorithm == "rht":
            x = hadamard_transform(x, self.rht_matrix_size)
        # Per-expert learned matrices can only be selected after token
        # dispatch.  Its FC1 transform and QDQ therefore run in moe_mlp.py.
        quantized_x = (
            x if self.algorithm == "hadamard_learning" else fake_mx_quantize(x, self.mx_format, self.group_size)
        )
        moe_comm_method = _EXTRA_CTX.moe_comm_method
        if moe_comm_method is None:
            raise RuntimeError("Missing MoE communication context.")
        w13_weight_list = getattr(layer, "w13_weight_list", None)
        w2_weight_list = getattr(layer, "w2_weight_list", None)
        w1 = w13_weight_list if isinstance(w13_weight_list, list) else layer.w13_weight
        w2 = w2_weight_list if isinstance(w2_weight_list, list) else layer.w2_weight
        has_bias = bool(getattr(getattr(layer, "moe", None), "has_bias", False))
        return moe_comm_method.fused_experts(
            fused_experts_input=build_fused_experts_input(
                hidden_states=quantized_x,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                w1=w1,
                w2=w2,
                quant_type=QuantType.NONE,
                dynamic_eplb=self.dynamic_eplb,
                expert_map=expert_map,
                global_redundant_expert_num=global_redundant_expert_num,
                mc2_mask=mc2_mask,
                apply_router_weight_on_input=apply_router_weight_on_input,
                log2phy=log2phy,
                pertoken_scale=pertoken_scale,
                activation=activation,
                fake_mx_format=self.mx_format,
                fake_mx_group_size=self.group_size,
                fake_mx_algorithm=self.algorithm,
                fake_mx_rht_signs=None,
                fake_mx_rht_group_size=self.rht_matrix_size,
                fake_mx_w13_transform=getattr(layer, "w13_transform_weight", None),
                fake_mx_w2_transform=getattr(layer, "w2_transform_weight", None),
                w1_bias=layer.w13_bias if has_bias else None,
                w2_bias=layer.w2_bias if has_bias else None,
                w1_scale=None,
                w2_scale=None,
                w1_scale_bias=None,
                w2_scale_bias=None,
                swiglu_limit=getattr(layer, "swiglu_limit", 0.0),
                lora_context=getattr(layer, "_ascend_moe_lora_context", None),
            )
        )


@register_scheme("W4A4_MXFP4_FAKE", "moe")
class AscendW4A4MXFP4FakeFusedMoEMethod(_AscendFakeMXFusedMoEMethod):
    mx_format: FakeMXFormat = "mxfp4"


@register_scheme("W8A8_MXFP8_FAKE", "moe")
class AscendW8A8MXFP8FakeFusedMoEMethod(_AscendFakeMXFusedMoEMethod):
    mx_format: FakeMXFormat = "mxfp8"


class _AscendPrequantizedWeightFakeMXFusedMoEMethod(_AscendFakeMXFusedMoEMethod):
    prequantized_weight = True
    required_weight_state = "prequantized_qdq"


@register_scheme("W4A4_MXFP4_OMNIQUANT_FAKE", "moe")
class AscendW4A4MXFP4OmniQuantFakeFusedMoEMethod(_AscendFakeMXFusedMoEMethod):
    mx_format: FakeMXFormat = "mxfp4"
    algorithm = "omniquant"


@register_scheme("W8A8_MXFP8_OMNIQUANT_FAKE", "moe")
class AscendW8A8MXFP8OmniQuantFakeFusedMoEMethod(_AscendFakeMXFusedMoEMethod):
    mx_format: FakeMXFormat = "mxfp8"
    algorithm = "omniquant"


@register_scheme("W4A4_MXFP4_AUTOROUND_FAKE", "moe")
class AscendW4A4MXFP4AutoRoundFakeFusedMoEMethod(_AscendFakeMXFusedMoEMethod):
    mx_format: FakeMXFormat = "mxfp4"
    algorithm = "autoround"


@register_scheme("W8A8_MXFP8_AUTOROUND_FAKE", "moe")
class AscendW8A8MXFP8AutoRoundFakeFusedMoEMethod(_AscendFakeMXFusedMoEMethod):
    mx_format: FakeMXFormat = "mxfp8"
    algorithm = "autoround"


@register_scheme("W4A4_MXFP4_RHT_FAKE", "moe")
class AscendW4A4MXFP4RHTFakeFusedMoEMethod(_AscendFakeMXFusedMoEMethod):
    """RHT for MoE: requires pre-rotated checkpoint (rht_rotated_fp).
    Unlike Linear RHT, MoE does not rotate weights at load time because
    per-expert rotation must align with the dispatch/GMM execution path."""

    mx_format: FakeMXFormat = "mxfp4"
    algorithm = "rht"
    required_weight_state = "rht_rotated_fp"


@register_scheme("W8A8_MXFP8_RHT_FAKE", "moe")
class AscendW8A8MXFP8RHTFakeFusedMoEMethod(_AscendFakeMXFusedMoEMethod):
    mx_format: FakeMXFormat = "mxfp8"
    algorithm = "rht"
    required_weight_state = "rht_rotated_fp"


@register_scheme("W4A4_MXFP4_HADAMARD_LEARNING_FAKE", "moe")
class AscendW4A4MXFP4HadamardLearningFakeFusedMoEMethod(_AscendFakeMXFusedMoEMethod):
    """LHT for MoE: requires pre-transformed checkpoint (hadamard_learning_transformed_fp).
    Unlike Linear LHT, MoE does not apply inverse transform at load time
    because per-expert transform matrices are selected after token dispatch."""

    mx_format: FakeMXFormat = "mxfp4"
    algorithm = "hadamard_learning"
    required_weight_state = "hadamard_learning_transformed_fp"


@register_scheme("W8A8_MXFP8_HADAMARD_LEARNING_FAKE", "moe")
class AscendW8A8MXFP8HadamardLearningFakeFusedMoEMethod(_AscendFakeMXFusedMoEMethod):
    mx_format: FakeMXFormat = "mxfp8"
    algorithm = "hadamard_learning"
    required_weight_state = "hadamard_learning_transformed_fp"


def _build_flatquant_sidecar_key(
    layer_prefix: str, expert_idx: int, fc_name: str, comp: str
) -> str:
    """Build sidecar tensor key for a per-expert FlatQuant state.

    Format: layers.{N}.experts.{E}.{fc}.{comp_short}
    Example: layers.0.experts.17.fc1.left_trans
    """
    parts = layer_prefix.split(".")
    layer_idx = None
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            layer_idx = parts[i + 1]
            break
    if layer_idx is None:
        return f"{layer_prefix}.{fc_name}.{comp}"

    comp_short = "diag" if comp == "diag_scale" else comp
    return f"layers.{layer_idx}.experts.{expert_idx}.{fc_name}.{comp_short}"


class _AscendFakeMXFlatQuantFusedMoEMethod(_AscendFakeMXFusedMoEMethod):
    """FlatQuant transform + fake MX QDQ for routed MoE experts.

    Each routed expert has independent FC1 and FC2 FlatQuant state
    (left_trans, right_trans, diag_scale).  At load time the weight is
    inverse-transformed and QDQ'd per expert; at forward time the
    activation is forward-transformed and QDQ'd per expert segment.

    Shared expert is NOT handled here — it uses the Linear direct W4A4
    scheme via ``module_quant_overrides``.
    """

    algorithm = "flatquant"

    def __init__(self):
        super().__init__()
        quant_description = _quant_description()
        self.flatquant_params_path = quant_description.get("flatquant_params_path")
        if not self.flatquant_params_path:
            raise ValueError("MoE FlatQuant requires flatquant_params_path.")
        self.matrix_size = int(
            quant_description.get("flatquant_matrix_size", DEFAULT_TRANSFORM_MATRIX_SIZE)
        )
        self.use_diag_scale = bool(quant_description.get("flatquant_use_diag", True))

    def _decompose_dim(self, dim: int) -> tuple[int, int]:
        """Decompose a feature dimension into FlatQuant Kronecker dims.

        Prefer ``matrix_size`` as right_dim when divisible, matching the
        Dense FlatQuant path.  Fall back to ``_get_decompose_dim`` otherwise.
        """
        if dim % self.matrix_size == 0:
            return dim // self.matrix_size, self.matrix_size
        return _get_decompose_dim(dim, 1)

    def get_weight(
        self,
        num_experts: int,
        intermediate_size_per_partition: int,
        hidden_sizes: int,
        params_dtype: torch.dtype,
    ) -> dict[str, Any]:
        weights = super().get_weight(
            num_experts, intermediate_size_per_partition, hidden_sizes, params_dtype
        )
        # FC1 K = hidden_sizes, FC2 K = intermediate_size_per_partition.
        # AMCT Kronecker decomposition: left_dim * right_dim = K.
        # Prefer matrix_size as right_dim when K is divisible (matches Dense path).
        fc1_left_dim, fc1_right_dim = self._decompose_dim(hidden_sizes)
        fc2_left_dim, fc2_right_dim = self._decompose_dim(intermediate_size_per_partition)
        weights.update({
            "fc1_left_trans": torch.eye(fc1_left_dim, dtype=torch.float32).unsqueeze(0).repeat(num_experts, 1, 1),
            "fc1_right_trans": torch.eye(fc1_right_dim, dtype=torch.float32).unsqueeze(0).repeat(num_experts, 1, 1),
            "fc1_diag_scale": torch.ones(num_experts, hidden_sizes, dtype=torch.float32),
            "fc2_left_trans": torch.eye(fc2_left_dim, dtype=torch.float32).unsqueeze(0).repeat(num_experts, 1, 1),
            "fc2_right_trans": torch.eye(fc2_right_dim, dtype=torch.float32).unsqueeze(0).repeat(num_experts, 1, 1),
            "fc2_diag_scale": torch.ones(num_experts, intermediate_size_per_partition, dtype=torch.float32),
        })
        return weights

    def _load_per_expert_state(
        self, layer: torch.nn.Module, ext_params: dict[str, torch.Tensor], fc_name: str,
        expert_map: torch.Tensor | None,
    ) -> None:
        """Load FlatQuant state from sidecar into layer parameters.

        ``expert_map`` maps logical expert ID to local physical slot
        (``expert_map[logical_id] = physical_slot`` or ``-1`` if not on this
        rank).  When ``expert_map`` is None (EP=1), physical slot equals
        logical ID and no reverse lookup is needed.
        """
        layer_prefix = getattr(layer, "prefix", "") or ""
        num_experts = getattr(layer, "fc1_left_trans").shape[0]

        # Build physical_slot -> logical_expert_id reverse mapping.
        if expert_map is not None:
            phy_to_logical: dict[int, int] = {}
            for logical_id in range(expert_map.numel()):
                slot = int(expert_map[logical_id].item())
                if slot != -1:
                    phy_to_logical[slot] = logical_id
        else:
            phy_to_logical = None

        for comp in ["left_trans", "right_trans", "diag_scale"]:
            param = getattr(layer, f"{fc_name}_{comp}")
            for slot in range(num_experts):
                logical_e = slot if phy_to_logical is None else phy_to_logical.get(slot, slot)
                key = _build_flatquant_sidecar_key(layer_prefix, logical_e, fc_name, comp)
                if key in ext_params:
                    param.data[slot].copy_(
                        ext_params[key].to(device=param.device, dtype=param.dtype)
                    )
                else:
                    logger.warning_once(
                        "FlatQuant sidecar missing key %s; using identity.", key
                    )

    def _inverse_transform_and_qdq_weight(
        self,
        weight: torch.Tensor,
        left_trans: torch.Tensor,
        right_trans: torch.Tensor,
        diag_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Apply FlatQuant inverse weight transform, then MXFP4 QDQ."""
        left_dim = left_trans.shape[0]
        right_dim = right_trans.shape[0]
        transformed = transform_flatquant_weight(
            weight, left_trans, right_trans, diag_scale, left_dim, right_dim
        )
        return fake_mx_quantize(transformed.to(weight.dtype), self.mx_format, self.group_size)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "_fake_mx_flatquant_processed", False):
            return

        ext_params = _load_transform_params(self.flatquant_params_path)

        # Obtain expert_map for physical-to-logical reverse lookup.
        expert_map = getattr(layer, "_expert_map", None)

        # Load FlatQuant state from sidecar into layer parameters.
        self._load_per_expert_state(layer, ext_params, "fc1", expert_map)
        self._load_per_expert_state(layer, ext_params, "fc2", expert_map)

        # Per-expert inverse weight transform + QDQ.
        num_experts = layer.w13_weight.shape[0]
        for e in range(num_experts):
            # FC1: w13[e] shape [2*I, H], K=H
            layer.w13_weight.data[e] = self._inverse_transform_and_qdq_weight(
                layer.w13_weight.data[e],
                layer.fc1_left_trans.data[e],
                layer.fc1_right_trans.data[e],
                layer.fc1_diag_scale.data[e],
            )
            # FC2: w2[e] shape [H, I], K=I
            layer.w2_weight.data[e] = self._inverse_transform_and_qdq_weight(
                layer.w2_weight.data[e],
                layer.fc2_left_trans.data[e],
                layer.fc2_right_trans.data[e],
                layer.fc2_diag_scale.data[e],
            )

        # Convert to GMM layout [E, K, N] (same as parent).
        w13_data = layer.w13_weight.data.transpose(1, 2).contiguous()
        w2_data = layer.w2_weight.data.transpose(1, 2).contiguous()
        layer.w13_weight = torch.nn.Parameter(maybe_trans_nz(w13_data), requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(maybe_trans_nz(w2_data), requires_grad=False)

        # Freeze FlatQuant state as Parameters.
        for param_name in [
            "fc1_left_trans", "fc1_right_trans", "fc1_diag_scale",
            "fc2_left_trans", "fc2_right_trans", "fc2_diag_scale",
        ]:
            param = getattr(layer, param_name)
            setattr(layer, param_name, torch.nn.Parameter(param.data.contiguous(), requires_grad=False))

        layer._fake_mx_flatquant_processed = True
        layer._fake_mx_weight_processed = True

    def _pack_fc_state(self, layer: torch.nn.Module, fc_name: str) -> dict[str, torch.Tensor]:
        """Pack per-expert FlatQuant state for one FC into a dict for runtime."""
        state = {
            "left_trans": getattr(layer, f"{fc_name}_left_trans"),
            "right_trans": getattr(layer, f"{fc_name}_right_trans"),
        }
        if self.use_diag_scale:
            state["diag_scale"] = getattr(layer, f"{fc_name}_diag_scale")
        return state

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        is_prefill: bool = True,
        enable_force_load_balance: bool = False,
        log2phy: torch.Tensor | None = None,
        global_redundant_expert_num: int = 0,
        pertoken_scale: Any | None = None,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        mc2_mask: torch.Tensor | None = None,
        tid2eid: Any | None = None,
    ) -> torch.Tensor:
        self._validate_execution_path()
        num_shared_experts = getattr(layer, "n_shared_experts", 0) or 0
        num_logical_experts = get_moe_num_logical_experts(
            layer,
            num_experts,
            global_redundant_expert_num=global_redundant_expert_num,
            num_shared_experts=num_shared_experts,
        )
        if router_logits.shape[1] != num_logical_experts:
            raise AssertionError("Number of global experts mismatch (excluding redundancy)")

        # Router sees ORIGINAL BF16 x — no transform, no QDQ.
        topk_weights, topk_ids = select_experts(
            hidden_states=x,
            router_logits=router_logits,
            top_k=top_k,
            use_grouped_topk=use_grouped_topk,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
            num_experts=num_logical_experts,
            tid2eid=tid2eid,
        )
        if topk_weights is None or topk_ids is None:
            raise RuntimeError("topk_weights and topk_ids must be set before fused MoE execution.")
        if enable_force_load_balance:
            random_matrix = torch.rand(topk_ids.size(0), num_logical_experts, device=topk_ids.device)
            topk_ids = torch.argsort(random_matrix, dim=1)[:, : topk_ids.size(1)].to(topk_ids.dtype)

        topk_weights = topk_weights.to(x.dtype)
        # FC1 activation transform + QDQ happens in moe_mlp.py per-expert segment.
        # Pass ORIGINAL x and FlatQuant state via build_fused_experts_input.
        moe_comm_method = _EXTRA_CTX.moe_comm_method
        if moe_comm_method is None:
            raise RuntimeError("Missing MoE communication context.")
        w13_weight_list = getattr(layer, "w13_weight_list", None)
        w2_weight_list = getattr(layer, "w2_weight_list", None)
        w1 = w13_weight_list if isinstance(w13_weight_list, list) else layer.w13_weight
        w2 = w2_weight_list if isinstance(w2_weight_list, list) else layer.w2_weight
        has_bias = bool(getattr(getattr(layer, "moe", None), "has_bias", False))
        return moe_comm_method.fused_experts(
            fused_experts_input=build_fused_experts_input(
                hidden_states=x,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                w1=w1,
                w2=w2,
                quant_type=QuantType.NONE,
                dynamic_eplb=self.dynamic_eplb,
                expert_map=expert_map,
                global_redundant_expert_num=global_redundant_expert_num,
                mc2_mask=mc2_mask,
                apply_router_weight_on_input=apply_router_weight_on_input,
                log2phy=log2phy,
                pertoken_scale=pertoken_scale,
                activation=activation,
                fake_mx_format=self.mx_format,
                fake_mx_group_size=self.group_size,
                fake_mx_algorithm="flatquant",
                fake_mx_rht_signs=None,
                fake_mx_rht_group_size=self.rht_matrix_size,
                fake_mx_w13_transform=None,
                fake_mx_w2_transform=None,
                fake_mx_flatquant_fc1_state=self._pack_fc_state(layer, "fc1"),
                fake_mx_flatquant_fc2_state=self._pack_fc_state(layer, "fc2"),
                w1_bias=layer.w13_bias if has_bias else None,
                w2_bias=layer.w2_bias if has_bias else None,
                w1_scale=None,
                w2_scale=None,
                w1_scale_bias=None,
                w2_scale_bias=None,
                swiglu_limit=getattr(layer, "swiglu_limit", 0.0),
                lora_context=getattr(layer, "_ascend_moe_lora_context", None),
            )
        )


@register_scheme("W4A4_MXFP4_FLATQUANT_FAKE", "moe")
class AscendW4A4MXFP4FakeFlatQuantFusedMoEMethod(_AscendFakeMXFlatQuantFusedMoEMethod):
    """W4A4 MXFP4 FlatQuant fake-QDQ for FusedMoE (routed experts only)."""
    mx_format: FakeMXFormat = "mxfp4"


@register_scheme("W8A8_MXFP8_FLATQUANT_FAKE", "moe")
class AscendW8A8MXFP8FakeFlatQuantFusedMoEMethod(_AscendFakeMXFlatQuantFusedMoEMethod):
    mx_format: FakeMXFormat = "mxfp8"
