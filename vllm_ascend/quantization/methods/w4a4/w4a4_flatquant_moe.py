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
"""MXFP4 FlatQuant MoE scheme for vllm-ascend (graph-capturable).

This file only implements ``W4A4_MXFP4_FLATQUANT`` for MoE layers.

The important difference from a naive FlatQuant MoE implementation is that the
forward path never loops over experts in Python and never calls
``repeat_interleave`` (which synchronizes on its tensor ``repeats`` during ACL
graph capture).  Rows are mapped to experts with ``cumsum + bucketize``, the
per-expert Kronecker transform uses two batched matmuls (``bmm``), and the
expert MLP uses ``npu_grouped_matmul`` like the normal W4A4-MXFP4 MoE.

Transform math
--------------
For expert ``e`` with token slice ``X_e`` of shape ``[c_e, M, N]``::

    X'_e = L_e^T (X_e * diag_e) R_e

On disk ``left_trans`` is ``L``; this file stores it transposed as ``L^T`` and
does ``bmm(L^T_rows, X)`` followed by ``bmm(..., R_rows)``.  ``diag`` is
expanded with ``diag[ids]`` where ``ids`` comes from ``bucketize``.

Checkpoint contract (per expert, from ``amct_pytorch.deploy``)
--------------------------------------------------------------
For each of ``gate_proj`` / ``up_proj`` / ``down_proj``::

    weight        uint8 [out, in/2]      # two 4-bit values per byte
    weight_scale  uint8 [out, in/32]     # e8m0
    left_trans    f32   [left, left]     # on disk L; stored as L^T after loading
    right_trans   f32   [right, right]
    diag_scale    f32   [in]
    clip_ratio    f32   [1]              # AMCT export writes 1.0

``w13`` is ``[gate; up]`` along the output dimension (gate first, up second).
"""

from __future__ import annotations

import torch
import torch_npu
from vllm.config import get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.ops.fused_moe.dataclass.fused_experts import build_fused_experts_input
from vllm_ascend.ops.fused_moe.dataclass.moe_mlp import MoEMlpComputeInput
from vllm_ascend.ops.fused_moe.moe_utils import maybe_normalize_mxfp_scale_layout

from ..base import AscendMoEScheme, QuantType
from ..registry import register_scheme

# Parameters whose checkpoint tensor is written per expert.  The default
# fused-MoE loader only understands per-channel/per-group weights and scales,
# so ``method_adapters.py`` must route these names through
# ``moe_param_weight_loader`` below.
PER_EXPERT_PARAMS = frozenset(
    {
        "w13_diag_scale",
        "w13_left_trans",
        "w13_right_trans",
        "w13_clip_ratio",
        "w2_diag_scale",
        "w2_left_trans",
        "w2_right_trans",
        "w2_clip_ratio",
    }
)


def _solve_kronecker_decompose(n: int) -> tuple[int, int]:
    """Fallback only: AMCT writes the real dims into ``flatquant_trans_dims``."""
    import math

    a = int(math.sqrt(n))
    if a * a < n:
        a += 1
    while True:
        tmp = a * a - n
        b = int(math.sqrt(tmp))
        if b * b == tmp:
            break
        a += 1
    return a - b, a + b


def _group_counts(group_list: torch.Tensor, group_list_type: int, num_experts: int) -> torch.Tensor:
    """Return per-expert token counts as an int64 device tensor.

    No host read: type 0 is differentiated on device, type 1 is already counts.
    Type 2 is a (expert, token) table and is not used by this path.  The
    dispatcher may hand us a padded buffer, so slice to the real expert count
    (a static Python int).
    """
    if group_list_type == 0:
        counts = torch.diff(group_list, prepend=group_list.new_zeros(1))
    elif group_list_type == 1:
        counts = group_list
    else:
        raise ValueError(f"FlatQuant MoE does not support group_list_type={group_list_type}")
    return counts[:num_experts].to(torch.int64)


def _expert_ids(counts: torch.Tensor, tokens: int, num_experts: int) -> torch.Tensor:
    """Map each dispatched row to its expert without a host read.

    ``repeat_interleave`` synchronizes on the tensor ``repeats`` during ACL
    graph capture, so use cumsum + bucketize instead (the same fix the A5
    iteration report records).  Rows beyond ``sum(counts)`` are padding; clamp
    them to the last expert so the gather stays in range.  Downstream grouped
    ops only consume the real rows.
    """
    cumsum = torch.cumsum(counts, dim=0)
    ids = torch.bucketize(
        torch.arange(tokens, device=counts.device, dtype=torch.int64),
        cumsum,
        right=True,
    )
    return ids.clamp_max(num_experts - 1)


def _per_expert_param_loader(param_name: str):
    """Copy one checkpoint tensor into its expert slot of a stacked parameter."""

    def loader(
        param,
        loaded_weight,
        weight_name,
        shard_id,
        expert_id,
        return_success: bool = False,
    ):
        slot = param.data[expert_id]
        slot.copy_(loaded_weight.reshape(slot.shape).to(slot.dtype))
        return True if return_success else None

    return loader if param_name in PER_EXPERT_PARAMS else None


class _AscendFlatQuantMXFP4MoEFusedMoEMethod(AscendMoEScheme):
    """Shared plumbing: parameter declaration, loading and the grouped MLP."""

    # NONE is deliberate: the MoE dispatcher must hand us BF16 tokens.  The
    # FlatQuant transform runs before activation quantization.
    quant_type: QuantType = QuantType.NONE
    act_quant_type = None
    fused_activations = frozenset()
    supports_eplb = False

    def __init__(self):
        quant_description = get_current_vllm_config().quant_config.quant_description
        self.group_size = int(quant_description.get("group_size", 0) or 0)
        if self.group_size <= 0:
            raise ValueError("W4A4_MXFP4_FLATQUANT requires group_size in quant_model_description.json")

        # Do not recompute the Kronecker split here: AMCT uses matrix_size=128,
        # which is not the balanced split returned by
        # solve_kronecker_decompose().  The description carries the exact dims.
        dims = quant_description.get("flatquant_trans_dims") or {}
        self._w13_dims = tuple(dims.get("w13", ()))
        self._w2_dims = tuple(dims.get("w2", ()))

        max_tp = int(quant_description.get("max_supported_tp", 1))
        tp_size = get_tensor_model_parallel_world_size()
        if tp_size > max_tp:
            raise ValueError(
                f"For {self.__class__.__name__}, TP size ({tp_size}) exceeds "
                f"max_supported_tp ({max_tp}) from quant_model_description.json"
            )

    # ---- parameter declaration -------------------------------------------------
    def get_weight(
        self,
        num_experts: int,
        intermediate_size_per_partition: int,
        hidden_sizes: int,
        params_dtype: torch.dtype,
    ) -> dict:
        self._params_dtype = params_dtype
        if hidden_sizes % 2 or intermediate_size_per_partition % 2:
            raise ValueError(
                "mxfp4 packing needs even hidden/intermediate sizes, got "
                f"{hidden_sizes}/{intermediate_size_per_partition}"
            )
        return {
            "w13_weight": torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_sizes // 2,
                dtype=torch.uint8,
            ),
            "w2_weight": torch.empty(
                num_experts,
                hidden_sizes,
                intermediate_size_per_partition // 2,
                dtype=torch.uint8,
            ),
        }

    def _trans_dims(
        self, hidden_sizes: int, intermediate_size_per_partition: int
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        w13 = self._w13_dims or _solve_kronecker_decompose(hidden_sizes)
        w2 = self._w2_dims or _solve_kronecker_decompose(intermediate_size_per_partition)
        return tuple(w13), tuple(w2)

    def get_dynamic_quant_param(
        self,
        num_experts: int,
        intermediate_size_per_partition: int,
        hidden_sizes: int,
        params_dtype: torch.dtype,
    ) -> dict:
        w13_dims, w2_dims = self._trans_dims(hidden_sizes, intermediate_size_per_partition)
        # Keep the resolved dims for the forward path (older checkpoints may
        # not carry flatquant_trans_dims in the description).
        self._w13_dims = tuple(w13_dims)
        self._w2_dims = tuple(w2_dims)
        if hidden_sizes % self.group_size or intermediate_size_per_partition % self.group_size:
            raise ValueError(
                "mxfp4 scale group must divide hidden/intermediate sizes, got "
                f"group_size={self.group_size}, "
                f"hidden={hidden_sizes}, inter={intermediate_size_per_partition}"
            )
        return {
            # diag_scale is applied to the activation before the Kronecker
            # transform: x' = L^T (x * diag) R.
            # diag is optional in the checkpoint; default to ones so a
            # checkpoint without diag_scale runs the neutral path.
            "w13_diag_scale": torch.ones(num_experts, hidden_sizes, dtype=torch.float32),
            "w2_diag_scale": torch.ones(num_experts, intermediate_size_per_partition, dtype=torch.float32),
            "w13_weight_scale": torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_sizes // self.group_size,
                dtype=torch.uint8,
            ),
            "w2_weight_scale": torch.empty(
                num_experts,
                hidden_sizes,
                intermediate_size_per_partition // self.group_size,
                dtype=torch.uint8,
            ),
            "w13_left_trans": torch.empty(num_experts, w13_dims[0], w13_dims[0], dtype=torch.float32),
            "w13_right_trans": torch.empty(num_experts, w13_dims[1], w13_dims[1], dtype=torch.float32),
            "w13_clip_ratio": torch.ones(num_experts, dtype=torch.float32),
            "w2_left_trans": torch.empty(num_experts, w2_dims[0], w2_dims[0], dtype=torch.float32),
            "w2_right_trans": torch.empty(num_experts, w2_dims[1], w2_dims[1], dtype=torch.float32),
            "w2_clip_ratio": torch.ones(num_experts, dtype=torch.float32),
        }

    def moe_param_weight_loader(self, param_name: str):
        return _per_expert_param_loader(param_name)

    # ---- post-loading ----------------------------------------------------------
    @staticmethod
    def _pack_scale_pairs(scale: torch.Tensor) -> torch.Tensor:
        """[E, N, K/32] uint8 -> [E, K/64, N, 2] for the MXFP4 grouped matmul."""
        experts, out_dim, groups = scale.shape
        if groups % 2 != 0:
            scale = torch.nn.functional.pad(scale, (0, 1), mode="constant", value=0)
            groups += 1
        return scale.reshape(experts, out_dim, groups // 2, 2).transpose(1, 2)

    def process_weights_after_loading(self, layer) -> None:
        # The bmm path computes L^T x, so store left_trans as L^T (disk L is
        # transposed here), matching the old per-expert implementation.
        dtype = getattr(self, "_params_dtype", torch.bfloat16)
        layer.w13_left_trans = torch.nn.Parameter(layer.w13_left_trans.data.to(dtype).transpose(1, 2).contiguous())
        layer.w13_right_trans = torch.nn.Parameter(layer.w13_right_trans.data.to(dtype).contiguous())
        layer.w2_left_trans = torch.nn.Parameter(layer.w2_left_trans.data.to(dtype).transpose(1, 2).contiguous())
        layer.w2_right_trans = torch.nn.Parameter(layer.w2_right_trans.data.to(dtype).contiguous())
        layer.w13_diag_scale = torch.nn.Parameter(layer.w13_diag_scale.data.to(dtype).contiguous())
        layer.w2_diag_scale = torch.nn.Parameter(layer.w2_diag_scale.data.to(dtype).contiguous())

        # Same layout as the normal W4A4-MXFP4 MoE: grouped matmul consumes
        # [E, K/2, N] packed weights and [E, K/64, N, 2] e8m0 scales.
        layer.w13_weight.data = layer.w13_weight.data.transpose(1, 2)
        layer.w2_weight.data = layer.w2_weight.data.transpose(1, 2)
        layer.w13_weight_scale.data = self._pack_scale_pairs(layer.w13_weight_scale.data)
        layer.w2_weight_scale.data = self._pack_scale_pairs(layer.w2_weight_scale.data)

        # The graph-safe path uses npu_dynamic_mx_quant, which is equivalent to
        # npu_kronecker_quant only when clip_ratio == 1.0.  Fail loudly instead
        # of silently serving a checkpoint that needs the clip path.
        for name in ("w13_clip_ratio", "w2_clip_ratio"):
            clip = getattr(layer, name, None)
            if clip is not None and not torch.allclose(clip, torch.ones_like(clip)):
                raise NotImplementedError(
                    f"{self.__class__.__name__} only supports clip_ratio=1.0; {name} contains values other than 1.0"
                )

    # ---- forward ---------------------------------------------------------------
    def apply(
        self,
        layer,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts=None,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        moe_comm_method = _EXTRA_CTX.moe_comm_method
        return moe_comm_method.fused_experts(
            fused_experts_input=build_fused_experts_input(
                hidden_states=x,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                layer=layer,
                quant_type=self.quant_type,
                dynamic_eplb=False,
                expert_map=getattr(layer, "ascend_expert_map", None),
                global_redundant_expert_num=getattr(layer, "global_redundant_expert_num", 0),
                mc2_mask=getattr(layer, "ascend_mc2_mask", None),
                apply_router_weight_on_input=getattr(layer, "apply_router_weight_on_input", False),
                pertoken_scale=getattr(layer, "ascend_pertoken_scale", None),
                activation=getattr(layer, "activation", None),
            ),
            quant_method=self,
        )

    def _flatquant_transform(
        self,
        x: torch.Tensor,
        left: torch.Tensor,
        right: torch.Tensor,
        diag: torch.Tensor,
        counts: torch.Tensor,
        dims: tuple[int, int],
    ) -> torch.Tensor:
        """Apply the per-expert ``L^T (x * diag) R`` with batched matmuls."""
        if not dims:
            dims = _solve_kronecker_decompose(x.shape[-1])
        left_dim, right_dim = dims
        tokens = x.shape[0]
        if x.shape[-1] != left_dim * right_dim:
            raise ValueError(f"FlatQuant transform dims {left_dim}x{right_dim} do not cover input {tuple(x.shape)}.")

        ids = _expert_ids(counts, tokens, left.shape[0])
        x = x * diag[ids]

        # left is L^T: bmm(L^T, x) = L^T x; then bmm(..., R).
        left_rows = left[ids]
        right_rows = right[ids]
        out = torch.bmm(left_rows, x.reshape(tokens, left_dim, right_dim))
        out = torch.bmm(out, right_rows)
        return out.reshape(tokens, left_dim * right_dim)

    @staticmethod
    def _quantize_mxfp4(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_quant, x_scale = torch_npu.npu_dynamic_mx_quant(x, dst_type=torch_npu.float4_e2m1fn_x2)
        return x_quant, maybe_normalize_mxfp_scale_layout(x_scale)

    def apply_gmm1(self, mlp_compute_input: MoEMlpComputeInput) -> torch.Tensor:
        hidden_states = mlp_compute_input.hidden_states
        layer = mlp_compute_input.layer
        counts = _group_counts(
            mlp_compute_input.group_list,
            mlp_compute_input.group_list_type,
            layer.w13_left_trans.shape[0],
        )
        x = self._flatquant_transform(
            hidden_states,
            layer.w13_left_trans,
            layer.w13_right_trans,
            layer.w13_diag_scale,
            counts,
            self._w13_dims,
        )
        x_quant, x_scale = self._quantize_mxfp4(x)
        out_dtype = hidden_states.dtype if hidden_states.dtype in (torch.bfloat16, torch.float16) else torch.bfloat16
        return torch_npu.npu_grouped_matmul(
            x=[x_quant],
            weight=[layer.w13_weight],
            scale=[layer.w13_weight_scale],
            per_token_scale=[x_scale],
            split_item=2,
            group_type=0,
            group_list=counts,
            group_list_type=1,
            output_dtype=out_dtype,
            scale_dtype=torch_npu.float8_e8m0fnu,
            per_token_scale_dtype=torch_npu.float8_e8m0fnu,
            x_dtype=torch_npu.float4_e2m1fn_x2,
            weight_dtype=torch_npu.float4_e2m1fn_x2,
        )[0]

    def apply_act_quant(self, mlp_compute_input: MoEMlpComputeInput, hidden_states: torch.Tensor):
        layer = mlp_compute_input.layer
        counts = _group_counts(
            mlp_compute_input.group_list,
            mlp_compute_input.group_list_type,
            layer.w2_left_trans.shape[0],
        )
        x = self._flatquant_transform(
            hidden_states,
            layer.w2_left_trans,
            layer.w2_right_trans,
            layer.w2_diag_scale,
            counts,
            self._w2_dims,
        )
        return self._quantize_mxfp4(x)

    def apply_gmm2(self, mlp_compute_input: MoEMlpComputeInput, hidden_states, act_out_scale):
        layer = mlp_compute_input.layer
        counts = _group_counts(
            mlp_compute_input.group_list,
            mlp_compute_input.group_list_type,
            layer.w2_left_trans.shape[0],
        )
        out_dtype = mlp_compute_input.hidden_states.dtype
        if out_dtype not in (torch.bfloat16, torch.float16):
            out_dtype = torch.bfloat16
        return torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[layer.w2_weight],
            scale=[layer.w2_weight_scale],
            bias=None,
            per_token_scale=[act_out_scale],
            split_item=2,
            group_list_type=1,
            group_type=0,
            group_list=counts,
            output_dtype=out_dtype,
            scale_dtype=torch_npu.float8_e8m0fnu,
            per_token_scale_dtype=torch_npu.float8_e8m0fnu,
            x_dtype=torch_npu.float4_e2m1fn_x2,
            weight_dtype=torch_npu.float4_e2m1fn_x2,
        )[0]


@register_scheme("W4A4_MXFP4_FLATQUANT", "moe")
class AscendW4A4MXFP4FlatQuantFusedMoEMethod(_AscendFlatQuantMXFP4MoEFusedMoEMethod):
    """mxfp4 FlatQuant MoE: fp4 activations x fp4 weights, e8m0 per 32 elements."""
