# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
# This file is a part of the vllm-ascend project.


import torch
import torch_npu
from torch.nn.functional import pad
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.device.mxfp_compat import (
    ensure_mxfp8_moe_available,
)
from vllm_ascend.ops.activation import AscendSwigluOAIAndMul, AscendSwigluStepAndMul
from vllm_ascend.ops.fused_moe.moe_runtime_args import MoEMlpComputeInput
from vllm_ascend.quantization.fake_mx import (
    fake_mx_quantize,
    hadamard_transform,
    learned_hadamard_transform,
)
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import (
    dispose_tensor,
    enable_custom_op,
    get_ascend_device_type,
    get_weight_prefetch_method,
)

ASCEND_DEVICE_TYPE = get_ascend_device_type()


def _custom_gmm_swiglu_enabled(fusion, dynamic_eplb):
    return fusion and dynamic_eplb and enable_custom_op()


def cumsum_group_list(
    group_list: torch.Tensor, src_list_type: int, dst_list_type: int, active_num: int = 0, expert_num: int = 0
) -> torch.Tensor:
    if src_list_type not in [0, 1, 2]:
        raise ValueError(f"group_list_type should be in [0, 1, 2], but received {src_list_type}")

    if src_list_type == dst_list_type:
        return group_list
    if src_list_type == 1 and dst_list_type == 0:
        return group_list.cumsum(dim=0)
    if src_list_type == 0 and dst_list_type == 1:
        group_diff = torch.diff(group_list)
        new_group = torch.cat([group_list[0].unsqueeze(0), group_diff], dim=0)
        return new_group
    if src_list_type == 2 and dst_list_type == 0:
        experts = pad(group_list[:, 0], (1, 0))
        tokens = pad(group_list[:, 1].cumsum(dim=0), (1, 0))
        cumsum_group_list = torch.full(
            size=(expert_num,), fill_value=active_num, dtype=group_list.dtype, device=group_list.device
        )

        for i, (start, end) in enumerate(zip(experts[:-1], experts[1:])):
            if end > start:
                cumsum_group_list[start:end] = tokens[i]

        return cumsum_group_list
    raise NotImplementedError(
        f"Conversion from src_list_type={src_list_type} to dst_list_type={dst_list_type} is not implemented yet. "
        "This feature is under development."
    )


def _apply_expert_learned_hadamard(
    hidden_states: torch.Tensor,
    transform_weight: torch.Tensor,
    group_list: torch.Tensor,
    group_list_type: int,
) -> torch.Tensor:
    """Apply one learned block matrix to each expert's dispatched rows."""
    if transform_weight.ndim != 3:
        raise ValueError(
            f"MoE Hadamard Learning transform must have shape [experts, K, K], got {tuple(transform_weight.shape)}."
        )
    if hidden_states.shape[0] == 0:
        return hidden_states

    boundaries = cumsum_group_list(
        group_list,
        group_list_type,
        0,
        active_num=hidden_states.shape[0],
        expert_num=transform_weight.shape[0],
    )
    if boundaries.numel() != transform_weight.shape[0]:
        raise ValueError(
            "MoE Hadamard Learning group_list does not match local transforms: "
            f"{boundaries.numel()} groups vs {transform_weight.shape[0]} matrices."
        )

    outputs: list[torch.Tensor] = []
    start = 0
    for expert_idx, end_tensor in enumerate(boundaries):
        end = int(end_tensor.item())
        if end < start or end > hidden_states.shape[0]:
            raise ValueError("MoE Hadamard Learning received invalid expert token boundaries.")
        if end > start:
            outputs.append(
                learned_hadamard_transform(
                    hidden_states[start:end],
                    transform_weight[expert_idx],
                )
            )
        start = end
    if start != hidden_states.shape[0]:
        raise ValueError("MoE Hadamard Learning group_list does not cover all dispatched tokens.")
    return torch.cat(outputs, dim=0) if outputs else hidden_states


def _apply_expert_flatquant(
    hidden_states: torch.Tensor,
    fc_state: dict[str, torch.Tensor],
    group_list: torch.Tensor,
    group_list_type: int,
) -> torch.Tensor:
    """Apply per-expert FlatQuant activation transform.

    Dispatched tokens are arranged by local physical expert. group_list
    describes token boundaries per expert. We split by expert, apply the
    FlatQuant forward transform (x' = L.T @ (reshape(x) * diag) @ R) using
    that expert's L/R/diag, then concatenate back.

    This mirrors ``_apply_expert_learned_hadamard`` but uses the Kronecker
    decomposition (left/right/diag) instead of a single K×K matrix.

    Note: ``end_tensor.item()`` in the per-expert loop triggers a device-to-host
    sync.  This is acceptable for the first version (correctness validation),
    matching the pattern in ``_apply_expert_learned_hadamard``.  Batched
    device-side implementation should replace this once the math is validated.
    """
    # Lazy import to break circular dependency:
    # methods/fake_mx imports ops → moe_mlp → methods/fake_mx (transform_flatquant_activation)
    from vllm_ascend.quantization.methods.fake_mx import transform_flatquant_activation

    left_trans = fc_state["left_trans"]      # [E, L, L]
    right_trans = fc_state["right_trans"]    # [E, R, R]
    diag_scale = fc_state.get("diag_scale")  # [E, K] or None

    num_experts = left_trans.shape[0]
    if hidden_states.shape[0] == 0:
        return hidden_states

    boundaries = cumsum_group_list(
        group_list, group_list_type, 0,
        active_num=hidden_states.shape[0],
        expert_num=num_experts,
    )
    if boundaries.numel() != num_experts:
        raise ValueError(
            "MoE FlatQuant group_list does not match local transforms: "
            f"{boundaries.numel()} groups vs {num_experts} matrices."
        )

    outputs: list[torch.Tensor] = []
    start = 0
    for expert_idx, end_tensor in enumerate(boundaries):
        end = int(end_tensor.item())
        if end < start or end > hidden_states.shape[0]:
            raise ValueError("MoE FlatQuant received invalid expert token boundaries.")
        if end > start:
            seg = hidden_states[start:end]
            L = left_trans[expert_idx]       # [L_dim, L_dim]
            R = right_trans[expert_idx]      # [R_dim, R_dim]
            left_dim = L.shape[0]
            right_dim = R.shape[0]
            input_dim = seg.shape[-1]
            if left_dim * right_dim != input_dim:
                raise ValueError(
                    f"FlatQuant transform dimension mismatch: left({left_dim}) * right({right_dim}) "
                    f"!= input({input_dim}) for expert {expert_idx}."
                )
            diag = diag_scale[expert_idx] if diag_scale is not None else None
            transformed = transform_flatquant_activation(
                seg, L, R, diag, left_dim, right_dim
            )
            outputs.append(transformed)
        start = end
    if start != hidden_states.shape[0]:
        raise ValueError("MoE FlatQuant group_list does not cover all dispatched tokens.")
    return torch.cat(outputs, dim=0) if outputs else hidden_states


def _require_single_tensor_for_swiglu_quant(
    tensor_or_list: list[torch.Tensor] | torch.Tensor, *, name: str
) -> torch.Tensor:
    if isinstance(tensor_or_list, list):
        if len(tensor_or_list) != 1:
            raise ValueError(f"{name} must be a tensor or a single-element list, but got {len(tensor_or_list)}.")
        return tensor_or_list[0]
    return tensor_or_list


def quant_apply_mlp(
    hidden_states: torch.Tensor,
    w1: list[torch.Tensor] | torch.Tensor,
    w1_scale: list[torch.Tensor] | torch.Tensor,
    w2: list[torch.Tensor] | torch.Tensor,
    w2_scale: list[torch.Tensor] | torch.Tensor,
    group_list: torch.Tensor,
    group_list_type: int = 1,
    dynamic_scale: torch.Tensor = None,
    w1_scale_bias: torch.Tensor = None,
    w2_scale_bias: torch.Tensor = None,
    w1_offset: torch.Tensor | None = None,
    w2_offset: torch.Tensor | None = None,
    fusion: bool = False,
    dynamic_eplb: bool = False,
    use_mxfp_quant: bool = False,
    mxfp_quant_dtype: QuantType | None = None,
    act_quant_type: torch.dtype = torch.float8_e4m3fn,
    weight_quant_type: torch.dtype | None = None,
    scale_type: torch.dtype | None = None,
    per_token_scale_type: torch.dtype | None = None,
    use_bf16: bool = True,
    activation: str | None = None,
    swiglu_limit: float = 0.0,
    use_w4a8_per_channel_gmm_swiglu: bool = False,
) -> torch.Tensor:
    input_hidden_dtype = hidden_states.dtype
    use_gmm_swiglu_quant_fusion = use_mxfp_quant or (fusion and not dynamic_eplb)

    if use_mxfp_quant:
        ensure_mxfp8_moe_available("MXFP MoE MLP path")

        if w1_scale_bias is not None or w2_scale_bias is not None:
            raise NotImplementedError("MXFP path does not support scale_bias yet.")
        if w1_offset is not None or w2_offset is not None:
            raise NotImplementedError("MXFP path does not support antiquant offset yet.")

    if w1_offset is not None:
        unquantized_hidden_states = hidden_states
        quantized_hidden_states = None
    elif mxfp_quant_dtype == QuantType.W4A16MXFP4:
        quantized_hidden_states = None
        pertoken_scale = None
    elif dynamic_scale is None:
        unquantized_hidden_states = hidden_states
        hidden_states, pertoken_scale = DeviceOperator.npu_dynamic_quant(
            hidden_states=hidden_states,
            dynamic_scale=None,
            act_quant_type=act_quant_type,
            use_mxfp_quant=use_mxfp_quant,
        )
        dispose_tensor(unquantized_hidden_states)
        quantized_hidden_states = None
    else:
        unquantized_hidden_states = None
        pertoken_scale = (
            DeviceOperator.maybe_normalize_mxfp_scale_layout(dynamic_scale) if use_mxfp_quant else dynamic_scale
        )
        quantized_hidden_states = hidden_states

    bias1, bias2 = None, None
    _output_dtype = w2_scale[0].dtype if isinstance(w2_scale, list) else w2_scale.dtype

    weight_prefetch_method = get_weight_prefetch_method()
    if weight_prefetch_method:
        weight_prefetch_method.maybe_prefetch_moe_weight_postprocess(hidden_states)
    is_mc2 = _EXTRA_CTX.moe_comm_type == MoECommType.MC2
    if w1_scale_bias is None and w1_offset is None and is_mc2:
        if _custom_gmm_swiglu_enabled(fusion, dynamic_eplb) and not use_mxfp_quant:
            # gmm1: gate_up_proj & act_fn: swiglu
            hidden_states, swiglu_out_scale, _ = torch.ops._C_ascend.grouped_matmul_swiglu_quant_weight_nz_tensor_list(
                x=hidden_states,
                weight=w1,
                weight_scale=w1_scale,
                x_scale=pertoken_scale,
                group_list=cumsum_group_list(group_list, group_list_type, 0),
                swiglu_limit=swiglu_limit,
            )
        elif use_gmm_swiglu_quant_fusion:
            # gmm1: gate_up_proj & act_fn: swiglu
            hidden_states, swiglu_out_scale, _ = DeviceOperator.npu_grouped_matmul_swiglu_quant(
                x=hidden_states,
                weight=_require_single_tensor_for_swiglu_quant(w1, name="w1"),
                group_list=cumsum_group_list(group_list, group_list_type, 0),
                weight_scale=_require_single_tensor_for_swiglu_quant(w1_scale, name="w1_scale"),
                x_scale=pertoken_scale,
                bias=None,
                use_mxfp_quant=use_mxfp_quant,
                act_quant_type=act_quant_type,
                weight_quant_type=weight_quant_type,
                swiglu_limit=swiglu_limit,
                mxfp_quant_dtype=mxfp_quant_dtype,
            )
            if quantized_hidden_states is not None:
                dispose_tensor(quantized_hidden_states)
        else:
            if w1_scale[0].dtype != torch.float32:
                w1_scale[0] = w1_scale[0].to(torch.float32)
            # gmm1: gate_up_proj
            hidden_states = torch_npu.npu_grouped_matmul(
                x=[hidden_states],
                weight=w1,
                split_item=3,
                group_list_type=group_list_type,
                group_type=0,
                group_list=group_list,
                output_dtype=torch.int32,
            )[0]
            if quantized_hidden_states is not None:
                dispose_tensor(quantized_hidden_states)
            # act_fn: swiglu
            hidden_states, swiglu_out_scale = torch.ops._C_ascend.npu_dequant_swiglu_quant(
                x=hidden_states,
                weight_scale=w1_scale[0],
                activation_scale=pertoken_scale,
                bias=None,
                quant_scale=None,
                quant_offset=None,
                group_index=cumsum_group_list(group_list, group_list_type, 1),
                activate_left=True,
                quant_mode=1,
            )
        before_gmm2_evt = torch.npu.current_stream().record_event()
        # gmm2: down_proj
        hidden_states = DeviceOperator.npu_grouped_matmul_gmm2(
            hidden_states=hidden_states,
            weight=w2,
            weight_scale=w2_scale,
            per_token_scale=swiglu_out_scale,
            group_list=group_list,
            group_list_type=group_list_type,
            input_dtype=input_hidden_dtype,
            act_quant_type=act_quant_type,
            weight_quant_type=weight_quant_type,
            scale_type=scale_type,
            per_token_scale_type=per_token_scale_type,
            use_bf16=use_bf16,
            use_mxfp_quant=use_mxfp_quant,
            bias=None,
            fallback_output_dtype=w2_scale[0].dtype if isinstance(w2_scale, list) else w2_scale.dtype,
            mxfp_quant_dtype=mxfp_quant_dtype,
        )
    elif w1_offset is not None:
        # gmm1: gate_up_proj
        hidden_states = torch_npu.npu_grouped_matmul(
            x=[unquantized_hidden_states],
            weight=[w1],
            antiquant_scale=[w1_scale],
            antiquant_offset=[w1_offset],
            split_item=2,
            group_list_type=group_list_type,
            group_type=0,
            group_list=group_list,
            output_dtype=_output_dtype,
        )[0]
        dispose_tensor(unquantized_hidden_states)
        # act_fn: swiglu
        if activation == MoEActivation.SWIGLUSTEP:
            hidden_states = AscendSwigluStepAndMul.swiglustep_forward(hidden_states, limit=swiglu_limit or 7.0)
        else:
            hidden_states = torch_npu.npu_swiglu(hidden_states)
        before_gmm2_evt = torch.npu.current_stream().record_event()
        # gmm2: down_proj
        hidden_states = torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[w2],
            antiquant_scale=[w2_scale],
            antiquant_offset=[w2_offset],
            split_item=2,
            group_list_type=group_list_type,
            group_type=0,
            group_list=group_list,
            output_dtype=_output_dtype,
        )[0]
    else:
        if w1_scale_bias is not None:
            if group_list_type == 0:
                group_list = torch.cat([group_list[:1], torch.diff(group_list, dim=0)])
                group_list_type = 1
            bias1 = w1_scale_bias
            bias2 = w2_scale_bias
            # TODO w4a8 scene: dynamic acquisition of dtype in the future
            _output_dtype = torch.bfloat16

        if use_w4a8_per_channel_gmm_swiglu and enable_custom_op() and activation != MoEActivation.SWIGLUSTEP:
            hidden_states, swiglu_out_scale = torch.ops._C_ascend.grouped_matmul_swiglu_quant_v2(
                x=hidden_states,
                weight=w1,
                weight_scale=w1_scale if isinstance(w1_scale, list) else [w1_scale],
                x_scale=pertoken_scale,
                group_list=group_list,
                weight_assist_matrix=bias1,
                dequant_mode=0,
                group_list_type=group_list_type,
                swiglu_limit=swiglu_limit,
            )
        elif _custom_gmm_swiglu_enabled(fusion, dynamic_eplb) and not use_mxfp_quant:
            # gmm1: gate_up_proj & act_fn: swiglu
            hidden_states, swiglu_out_scale, _ = torch.ops._C_ascend.grouped_matmul_swiglu_quant_weight_nz_tensor_list(
                x=hidden_states,
                weight=w1,
                weight_scale=w1_scale,
                x_scale=pertoken_scale,
                group_list=cumsum_group_list(group_list, group_list_type, 0),
                bias=bias1,
                swiglu_limit=swiglu_limit,
            )
        elif use_gmm_swiglu_quant_fusion and activation != MoEActivation.SWIGLUSTEP:
            hidden_states, swiglu_out_scale, _ = DeviceOperator.npu_grouped_matmul_swiglu_quant(
                x=hidden_states,
                weight=_require_single_tensor_for_swiglu_quant(w1, name="w1"),
                group_list=cumsum_group_list(group_list, group_list_type, 0),
                weight_scale=_require_single_tensor_for_swiglu_quant(w1_scale, name="w1_scale"),
                x_scale=pertoken_scale,
                bias=bias1,
                use_mxfp_quant=use_mxfp_quant,
                act_quant_type=act_quant_type,
                weight_quant_type=weight_quant_type,
                swiglu_limit=swiglu_limit,
                mxfp_quant_dtype=mxfp_quant_dtype,
            )
            if quantized_hidden_states is not None:
                dispose_tensor(quantized_hidden_states)
        else:
            w1_scale[0] = w1_scale[0].to(w2_scale[0].dtype)
            # gmm1: gate_up_proj
            hidden_states = torch_npu.npu_grouped_matmul(
                x=[hidden_states],
                weight=w1,
                scale=w1_scale,
                bias=bias1,
                per_token_scale=[pertoken_scale],
                split_item=2,
                group_list_type=group_list_type,
                group_type=0,
                group_list=group_list,
                output_dtype=_output_dtype,
            )[0]
            if quantized_hidden_states is not None:
                dispose_tensor(quantized_hidden_states)
            # act_fn: swiglu
            if activation == MoEActivation.SWIGLUSTEP:
                hidden_states = AscendSwigluStepAndMul.swiglustep_forward(hidden_states, limit=swiglu_limit or 7.0)
                hidden_states, swiglu_out_scale = torch_npu.npu_dynamic_quant(hidden_states)
            elif HAS_TRITON:
                from vllm_ascend.ops.triton.activation.swiglu_quant import swiglu_quant

                hidden_states, swiglu_out_scale = swiglu_quant(
                    hidden_states, group_list=group_list, group_list_type=group_list_type
                )
            else:
                hidden_states = torch_npu.npu_swiglu(hidden_states)
                hidden_states, swiglu_out_scale = torch_npu.npu_dynamic_quant(hidden_states)
        before_gmm2_evt = torch.npu.current_stream().record_event()
        # gmm2: down_proj
        hidden_states = DeviceOperator.npu_grouped_matmul_gmm2(
            hidden_states=hidden_states,
            weight=w2,
            weight_scale=w2_scale,
            per_token_scale=swiglu_out_scale,
            group_list=group_list,
            group_list_type=group_list_type,
            input_dtype=input_hidden_dtype,
            act_quant_type=act_quant_type,
            weight_quant_type=weight_quant_type,
            scale_type=scale_type,
            per_token_scale_type=per_token_scale_type,
            use_bf16=use_bf16,
            use_mxfp_quant=use_mxfp_quant,
            bias=bias2,
            fallback_output_dtype=_output_dtype,
            mxfp_quant_dtype=mxfp_quant_dtype,
        )
    return hidden_states, before_gmm2_evt


def unquant_apply_mlp(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    group_list: torch.Tensor,
    w1_bias: torch.Tensor = None,
    w2_bias: torch.Tensor = None,
    activation: str | None = None,
    group_list_type: int = 1,
    topk_scales: torch.Tensor | None = None,
    need_trans: bool = True,
    swiglu_limit: float = 0.0,
    lora_context=None,
    expanded_row_idx: torch.Tensor | None = None,
    topk_ids: torch.Tensor | None = None,
    fake_mx_format: str | None = None,
    fake_mx_group_size: int = 32,
    fake_mx_algorithm: str = "rtn",
    fake_mx_rht_signs: torch.Tensor | None = None,
    fake_mx_rht_group_size: int = 32,
    fake_mx_w13_transform: torch.Tensor | None = None,
    fake_mx_w2_transform: torch.Tensor | None = None,
    fake_mx_flatquant_fc1_state: dict[str, torch.Tensor] | None = None,
    fake_mx_flatquant_fc2_state: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    if need_trans:
        w1 = w1.transpose(1, 2)
        w2 = w2.transpose(1, 2)

    if fake_mx_algorithm == "hadamard_learning":
        if fake_mx_w13_transform is None:
            raise ValueError("fake-MX Hadamard Learning requires per-expert FC1 transforms.")
        hidden_states = _apply_expert_learned_hadamard(
            hidden_states,
            fake_mx_w13_transform,
            group_list,
            group_list_type,
        )
        if fake_mx_format is None:
            raise ValueError("fake-MX Hadamard Learning requires a fake MX format.")
        hidden_states = fake_mx_quantize(
            hidden_states,
            fake_mx_format,
            fake_mx_group_size,
        )
    elif fake_mx_algorithm == "flatquant":
        if fake_mx_flatquant_fc1_state is None:
            raise ValueError("fake-MX FlatQuant requires per-expert FC1 state.")
        if fake_mx_format is None:
            raise ValueError("fake-MX FlatQuant requires a fake MX format.")
        hidden_states = _apply_expert_flatquant(
            hidden_states,
            fake_mx_flatquant_fc1_state,
            group_list,
            group_list_type,
        )
        hidden_states = fake_mx_quantize(
            hidden_states,
            fake_mx_format,
            fake_mx_group_size,
        )

    gate_up_out = torch_npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w1],
        bias=[w1_bias.to(dtype=torch.float32)] if w1_bias is not None else None,
        split_item=2,
        group_list_type=group_list_type,
        group_type=0,
        group_list=group_list,
    )[0]

    # MoE LoRA: only attempt injection when an adapter wraps this layer and the
    # comm method provided AllGather routing metadata (expanded_row_idx). Lazy
    # import keeps the core MLP free of any LoRA dependency on the common path.
    lora_routing = None
    if lora_context is not None:  # LoRA applied
        if expanded_row_idx is None or topk_ids is None:
            raise AssertionError(
                "MoE LoRA requires expanded_row_idx and topk_ids metadata, "
                "which are only available in AllGather communication mode. "
                "Please ensure you are running in a supported configuration."
            )

        from vllm_ascend.lora.fused_moe import moe_lora_apply_w2, moe_lora_apply_w13

        # LoRA w13 delta: applied to gate_up_out before activation, with the MLP
        # input as the lora_a input (mirrors the base gate_up GMM above).
        lora_routing = moe_lora_apply_w13(
            lora_context,
            gate_up_out=gate_up_out,
            hidden_states=hidden_states,
            expanded_row_idx=expanded_row_idx,
            topk_ids=topk_ids,
        )

    if activation == MoEActivation.SWIGLUOAI:
        num_experts, _, hidden_size = w1.shape
        gate_up_out = AscendSwigluOAIAndMul.swiglu_oai_forward(gate_up_out.view(-1, hidden_size))
    elif activation == MoEActivation.SWIGLUSTEP:
        gate_up_out = AscendSwigluStepAndMul.swiglustep_forward(gate_up_out, limit=swiglu_limit or 7.0)
    elif activation == MoEActivation.GELU:
        gate, up = gate_up_out.chunk(2, dim=-1)
        gate_up_out = torch.nn.functional.gelu(gate) * up
    elif activation == MoEActivation.GELU_TANH:
        gate, up = gate_up_out.chunk(2, dim=-1)
        gate_up_out = torch.nn.functional.gelu(gate, approximate="tanh") * up
    else:
        if swiglu_limit > 0:
            gate, up = gate_up_out.chunk(2, dim=-1)
            gate.clamp_(max=swiglu_limit)
            up.clamp_(min=-swiglu_limit, max=swiglu_limit)
        gate_up_out = torch_npu.npu_swiglu(gate_up_out)

    if fake_mx_algorithm == "rht":
        gate_up_out = hadamard_transform(gate_up_out, fake_mx_rht_group_size)
    elif fake_mx_algorithm == "hadamard_learning":
        if fake_mx_w2_transform is None:
            raise ValueError("fake-MX Hadamard Learning requires per-expert FC2 transforms.")
        gate_up_out = _apply_expert_learned_hadamard(
            gate_up_out,
            fake_mx_w2_transform,
            group_list,
            group_list_type,
        )
    elif fake_mx_algorithm == "flatquant":
        if fake_mx_flatquant_fc2_state is None:
            raise ValueError("fake-MX FlatQuant requires per-expert FC2 state.")
        gate_up_out = _apply_expert_flatquant(
            gate_up_out,
            fake_mx_flatquant_fc2_state,
            group_list,
            group_list_type,
        )

    if fake_mx_format is not None:
        gate_up_out = fake_mx_quantize(
            gate_up_out,
            fake_mx_format,
            fake_mx_group_size,
        )

    if topk_scales is not None:
        gate_up_out *= topk_scales

    hidden_states = torch_npu.npu_grouped_matmul(
        x=[gate_up_out],
        weight=[w2],
        bias=[w2_bias.to(dtype=torch.float32)] if w2_bias is not None else None,
        split_item=2,
        group_list_type=group_list_type,
        group_type=0,
        group_list=group_list,
    )[0]

    # LoRA w2 delta: applied to the down-proj output, with the activation output
    # as the lora_a input. Reuses the per-row routing computed for w13.
    if lora_routing is not None:
        moe_lora_apply_w2(
            lora_context,
            down_out=hidden_states,
            silu_out=gate_up_out,
            lora_routing=lora_routing,
        )
    return hidden_states, None


def unified_apply_mlp(*, mlp_compute_input: MoEMlpComputeInput) -> torch.Tensor:
    """
    Unified MoE MLP entry.
    Quant path is dispatched by DeviceOperator with explicit typed kernel flags.
    """
    hidden_states = mlp_compute_input.hidden_states
    group_list = mlp_compute_input.group_list
    group_list_type = mlp_compute_input.group_list_type
    dynamic_scale = mlp_compute_input.dynamic_scale
    topk_scales = mlp_compute_input.topk_scales
    w1 = mlp_compute_input.weights.w1
    w2 = mlp_compute_input.weights.w2
    w1_bias = mlp_compute_input.weights.w1_bias
    w2_bias = mlp_compute_input.weights.w2_bias
    w1_scale = mlp_compute_input.weights.w1_scale
    w2_scale = mlp_compute_input.weights.w2_scale
    w1_scale_bias = mlp_compute_input.weights.w1_scale_bias
    w2_scale_bias = mlp_compute_input.weights.w2_scale_bias
    w1_offset = mlp_compute_input.weights.w1_offset
    w2_offset = mlp_compute_input.weights.w2_offset
    activation = mlp_compute_input.activation
    need_trans = mlp_compute_input.need_trans
    dynamic_eplb = mlp_compute_input.dynamic_eplb
    fusion = mlp_compute_input.fusion
    swiglu_limit = mlp_compute_input.swiglu_limit

    if not mlp_compute_input.quant.is_quant:
        return unquant_apply_mlp(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            w1_bias=w1_bias,
            w2_bias=w2_bias,
            activation=activation,
            group_list=group_list,
            group_list_type=group_list_type,
            topk_scales=topk_scales,
            need_trans=need_trans,
            swiglu_limit=swiglu_limit,
            lora_context=mlp_compute_input.lora_context,
            expanded_row_idx=mlp_compute_input.expanded_row_idx,
            topk_ids=mlp_compute_input.topk_ids,
            fake_mx_format=mlp_compute_input.quant.fake_mx_format,
            fake_mx_group_size=mlp_compute_input.quant.fake_mx_group_size,
            fake_mx_algorithm=mlp_compute_input.quant.fake_mx_algorithm,
            fake_mx_rht_signs=mlp_compute_input.quant.fake_mx_rht_signs,
            fake_mx_rht_group_size=mlp_compute_input.quant.fake_mx_rht_group_size,
            fake_mx_w13_transform=mlp_compute_input.quant.fake_mx_w13_transform,
            fake_mx_w2_transform=mlp_compute_input.quant.fake_mx_w2_transform,
            fake_mx_flatquant_fc1_state=mlp_compute_input.quant.fake_mx_flatquant_fc1_state,
            fake_mx_flatquant_fc2_state=mlp_compute_input.quant.fake_mx_flatquant_fc2_state,
        )

    assert w1_scale is not None and w2_scale is not None
    act_quant_type = torch.int8 if mlp_compute_input.quant.is_int_quant else torch.float8_e4m3fn
    weight_quant_type = torch.float8_e4m3fn
    scale_type = None
    per_token_scale_type = None
    use_bf16 = hidden_states.dtype == torch.bfloat16
    use_mxfp_quant = mlp_compute_input.quant.is_mxfp
    mxfp_quant_dtype = mlp_compute_input.quant.quant_type

    if use_mxfp_quant:
        mxfp = mlp_compute_input.quant.mxfp
        assert mxfp is not None, "mlp_compute_input.quant.mxfp is required when quant_type is MXFP8."
        act_quant_type = mxfp.act_quant_type or act_quant_type
        if mxfp_quant_dtype == QuantType.W4A16MXFP4:
            act_quant_type = mxfp.act_quant_type
        weight_quant_type = mxfp.weight_quant_type or weight_quant_type
        if mxfp_quant_dtype in [QuantType.W4A8MXFP, QuantType.W4A16MXFP4]:
            weight_quant_type = mxfp.weight_quant_type
        scale_type = mxfp.scale_dtype
        per_token_scale_type = mxfp.per_token_scale_dtype
        use_bf16 = mxfp.use_bf16

    return quant_apply_mlp(
        hidden_states=hidden_states,
        w1=w1,
        w1_scale=w1_scale,
        w2=w2,
        w2_scale=w2_scale,
        group_list=group_list,
        dynamic_scale=dynamic_scale,
        group_list_type=group_list_type,
        w1_scale_bias=w1_scale_bias,
        w2_scale_bias=w2_scale_bias,
        w1_offset=w1_offset,
        w2_offset=w2_offset,
        fusion=fusion,
        dynamic_eplb=dynamic_eplb,
        use_mxfp_quant=use_mxfp_quant,
        mxfp_quant_dtype=mxfp_quant_dtype,
        act_quant_type=act_quant_type,
        weight_quant_type=weight_quant_type,
        scale_type=scale_type,
        per_token_scale_type=per_token_scale_type,
        use_bf16=use_bf16,
        activation=activation,
        swiglu_limit=swiglu_limit,
        use_w4a8_per_channel_gmm_swiglu=mlp_compute_input.quant.use_w4a8_per_channel_gmm_swiglu,
    )
