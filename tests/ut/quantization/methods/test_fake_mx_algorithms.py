from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn.functional as F

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.quantization.fake_mx import (
    hadamard_transform,
    learned_hadamard_transform,
    randomized_hadamard_transform,
)
from vllm_ascend.quantization.methods.fake_mx import (
    AscendW4A4MXFP4AutoRoundFakeLinearMethod,
    AscendW4A4MXFP4FakeFlatQuantLinearMethod,
    AscendW4A4MXFP4FakeFusedMoEMethod,
    AscendW4A4MXFP4FakeLinearMethod,
    AscendW4A4MXFP4HadamardLearningFakeFusedMoEMethod,
    AscendW4A4MXFP4HadamardLearningFakeLinearMethod,
    AscendW4A4MXFP4OmniQuantFakeFusedMoEMethod,
    AscendW4A4MXFP4RHTFakeFusedMoEMethod,
    AscendW4A4MXFP4RHTFakeLinearMethod,
    _inverse_fp32,
    transform_lht_weight,
)


def _mock_vllm_config(quant_description):
    return Mock(quant_config=Mock(quant_description=quant_description))


def test_fake_mx_runtime_targets_only_accept_attn_cache():
    with patch(
        "vllm_ascend.quantization.methods.fake_mx.get_current_vllm_config",
        return_value=_mock_vllm_config({"group_size": 32}),
    ):
        method = AscendW4A4MXFP4FakeLinearMethod()
    assert method.quant_targets == frozenset()

    with patch(
        "vllm_ascend.quantization.methods.fake_mx.get_current_vllm_config",
        return_value=_mock_vllm_config(
            {"group_size": 32, "fake_mx_quant_targets": ["attn-cache"]}
        ),
    ):
        method = AscendW4A4MXFP4FakeLinearMethod()
    assert method.quant_targets == frozenset({"attn-cache"})

    with (
        patch(
            "vllm_ascend.quantization.methods.fake_mx.get_current_vllm_config",
            return_value=_mock_vllm_config(
                {"group_size": 32, "fake_mx_quant_targets": ["attn-linear"]}
            ),
        ),
        pytest.raises(ValueError, match="Unsupported fake-MX quant targets"),
    ):
        AscendW4A4MXFP4FakeLinearMethod()


def test_prequantized_algorithm_rejects_missing_weight_state():
    with (
        patch(
            "vllm_ascend.quantization.methods.fake_mx.get_current_vllm_config",
            return_value=_mock_vllm_config({"group_size": 32}),
        ),
        pytest.raises(ValueError, match="prequantized_qdq"),
    ):
        AscendW4A4MXFP4AutoRoundFakeLinearMethod()


def test_autoround_preserves_prequantized_weight_during_post_load():
    config = {
        "group_size": 4,
        "fake_mx_weight_state": "prequantized_qdq",
    }
    with patch(
        "vllm_ascend.quantization.methods.fake_mx.get_current_vllm_config",
        return_value=_mock_vllm_config(config),
    ):
        method = AscendW4A4MXFP4AutoRoundFakeLinearMethod()

    layer = torch.nn.Linear(4, 1, bias=False)
    layer.weight.data.copy_(torch.tensor([[6.0, 5.5, 3.0, 0.25]]))
    expected = layer.weight.detach().clone()

    method.process_weights_after_loading(layer)

    torch.testing.assert_close(layer.weight, expected)


def test_rtn_writes_mxfp4_error_into_weight_once():
    with patch(
        "vllm_ascend.quantization.methods.fake_mx.get_current_vllm_config",
        return_value=_mock_vllm_config({"group_size": 4}),
    ):
        method = AscendW4A4MXFP4FakeLinearMethod()

    layer = torch.nn.Linear(4, 1, bias=False)
    layer.weight.data.copy_(torch.tensor([[6.0, 5.0, 3.0, 0.25]]))

    method.process_weights_after_loading(layer)
    first_result = layer.weight.detach().clone()
    method.process_weights_after_loading(layer)

    torch.testing.assert_close(first_result, torch.tensor([[6.0, 6.0, 3.0, 0.5]]))
    torch.testing.assert_close(layer.weight, first_result)


def test_rht_online_activation_and_offline_weight_rotation_are_fp_equivalent():
    weight = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [-2.0, 1.0, 0.5, 3.0],
        ]
    )
    activation = torch.tensor([[0.5, -1.0, 2.0, 3.0]])
    signs = torch.tensor([1, -1, 1, -1], dtype=torch.int8)

    rotated_weight = randomized_hadamard_transform(weight, signs, group_size=4)
    rotated_activation = randomized_hadamard_transform(activation, signs, group_size=4)

    torch.testing.assert_close(
        F.linear(rotated_activation, rotated_weight),
        F.linear(activation, weight),
    )


def test_rht_rejects_unrotated_checkpoint_marker():
    config = {
        "group_size": 32,
        "rht_group_size": 32,
        "fake_mx_weight_state": "prequantized_qdq",
    }
    with (
        patch(
            "vllm_ascend.quantization.methods.fake_mx.get_current_vllm_config",
            return_value=_mock_vllm_config(config),
        ),
        pytest.raises(ValueError, match="rht_rotated_fp"),
    ):
        AscendW4A4MXFP4RHTFakeLinearMethod()


def test_hadamard_learning_online_activation_and_offline_weight_are_fp_equivalent():
    weight = torch.tensor([[1.0, 2.0], [-2.0, 1.0]])
    activation = torch.tensor([[0.5, -1.0]])
    transform = torch.tensor([[2.0, 0.5], [0.25, 1.5]])
    transformed_weight = weight.reshape(-1, 2).matmul(torch.linalg.inv(transform).T)
    transformed_activation = learned_hadamard_transform(activation, transform)

    torch.testing.assert_close(
        F.linear(transformed_activation, transformed_weight),
        F.linear(activation, weight),
    )


def test_hadamard_learning_loads_transform_and_applies_before_fake_mx():
    config = {
        "group_size": 2,
        "hadamard_learning_matrix_size": 2,
        "fake_mx_weight_state": "hadamard_learning_transformed_fp",
    }
    with patch(
        "vllm_ascend.quantization.methods.fake_mx.get_current_vllm_config",
        return_value=_mock_vllm_config(config),
    ):
        method = AscendW4A4MXFP4HadamardLearningFakeLinearMethod()

    layer = torch.nn.Linear(2, 2, bias=False)
    layer.weight.data.copy_(torch.eye(2))
    layer.transform_weight = torch.nn.Parameter(torch.tensor([[1.0, 1.0], [1.0, -1.0]]))
    method.process_weights_after_loading(layer)

    actual = method.apply(layer, torch.tensor([[1.0, 1.0]]))
    expected = F.linear(torch.tensor([[2.0, 0.0]]), layer.weight)
    torch.testing.assert_close(actual, expected)


def test_flatquant_applies_transform_clip_and_mxfp4_qdq():
    config = {
        "group_size": 4,
        "fake_mx_weight_state": "flatquant_transformed_fp",
        "max_supported_tp": 4,
    }
    with (
        patch(
            "vllm_ascend.quantization.methods.fake_mx.get_current_vllm_config",
            return_value=_mock_vllm_config(config),
        ),
        patch(
            "vllm_ascend.quantization.methods.fake_mx.get_tensor_model_parallel_world_size",
            return_value=1,
        ),
    ):
        method = AscendW4A4MXFP4FakeFlatQuantLinearMethod()

    layer = torch.nn.Linear(4, 4, bias=False)
    layer.weight.data.copy_(torch.eye(4))
    layer.left_trans = torch.nn.Parameter(torch.eye(2), requires_grad=False)
    layer.right_trans = torch.nn.Parameter(torch.eye(2), requires_grad=False)
    layer.clip_ratio = torch.nn.Parameter(torch.tensor([0.5]), requires_grad=False)
    method.process_weights_after_loading(layer)
    activation = torch.tensor([[6.0, 5.0, 3.0, 2.0]])

    actual = method.apply(layer, activation)

    expected_activation = torch.tensor([[3.0, 3.0, 3.0, 2.0]])
    torch.testing.assert_close(actual, F.linear(expected_activation, layer.weight))


def test_flatquant_rejects_plain_checkpoint_marker():
    config = {
        "group_size": 4,
        "fake_mx_weight_state": "rht_rotated_fp",
    }
    with (
        patch(
            "vllm_ascend.quantization.methods.fake_mx.get_current_vllm_config",
            return_value=_mock_vllm_config(config),
        ),
        pytest.raises(ValueError, match="flatquant_transformed_fp"),
    ):
        AscendW4A4MXFP4FakeFlatQuantLinearMethod()


def test_omniquant_moe_preserves_prequantized_expert_weights():
    config = {
        "group_size": 4,
        "fake_mx_weight_state": "prequantized_qdq",
    }
    ascend_config = Mock(eplb_config=Mock(dynamic_eplb=False))
    with (
        patch(
            "vllm_ascend.quantization.methods.fake_mx.get_current_vllm_config",
            return_value=_mock_vllm_config(config),
        ),
        patch(
            "vllm_ascend.quantization.methods.fake_mx.get_ascend_config",
            return_value=ascend_config,
        ),
    ):
        method = AscendW4A4MXFP4OmniQuantFakeFusedMoEMethod()

    layer = torch.nn.Module()
    layer.w13_weight = torch.nn.Parameter(torch.randn(2, 8, 4), requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(torch.randn(2, 4, 4), requires_grad=False)
    expected_w13 = layer.w13_weight.detach().clone()
    expected_w2 = layer.w2_weight.detach().clone()

    with patch("vllm_ascend.quantization.methods.fake_mx.maybe_trans_nz", side_effect=lambda weight: weight):
        method.process_weights_after_loading(layer)

    torch.testing.assert_close(layer.w13_weight, expected_w13.transpose(1, 2).contiguous())
    torch.testing.assert_close(layer.w2_weight, expected_w2.transpose(1, 2).contiguous())


def test_rtn_fake_mx_moe_qdq_precedes_ascend_gmm_layout_transform():
    ascend_config = Mock(eplb_config=Mock(dynamic_eplb=False))
    with (
        patch(
            "vllm_ascend.quantization.methods.fake_mx.get_current_vllm_config",
            return_value=_mock_vllm_config({"group_size": 4}),
        ),
        patch(
            "vllm_ascend.quantization.methods.fake_mx.get_ascend_config",
            return_value=ascend_config,
        ),
    ):
        method = AscendW4A4MXFP4FakeFusedMoEMethod()

    layer = torch.nn.Module()
    layer.w13_weight = torch.nn.Parameter(torch.randn(2, 8, 4), requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(torch.randn(2, 4, 4), requires_grad=False)
    original_w13 = layer.w13_weight.detach().clone()
    original_w2 = layer.w2_weight.detach().clone()

    with (
        patch(
            "vllm_ascend.quantization.methods.fake_mx.fake_mx_quantize",
            side_effect=lambda weight, _format, _group_size: weight + 1,
        ) as fake_quantize,
        patch("vllm_ascend.quantization.methods.fake_mx.maybe_trans_nz", side_effect=lambda weight: weight),
    ):
        method.process_weights_after_loading(layer)

    assert fake_quantize.call_count == 2
    torch.testing.assert_close(layer.w13_weight, (original_w13 + 1).transpose(1, 2).contiguous())
    torch.testing.assert_close(layer.w2_weight, (original_w2 + 1).transpose(1, 2).contiguous())
    assert layer.w13_weight.shape == (2, 4, 8)
    assert layer.w2_weight.shape == (2, 4, 4)


def test_fake_mx_moe_rejects_monolithic_fused_mc2():
    ascend_config = Mock(eplb_config=Mock(dynamic_eplb=False))
    with (
        patch(
            "vllm_ascend.quantization.methods.fake_mx.get_current_vllm_config",
            return_value=_mock_vllm_config({"group_size": 4}),
        ),
        patch(
            "vllm_ascend.quantization.methods.fake_mx.get_ascend_config",
            return_value=ascend_config,
        ),
    ):
        method = AscendW4A4MXFP4FakeFusedMoEMethod()

    with (
        patch(
            "vllm_ascend.quantization.methods.fake_mx._EXTRA_CTX",
            Mock(moe_comm_type=MoECommType.FUSED_MC2),
        ),
        pytest.raises(NotImplementedError, match="split dispatch.*GMM1.*GMM2"),
    ):
        method._validate_execution_path()


# ---- F1: RHT MoE fix tests ----


def test_rht_moe_no_longer_requires_weight_state():
    """MoE RHT should not require fake_mx_weight_state marker."""
    ascend_config = Mock(eplb_config=Mock(dynamic_eplb=False))
    config = {"group_size": 4, "rht_group_size": 4}
    with (
        patch(
            "vllm_ascend.quantization.methods.fake_mx.get_current_vllm_config",
            return_value=_mock_vllm_config(config),
        ),
        patch(
            "vllm_ascend.quantization.methods.fake_mx.get_ascend_config",
            return_value=ascend_config,
        ),
    ):
        # Should not raise
        method = AscendW4A4MXFP4RHTFakeFusedMoEMethod()
    assert method.algorithm == "rht"
    assert method.required_weight_state is None


def test_rht_moe_rotates_weights_at_load_time():
    """MoE RHT should rotate w13/w2 weights via hadamard_transform at load time."""
    ascend_config = Mock(eplb_config=Mock(dynamic_eplb=False))
    config = {"group_size": 4, "rht_group_size": 4}
    with (
        patch(
            "vllm_ascend.quantization.methods.fake_mx.get_current_vllm_config",
            return_value=_mock_vllm_config(config),
        ),
        patch(
            "vllm_ascend.quantization.methods.fake_mx.get_ascend_config",
            return_value=ascend_config,
        ),
    ):
        method = AscendW4A4MXFP4RHTFakeFusedMoEMethod()

    layer = torch.nn.Module()
    layer.w13_weight = torch.nn.Parameter(torch.randn(2, 8, 4), requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(torch.randn(2, 4, 4), requires_grad=False)
    original_w13 = layer.w13_weight.detach().clone()
    original_w2 = layer.w2_weight.detach().clone()

    with (
        patch(
            "vllm_ascend.quantization.methods.fake_mx.fake_mx_quantize",
            side_effect=lambda weight, _format, _group_size: weight,
        ),
        patch("vllm_ascend.quantization.methods.fake_mx.maybe_trans_nz", side_effect=lambda weight: weight),
    ):
        method.process_weights_after_loading(layer)

    expected_w13 = hadamard_transform(original_w13, 4).transpose(1, 2).contiguous()
    expected_w2 = hadamard_transform(original_w2, 4).transpose(1, 2).contiguous()
    torch.testing.assert_close(layer.w13_weight, expected_w13)
    torch.testing.assert_close(layer.w2_weight, expected_w2)
    assert getattr(layer, "_fake_mx_rht_weight_rotated", False) is True


def test_rht_moe_rotation_is_idempotent():
    """process_weights_after_loading should not re-rotate if already rotated."""
    ascend_config = Mock(eplb_config=Mock(dynamic_eplb=False))
    config = {"group_size": 4, "rht_group_size": 4}
    with (
        patch(
            "vllm_ascend.quantization.methods.fake_mx.get_current_vllm_config",
            return_value=_mock_vllm_config(config),
        ),
        patch(
            "vllm_ascend.quantization.methods.fake_mx.get_ascend_config",
            return_value=ascend_config,
        ),
    ):
        method = AscendW4A4MXFP4RHTFakeFusedMoEMethod()

    layer = torch.nn.Module()
    layer.w13_weight = torch.nn.Parameter(torch.randn(2, 8, 4), requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(torch.randn(2, 4, 4), requires_grad=False)

    with (
        patch(
            "vllm_ascend.quantization.methods.fake_mx.fake_mx_quantize",
            side_effect=lambda weight, _format, _group_size: weight,
        ),
        patch("vllm_ascend.quantization.methods.fake_mx.maybe_trans_nz", side_effect=lambda weight: weight),
    ):
        method.process_weights_after_loading(layer)
    first_pass_w13 = layer.w13_weight.detach().clone()
    # Second call should be a no-op (idempotent guard)
    method.process_weights_after_loading(layer)

    torch.testing.assert_close(layer.w13_weight, first_pass_w13)


# ---- F2: LHT MoE fix tests ----


def test_lht_weight_formula_uses_inv_t_transpose():
    """transform_lht_weight must compute W @ inv(T).T, not W @ T."""
    weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    transform = torch.tensor([[2.0, 0.5], [0.25, 1.5]])  # non-orthogonal

    result = transform_lht_weight(weight, transform, 2)

    # Expected: reshape(W, -1, K) @ inv(T).T
    inv_t_t = _inverse_fp32(transform, transpose=True)
    expected = weight.to(torch.float32).reshape(-1, 2) @ inv_t_t
    expected = expected.reshape(weight.shape)
    torch.testing.assert_close(result, expected)

    # Verify it differs from old formula W @ T (since T is non-orthogonal)
    old = weight.to(torch.float32).reshape(-1, 2) @ transform.to(torch.float32)
    old = old.reshape(weight.shape)
    assert not torch.allclose(result, old), "Formula should differ from W@T for non-orthogonal T"


def test_lht_moe_no_longer_requires_weight_state():
    """MoE LHT should not require fake_mx_weight_state marker."""
    ascend_config = Mock(eplb_config=Mock(dynamic_eplb=False))
    config = {"group_size": 2, "hadamard_learning_matrix_size": 2}
    with (
        patch(
            "vllm_ascend.quantization.methods.fake_mx.get_current_vllm_config",
            return_value=_mock_vllm_config(config),
        ),
        patch(
            "vllm_ascend.quantization.methods.fake_mx.get_ascend_config",
            return_value=ascend_config,
        ),
    ):
        # Should not raise
        method = AscendW4A4MXFP4HadamardLearningFakeFusedMoEMethod()
    assert method.algorithm == "hadamard_learning"
    assert method.required_weight_state is None
    assert method.params_path is None  # no lht_params_path in config


def test_lht_moe_get_weight_initializes_transform_to_identity():
    """get_weight should initialize transform weights to identity, not empty."""
    ascend_config = Mock(eplb_config=Mock(dynamic_eplb=False))
    config = {"group_size": 2, "hadamard_learning_matrix_size": 2}
    with (
        patch(
            "vllm_ascend.quantization.methods.fake_mx.get_current_vllm_config",
            return_value=_mock_vllm_config(config),
        ),
        patch(
            "vllm_ascend.quantization.methods.fake_mx.get_ascend_config",
            return_value=ascend_config,
        ),
    ):
        method = AscendW4A4MXFP4HadamardLearningFakeFusedMoEMethod()

    weights = method.get_weight(
        num_experts=3, intermediate_size_per_partition=2,
        hidden_sizes=4, params_dtype=torch.float32,
    )
    assert "w13_transform_weight" in weights
    assert "w2_transform_weight" in weights
    # Each expert should have an identity matrix
    eye2 = torch.eye(2, dtype=torch.float32)
    for e in range(3):
        torch.testing.assert_close(weights["w13_transform_weight"][e], eye2)
        torch.testing.assert_close(weights["w2_transform_weight"][e], eye2)


def test_lht_moe_transforms_weights_per_expert_at_load_time():
    """MoE LHT should apply per-expert W @ inv(T).T at load time."""
    ascend_config = Mock(eplb_config=Mock(dynamic_eplb=False))
    config = {"group_size": 2, "hadamard_learning_matrix_size": 2}
    with (
        patch(
            "vllm_ascend.quantization.methods.fake_mx.get_current_vllm_config",
            return_value=_mock_vllm_config(config),
        ),
        patch(
            "vllm_ascend.quantization.methods.fake_mx.get_ascend_config",
            return_value=ascend_config,
        ),
    ):
        method = AscendW4A4MXFP4HadamardLearningFakeFusedMoEMethod()

    layer = torch.nn.Module()
    layer.w13_weight = torch.nn.Parameter(
        torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
            dtype=torch.float32,
        ),
        requires_grad=False,
    )
    layer.w2_weight = torch.nn.Parameter(
        torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
            dtype=torch.float32,
        ),
        requires_grad=False,
    )
    # Per-expert non-orthogonal transform matrices
    layer.w13_transform_weight = torch.nn.Parameter(
        torch.tensor(
            [[[2.0, 0.5], [0.25, 1.5]], [[1.5, 0.25], [0.5, 2.0]]],
            dtype=torch.float32,
        ),
        requires_grad=False,
    )
    layer.w2_transform_weight = torch.nn.Parameter(
        torch.tensor(
            [[[2.0, 0.5], [0.25, 1.5]], [[1.5, 0.25], [0.5, 2.0]]],
            dtype=torch.float32,
        ),
        requires_grad=False,
    )
    original_w13 = layer.w13_weight.detach().clone()
    original_w2 = layer.w2_weight.detach().clone()

    with (
        patch(
            "vllm_ascend.quantization.methods.fake_mx.fake_mx_quantize",
            side_effect=lambda weight, _format, _group_size: weight,
        ),
        patch("vllm_ascend.quantization.methods.fake_mx.maybe_trans_nz", side_effect=lambda weight: weight),
    ):
        method.process_weights_after_loading(layer)

    for e in range(2):
        expected_w13_e = transform_lht_weight(
            original_w13[e], layer.w13_transform_weight.data[e], 2
        ).transpose(0, 1).contiguous()
        expected_w2_e = transform_lht_weight(
            original_w2[e], layer.w2_transform_weight.data[e], 2
        ).transpose(0, 1).contiguous()
        torch.testing.assert_close(layer.w13_weight.data[e], expected_w13_e)
        torch.testing.assert_close(layer.w2_weight.data[e], expected_w2_e)
    assert getattr(layer, "_fake_mx_lht_weight_transformed", False) is True


def test_lht_weight_and_activation_are_mathematically_paired():
    """Verify x' @ W'.T == x @ W.T when x' = x@T and W' = W@inv(T).T."""
    weight = torch.tensor([[1.0, 2.0], [-2.0, 1.0]])
    activation = torch.tensor([[0.5, -1.0]])
    transform = torch.tensor([[2.0, 0.5], [0.25, 1.5]])  # non-orthogonal

    transformed_weight = transform_lht_weight(weight, transform, 2)
    transformed_activation = learned_hadamard_transform(activation, transform)

    torch.testing.assert_close(
        F.linear(transformed_activation, transformed_weight),
        F.linear(activation, weight),
    )
