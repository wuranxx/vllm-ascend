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
    _AscendPrequantizedWeightFakeMXFusedMoEMethod,
    _AscendPrequantizedWeightFakeMXLinearMethod,
    _inverse_fp32,
    transform_lht_weight,
)


def _mock_vllm_config(quant_description):
    return Mock(quant_config=Mock(quant_description=quant_description))


class _PrequantizedLinear(_AscendPrequantizedWeightFakeMXLinearMethod):
    """Concrete prequantized Linear for testing.

    Production code only registers non-prequantized Linear schemes; this
    subclass materialises the prequantized contract (skip QDQ when
    ``fake_mx_weight_state='prequantized_qdq'``) for direct unit testing.
    """

    mx_format = "mxfp4"


class _PrequantizedMoE(_AscendPrequantizedWeightFakeMXFusedMoEMethod):
    """Concrete prequantized MoE for testing (same rationale as above)."""

    mx_format = "mxfp4"


def test_fake_mx_init_ignores_unrecognized_quant_targets_key():
    """The base fake-MX Linear method does not implement ``quant_targets``
    selection (no per-target routing in production code). Passing an
    unrecognized ``fake_mx_quant_targets`` key must not raise and must not
    alter the default QDQ-on-every-layer behaviour.
    """
    with patch(
        "vllm_ascend.quantization.methods.fake_mx.get_current_vllm_config",
        return_value=_mock_vllm_config(
            {"group_size": 32, "fake_mx_quant_targets": ["attn-cache"]}
        ),
    ):
        method = AscendW4A4MXFP4FakeLinearMethod()
    # No ``quant_targets`` attribute is materialised; the only observable
    # contract is that init succeeds and QDQ stays enabled (default).
    assert not hasattr(method, "quant_targets")
    assert method.prequantized_weight is False


def test_prequantized_algorithm_rejects_missing_weight_state():
    """A prequantized scheme (``required_weight_state='prequantized_qdq'``)
    must reject checkpoints that do not carry the matching marker.

    Uses ``_PrequantizedLinear`` because production code registers no
    concrete prequantized Linear scheme (OmniQuant/AutoRound Linear both
    perform QDQ at load time and are not prequantized).
    """
    with (
        patch(
            "vllm_ascend.quantization.methods.fake_mx.get_current_vllm_config",
            return_value=_mock_vllm_config({"group_size": 32}),
        ),
        pytest.raises(ValueError, match="prequantized_qdq"),
    ):
        _PrequantizedLinear()


def test_autoround_preserves_prequantized_weight_during_post_load():
    """A prequantized scheme (``prequantized_weight=True``) must skip the
    QDQ step in ``process_weights_after_loading`` and preserve the loaded
    weight bit-for-bit.

    Production AutoRound Linear is NOT prequantized (it performs QDQ at load
    time); this test validates the prequantized contract directly via
    ``_PrequantizedLinear`` so the skip-QDQ path stays covered.
    """
    config = {
        "group_size": 4,
        "fake_mx_weight_state": "prequantized_qdq",
    }
    with patch(
        "vllm_ascend.quantization.methods.fake_mx.get_current_vllm_config",
        return_value=_mock_vllm_config(config),
    ):
        method = _PrequantizedLinear()

    layer = torch.nn.Linear(4, 1, bias=False)
    layer.weight.data.copy_(torch.tensor([[6.0, 5.5, 3.0, 0.25]]))
    expected = layer.weight.detach().clone()

    method.process_weights_after_loading(layer)

    torch.testing.assert_close(layer.weight, expected)
    assert getattr(layer, "_fake_mx_weight_processed", False) is True


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


def test_rht_linear_does_not_require_weight_state_marker():
    """Linear RHT rotates weights at load time and accepts the original
    BF16 checkpoint directly, so it does NOT set ``required_weight_state``.

    Passing an unrelated ``fake_mx_weight_state`` marker must not raise —
    RHT ignores the marker entirely (consistent with MoE RHT after the
    F1 fix in work-03).
    """
    config = {
        "group_size": 32,
        "rht_group_size": 32,
        "fake_mx_weight_state": "prequantized_qdq",  # unrelated marker
    }
    with patch(
        "vllm_ascend.quantization.methods.fake_mx.get_current_vllm_config",
        return_value=_mock_vllm_config(config),
    ):
        # Should not raise — RHT does not validate weight_state.
        method = AscendW4A4MXFP4RHTFakeLinearMethod()
    assert method.required_weight_state is None
    assert method.algorithm == "rht"


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
    """LHT Linear loads the transform matrix from ``lht_params_path`` and
    applies ``W @ inv(T).T`` to the weight at load time, before fake-MX QDQ.

    Production code requires ``lht_params_path`` (no default); the previous
    test omitted it and hit the params_path validation error. The QDQ step
    is mocked as identity so the weight-transform assertion is exact.
    """
    transform = torch.tensor([[1.0, 1.0], [1.0, -1.0]])
    config = {
        "group_size": 2,
        "hadamard_learning_matrix_size": 2,
        "lht_params_path": "/fake/path.pt",
    }
    with (
        patch(
            "vllm_ascend.quantization.methods.fake_mx.get_current_vllm_config",
            return_value=_mock_vllm_config(config),
        ),
        patch(
            "vllm_ascend.quantization.methods.fake_mx._load_transform_params",
            return_value={"test.transform_weight": transform},
        ),
        patch(
            "vllm_ascend.quantization.methods.fake_mx.fake_mx_quantize",
            side_effect=lambda weight, _format, _group_size, **_: weight,
        ) as fake_quantize,
    ):
        method = AscendW4A4MXFP4HadamardLearningFakeLinearMethod()

        layer = torch.nn.Linear(2, 2, bias=False)
        layer.weight.data.copy_(torch.eye(2))
        layer.prefix = "test"
        layer.transform_weight = torch.nn.Parameter(torch.eye(2), requires_grad=False)
        method.process_weights_after_loading(layer)

        # Transform loaded from params path
        torch.testing.assert_close(layer.transform_weight.data, transform)
        # Weight transformed to W @ inv(T).T (QDQ mocked as identity)
        expected_weight = transform_lht_weight(torch.eye(2), transform, 2)
        torch.testing.assert_close(layer.weight.data, expected_weight.to(layer.weight.dtype))
        # QDQ ran exactly once on the transformed weight
        assert fake_quantize.call_count == 1
        assert getattr(layer, "_fake_mx_weight_processed", False) is True


def test_flatquant_applies_transform_clip_and_mxfp4_qdq():
    """FlatQuant Linear loads left_trans/right_trans/diag_scale from
    ``flatquant_params_path``, applies ``W' = inv(left) @ (W/diag) @ inv(right).T``
    at load time, then transforms activation online as
    ``x' = left.T @ (x*diag) @ right`` before QDQ.

    Production code requires ``flatquant_params_path`` (no default) and
    forces ``clip_ratio=1.0`` at load time. With identity transforms and
    diag=ones, both weight and activation are preserved; QDQ is mocked as
    identity so the assertion is exact.
    """
    config = {
        "group_size": 4,
        "max_supported_tp": 4,
        "flatquant_params_path": "/fake/path.pt",
        "flatquant_use_diag": False,
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
        patch(
            "vllm_ascend.quantization.methods.fake_mx._load_transform_params",
            return_value={
                "test.left_trans": torch.eye(2),
                "test.right_trans": torch.eye(2),
                "test.diag_scale": torch.ones(4),
            },
        ),
        patch(
            "vllm_ascend.quantization.methods.fake_mx.fake_mx_quantize",
            side_effect=lambda weight, _format, _group_size, **_: weight,
        ),
    ):
        method = AscendW4A4MXFP4FakeFlatQuantLinearMethod()

        layer = torch.nn.Linear(4, 4, bias=False)
        layer.weight.data.copy_(torch.eye(4))
        layer.prefix = "test"
        layer.left_trans = torch.nn.Parameter(torch.eye(2), requires_grad=False)
        layer.right_trans = torch.nn.Parameter(torch.eye(2), requires_grad=False)
        layer.clip_ratio = torch.nn.Parameter(torch.tensor([0.5]), requires_grad=False)
        layer.diag_scale = torch.nn.Parameter(torch.ones(4), requires_grad=False)
        method.process_weights_after_loading(layer)

        activation = torch.tensor([[6.0, 5.0, 3.0, 2.0]])
        actual = method.apply(layer, activation)

        # identity transforms + diag=ones => weight and activation preserved (QDQ mocked)
        torch.testing.assert_close(layer.weight.data, torch.eye(4))
        torch.testing.assert_close(actual, F.linear(activation, layer.weight))


def test_flatquant_linear_does_not_require_weight_state_marker():
    """FlatQuant Linear transforms weights at load time and accepts the
    original BF16 checkpoint directly, so it does NOT set
    ``required_weight_state``.

    Passing an unrelated ``fake_mx_weight_state`` marker must not raise.
    Also verifies ``__init__`` no longer crashes on uninitialized TP group
    (the previous test omitted the TP mock).
    """
    config = {
        "group_size": 4,
        "max_supported_tp": 4,
        "flatquant_params_path": "/fake/path.pt",
        "fake_mx_weight_state": "rht_rotated_fp",  # unrelated marker
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
        # Should not raise — FlatQuant does not validate weight_state.
        method = AscendW4A4MXFP4FakeFlatQuantLinearMethod()
    assert method.required_weight_state is None
    assert method.algorithm == "flatquant"


def test_prequantized_moe_preserves_prequantized_expert_weights():
    """A prequantized MoE scheme (``prequantized_weight=True``) must skip
    the QDQ step in ``process_weights_after_loading`` and only apply the
    layout transpose (checkpoint [E,N,K] -> Ascend GMM [E,K,N]).

    Production OmniQuant/AutoRound MoE are NOT prequantized (they perform
    QDQ at load time); this test validates the prequantized contract
    directly via ``_PrequantizedMoE`` so the skip-QDQ path stays covered.
    """
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
        method = _PrequantizedMoE()

    layer = torch.nn.Module()
    layer.w13_weight = torch.nn.Parameter(torch.randn(2, 8, 4), requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(torch.randn(2, 4, 4), requires_grad=False)
    expected_w13 = layer.w13_weight.detach().clone()
    expected_w2 = layer.w2_weight.detach().clone()

    with patch("vllm_ascend.quantization.methods.fake_mx.maybe_trans_nz", side_effect=lambda weight: weight):
        method.process_weights_after_loading(layer)

    # QDQ skipped (prequantized_weight=True); only layout transpose applied
    torch.testing.assert_close(layer.w13_weight, expected_w13.transpose(1, 2).contiguous())
    torch.testing.assert_close(layer.w2_weight, expected_w2.transpose(1, 2).contiguous())
    assert getattr(layer, "_fake_mx_weight_processed", False) is True


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


# ---- F3: OmniQuant MoE fix tests ----


def test_omniquant_moe_requires_params_path():
    """OmniQuant MoE must raise when omniquant_params_path is missing."""
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
        pytest.raises(ValueError, match="omniquant_params_path"),
    ):
        AscendW4A4MXFP4OmniQuantFakeFusedMoEMethod()


def test_omniquant_moe_scales_weights_at_load_time():
    """OmniQuant MoE must scale weight' = weight * exp(log_scale) at load time,
    and stash per-expert fc1/fc2 scales on the layer for runtime activation
    down-scale. QDQ is mocked as identity so the scale assertion is exact.
    """
    # 2 experts, hidden=4, intermediate=4. log_scale=ln(2) => scale=2 on one dim.
    fc1_log_scale = torch.tensor(
        [[0.6931, 0.0, 0.0, 0.0], [0.0, 0.6931, 0.0, 0.0]], dtype=torch.float32
    )
    fc2_log_scale = torch.zeros(2, 4, dtype=torch.float32)  # scale=1 (no-op)
    config = {"group_size": 4, "omniquant_params_path": "/fake/path.pt"}
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
        patch(
            "vllm_ascend.quantization.methods.fake_mx._load_transform_params",
            return_value={
                "test.w13_log_scale": fc1_log_scale,
                "test.w2_log_scale": fc2_log_scale,
            },
        ),
        patch(
            "vllm_ascend.quantization.methods.fake_mx.fake_mx_quantize",
            side_effect=lambda weight, _format, _group_size, **_: weight,
        ),
        patch("vllm_ascend.quantization.methods.fake_mx.maybe_trans_nz", side_effect=lambda weight: weight),
    ):
        method = AscendW4A4MXFP4OmniQuantFakeFusedMoEMethod()

        layer = torch.nn.Module()
        layer.prefix = "test"
        layer.w13_weight = torch.nn.Parameter(torch.ones(2, 8, 4), requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(torch.ones(2, 4, 4), requires_grad=False)
        # log_scale parameters created by get_weight in production; mirror here
        # so _copy_transform_param can copy loaded values into them.
        layer.w13_log_scale = torch.nn.Parameter(torch.zeros(2, 4), requires_grad=False)
        layer.w2_log_scale = torch.nn.Parameter(torch.zeros(2, 4), requires_grad=False)
        original_w13 = layer.w13_weight.detach().clone()
        original_w2 = layer.w2_weight.detach().clone()
        method.process_weights_after_loading(layer)

        # weight' = weight * scale  (broadcasting [E, 2*inter, hidden] * [E, 1, hidden])
        # then base process_weights_after_loading transposes [E, 2*inter, hidden] -> [E, hidden, 2*inter]
        expected_fc1_scale = torch.exp(fc1_log_scale).clamp(min=1e-4, max=1e4)
        expected_w13 = (original_w13 * expected_fc1_scale.unsqueeze(1)).transpose(1, 2).contiguous()
        torch.testing.assert_close(layer.w13_weight.data, expected_w13.to(layer.w13_weight.dtype))
        # fc2 scale = 1 => w2 unchanged by scale (QDQ mocked as identity too)
        torch.testing.assert_close(layer.w2_weight.data, original_w2.transpose(1, 2).contiguous())
        # per-expert scales stashed on layer
        assert hasattr(layer, "_fake_mx_fc1_scale")
        assert hasattr(layer, "_fake_mx_fc2_scale")
        assert getattr(layer, "_fake_mx_omniquant_processed", False) is True


def test_omniquant_moe_weight_and_activation_are_mathematically_paired():
    """Verify (x / scale) @ (weight * scale).T == x @ weight.T per expert.

    Mirrors Linear OmniQuant's math contract: weight is pre-scaled at load
    time, activation is divided by the same scale per-expert at runtime, so
    the linear output is preserved while MX QDQ error is reduced.
    """
    weight = torch.tensor([[1.0, 2.0, -1.0, 0.5], [0.25, -2.0, 1.0, 3.0]])  # [out=2, in=4]
    activation = torch.tensor([[0.5, -1.0, 2.0, 1.0]])  # [1, in=4]
    scale = torch.tensor([2.0, 0.5, 1.0, 4.0])  # per-input-dim, non-uniform

    scaled_weight = weight * scale  # [out, in] * [in] => broadcast on last dim
    scaled_activation = activation / scale  # [1, in] / [in]

    torch.testing.assert_close(
        F.linear(scaled_activation, scaled_weight),
        F.linear(activation, weight),
    )
