from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn.functional as F

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.quantization.fake_mx import learned_hadamard_transform, randomized_hadamard_transform
from vllm_ascend.quantization.methods.fake_mx import (
    AscendW4A4MXFP4AutoRoundFakeLinearMethod,
    AscendW4A4MXFP4FakeFlatQuantLinearMethod,
    AscendW4A4MXFP4FakeFusedMoEMethod,
    AscendW4A4MXFP4FakeLinearMethod,
    AscendW4A4MXFP4HadamardLearningFakeLinearMethod,
    AscendW4A4MXFP4OmniQuantFakeFusedMoEMethod,
    AscendW4A4MXFP4RHTFakeLinearMethod,
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
