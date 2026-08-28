import unittest
from typing import ClassVar
from unittest.mock import patch

import torch
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from vllm_ascend.ops.fused_moe.moe_mlp import (
    _apply_expert_flatquant,
    _apply_expert_learned_hadamard,
    _apply_expert_omniquant,
    cumsum_group_list,
    unified_apply_mlp,
    unquant_apply_mlp,
)
from vllm_ascend.ops.fused_moe.moe_runtime_args import (
    MoEMlpComputeInput,
    MoEQuantParams,
    MoEWeights,
)
from vllm_ascend.ops.fused_moe.moe_stage_params import MoEMxfpParams
from vllm_ascend.quantization.quant_type import QuantType

MXFP4_TEST_DTYPE = getattr(torch, "float4_e2m1fn_x2", torch.float16)


class TestCumsumGroupList(unittest.TestCase):
    glist_dict: ClassVar[dict[int, torch.Tensor]]

    @classmethod
    def setUpClass(cls):
        cls.glist_dict = {
            0: torch.tensor([0, 2, 3, 3]),
            1: torch.tensor([0, 2, 1, 0]),
            2: torch.tensor([[1, 2], [2, 1], [0, 0], [0, 0]]),
        }

    support_combine = [(0, 0), (1, 0), (2, 0), (0, 1)]
    unsupported_combine = [(0, 2), (2, 1), (1, 2)]

    def test_cumsum_group_list_supported_conversion(self):
        for src_list_type, dst_list_type in self.support_combine:
            with self.subTest(src=src_list_type, dst=dst_list_type):
                result = cumsum_group_list(self.glist_dict[src_list_type], src_list_type, dst_list_type, expert_num=4)
                self.assertTrue(torch.equal(result, self.glist_dict[dst_list_type]))

    def test_cumsum_group_list_invalid_type_valueerror(self):
        with self.assertRaises(ValueError) as excinfo:
            cumsum_group_list(self.glist_dict[0], 4, 0)
        self.assertIn("group_list_type should be in [0, 1, 2], but received", str(excinfo.exception))

    def test_cumsum_group_list_unsupported_conversion_notimplementederror(self):
        for src_list_type, dst_list_type in self.unsupported_combine:
            with self.subTest(src=src_list_type, dst=dst_list_type):
                with self.assertRaises(NotImplementedError) as excinfo:
                    cumsum_group_list(self.glist_dict[0], src_list_type, dst_list_type)
                self.assertIn("This feature is under development.", str(excinfo.exception))

    def test_hadamard_learning_selects_transform_after_expert_dispatch(self):
        hidden_states = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 3.0]])
        transforms = torch.stack(
            (
                torch.eye(2),
                torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
            )
        )

        actual = _apply_expert_learned_hadamard(
            hidden_states,
            transforms,
            group_list=torch.tensor([2, 1]),
            group_list_type=1,
        )

        torch.testing.assert_close(actual, torch.tensor([[1.0, 0.0], [0.0, 1.0], [3.0, 2.0]]))


class TestPerExpertActivationTransforms(unittest.TestCase):
    group_lists: ClassVar[dict[int, torch.Tensor]]
    hidden_states: ClassVar[torch.Tensor]

    @classmethod
    def setUpClass(cls):
        cls.group_lists = {
            0: torch.tensor([2, 3]),
            1: torch.tensor([2, 1]),
        }
        cls.hidden_states = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [2.0, 3.0], [9.0, 7.0]]
        )

    def test_learned_hadamard_preserves_padding_for_supported_group_list_types(self):
        transforms = torch.stack(
            (
                torch.eye(2),
                torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
            )
        )
        expected = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [3.0, 2.0], [9.0, 7.0]]
        )

        for group_list_type, group_list in self.group_lists.items():
            for has_padding in (False, True):
                with self.subTest(
                    group_list_type=group_list_type,
                    has_padding=has_padding,
                ):
                    rows = 4 if has_padding else 3
                    actual = _apply_expert_learned_hadamard(
                        self.hidden_states[:rows],
                        transforms,
                        group_list,
                        group_list_type,
                    )
                    torch.testing.assert_close(actual, expected[:rows])

    def test_flatquant_preserves_padding_for_supported_group_list_types(self):
        fc_state = {
            "left_trans": torch.ones((2, 1, 1)),
            "right_trans": torch.eye(2).repeat(2, 1, 1),
            "diag_scale": torch.tensor([[1.0, 1.0], [2.0, 1.0]]),
        }
        expected = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [4.0, 3.0], [9.0, 7.0]]
        )

        for group_list_type, group_list in self.group_lists.items():
            for has_padding in (False, True):
                with self.subTest(
                    group_list_type=group_list_type,
                    has_padding=has_padding,
                ):
                    rows = 4 if has_padding else 3
                    actual = _apply_expert_flatquant(
                        self.hidden_states[:rows],
                        fc_state,
                        group_list,
                        group_list_type,
                    )
                    torch.testing.assert_close(actual, expected[:rows])

    def test_omniquant_preserves_padding_for_supported_group_list_types(self):
        fc_scale = torch.tensor([[1.0, 1.0], [2.0, 1.0]])
        expected = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 3.0], [9.0, 7.0]]
        )

        for group_list_type, group_list in self.group_lists.items():
            for has_padding in (False, True):
                with self.subTest(
                    group_list_type=group_list_type,
                    has_padding=has_padding,
                ):
                    rows = 4 if has_padding else 3
                    actual = _apply_expert_omniquant(
                        self.hidden_states[:rows],
                        fc_scale,
                        group_list,
                        group_list_type,
                    )
                    torch.testing.assert_close(actual, expected[:rows])

    def test_omniquant_preserves_all_padding_for_supported_group_list_types(self):
        fc_scale = torch.tensor([[2.0, 2.0], [3.0, 3.0]])
        hidden_states = torch.tensor([[9.0, 7.0]])
        empty_group_lists = {
            0: torch.tensor([0, 0]),
            1: torch.tensor([0, 0]),
        }

        for group_list_type, group_list in empty_group_lists.items():
            with self.subTest(group_list_type=group_list_type):
                actual = _apply_expert_omniquant(
                    hidden_states,
                    fc_scale,
                    group_list,
                    group_list_type,
                )
                torch.testing.assert_close(actual, hidden_states)

    def test_omniquant_returns_empty_hidden_states_unchanged(self):
        hidden_states = torch.empty((0, 2))

        actual = _apply_expert_omniquant(
            hidden_states,
            torch.ones((2, 2)),
            torch.tensor([0, 0]),
            group_list_type=0,
        )

        self.assertIs(actual, hidden_states)

    def test_per_expert_transform_rejects_type_2_group_list(self):
        with (
            patch("vllm_ascend.ops.fused_moe.moe_mlp.logger.error") as mock_log_error,
            self.assertRaisesRegex(
                NotImplementedError,
                "do not support group_list_type=2",
            ),
        ):
            _apply_expert_omniquant(
                self.hidden_states[:3],
                torch.ones((2, 2)),
                torch.tensor([[4, 2], [5, 1]]),
                group_list_type=2,
            )
        mock_log_error.assert_called_once()

    def test_omniquant_rejects_boundaries_beyond_hidden_states(self):
        invalid_group_lists = {
            0: torch.tensor([2, 4]),
            1: torch.tensor([2, 2]),
        }

        for group_list_type, group_list in invalid_group_lists.items():
            with (
                self.subTest(group_list_type=group_list_type),
                self.assertRaisesRegex(ValueError, "invalid expert token boundaries"),
            ):
                _apply_expert_omniquant(
                    self.hidden_states[:3],
                    torch.ones((2, 2)),
                    group_list,
                    group_list_type,
                )


class TestW4A8RuntimeFlags(unittest.TestCase):
    def test_w4a8_per_channel_gmm_swiglu_flag(self):
        self.assertTrue(
            MoEQuantParams(quant_type=QuantType.W4A8, is_per_channel_weight=True).use_w4a8_per_channel_gmm_swiglu
        )
        self.assertFalse(
            MoEQuantParams(quant_type=QuantType.W4A8, is_per_channel_weight=False).use_w4a8_per_channel_gmm_swiglu
        )
        self.assertFalse(
            MoEQuantParams(quant_type=QuantType.W8A8, is_per_channel_weight=True).use_w4a8_per_channel_gmm_swiglu
        )


class TestUnifiedApplyMlpRequest(unittest.TestCase):
    def test_unquant_apply_mlp_wraps_tensor_weights_for_grouped_matmul(self):
        hidden_states = torch.randn(2, 8)
        gate_up_out = torch.randn(2, 16)
        expected = torch.randn(2, 8)
        w1 = torch.randn(2, 8, 16)
        w2 = torch.randn(2, 8, 8)

        with (
            patch(
                "vllm_ascend.ops.fused_moe.moe_mlp.torch_npu.npu_grouped_matmul",
                side_effect=[[gate_up_out], [expected]],
                create=True,
            ) as mock_grouped_matmul,
            patch(
                "vllm_ascend.ops.fused_moe.moe_mlp.torch_npu.npu_swiglu",
                return_value=gate_up_out,
                create=True,
            ),
        ):
            output, _ = unquant_apply_mlp(
                hidden_states=hidden_states,
                w1=w1,
                w2=w2,
                group_list=torch.tensor([1, 1]),
                need_trans=True,
            )

        self.assertTrue(output is expected)
        first_call, second_call = mock_grouped_matmul.call_args_list
        self.assertEqual(len(first_call.kwargs["weight"]), 1)
        self.assertEqual(len(second_call.kwargs["weight"]), 1)
        self.assertEqual(first_call.kwargs["weight"][0].shape, torch.Size([2, 16, 8]))
        self.assertEqual(second_call.kwargs["weight"][0].shape, torch.Size([2, 8, 8]))

    def test_request_unquant_path(self):
        hidden_states = torch.randn(2, 8)
        expected = torch.randn(2, 8)
        mlp_compute_input = MoEMlpComputeInput(
            hidden_states=hidden_states,
            group_list=torch.tensor([2, 2], dtype=torch.int64),
            group_list_type=1,
            dynamic_scale=None,
            topk_scales=None,
            weights=MoEWeights(
                w1=torch.randn(1, 16, 8),
                w2=torch.randn(1, 8, 8),
                w1_bias=torch.randn(1, 16),
                w2_bias=torch.randn(1, 8),
            ),
            quant=MoEQuantParams(quant_type=QuantType.NONE),
            fusion=False,
            activation="silu",
            need_trans=False,
            dynamic_eplb=False,
        )

        with (
            patch("vllm_ascend.ops.fused_moe.moe_mlp.unquant_apply_mlp", return_value=expected) as mock_unquant,
            patch("vllm_ascend.ops.fused_moe.moe_mlp.quant_apply_mlp") as mock_quant,
        ):
            output = unified_apply_mlp(mlp_compute_input=mlp_compute_input)

        self.assertTrue(output is expected)
        mock_unquant.assert_called_once()
        self.assertEqual(mock_unquant.call_args.kwargs["activation"], "silu")
        self.assertFalse(mock_unquant.call_args.kwargs["need_trans"])
        mock_quant.assert_not_called()

    def test_request_quant_path(self):
        for quant_type, mxfp_dtype in (
            (QuantType.MXFP8, torch.float8_e4m3fn),
            (QuantType.MXFP4, MXFP4_TEST_DTYPE),
        ):
            with self.subTest(quant_type=quant_type):
                hidden_states = torch.randn(2, 8)
                expected = torch.randn(2, 8)
                mlp_compute_input = MoEMlpComputeInput(
                    hidden_states=hidden_states,
                    group_list=torch.tensor([2, 2], dtype=torch.int64),
                    group_list_type=1,
                    dynamic_scale=torch.randn(2, 1),
                    topk_scales=None,
                    weights=MoEWeights(
                        w1=torch.randn(1, 16, 8),
                        w2=torch.randn(1, 8, 8),
                        w1_scale=[torch.randn(1)],
                        w2_scale=[torch.randn(1)],
                    ),
                    quant=MoEQuantParams(
                        quant_type=quant_type,
                        mxfp=MoEMxfpParams(
                            act_quant_type=mxfp_dtype,
                            weight_quant_type=mxfp_dtype,
                            use_bf16=False,
                        ),
                    ),
                    fusion=True,
                    activation="silu",
                    need_trans=False,
                    dynamic_eplb=True,
                )

                with (
                    patch("vllm_ascend.ops.fused_moe.moe_mlp.quant_apply_mlp", return_value=expected) as mock_quant,
                    patch("vllm_ascend.ops.fused_moe.moe_mlp.unquant_apply_mlp") as mock_unquant,
                ):
                    output = unified_apply_mlp(mlp_compute_input=mlp_compute_input)

                self.assertTrue(output is expected)
                mock_quant.assert_called_once()
                quant_kwargs = mock_quant.call_args.kwargs
                self.assertTrue(quant_kwargs["use_mxfp_quant"])
                self.assertTrue(quant_kwargs["fusion"])
                self.assertTrue(quant_kwargs["dynamic_eplb"])
                self.assertEqual(quant_kwargs["act_quant_type"], mxfp_dtype)
                self.assertEqual(quant_kwargs["weight_quant_type"], mxfp_dtype)
                self.assertFalse(quant_kwargs["use_bf16"])
                mock_unquant.assert_not_called()

    def test_request_quant_path_passes_w4a8_per_channel_flag(self):
        hidden_states = torch.randn(2, 8)
        expected = torch.randn(2, 8)
        mlp_compute_input = MoEMlpComputeInput(
            hidden_states=hidden_states,
            group_list=torch.tensor([2, 2], dtype=torch.int64),
            group_list_type=1,
            dynamic_scale=torch.randn(2, 1),
            topk_scales=None,
            weights=MoEWeights(
                w1=torch.randn(1, 16, 8),
                w2=torch.randn(1, 8, 8),
                w1_scale=[torch.randn(1, 16)],
                w2_scale=[torch.randn(1, 8)],
            ),
            quant=MoEQuantParams(quant_type=QuantType.W4A8, is_per_channel_weight=True),
            fusion=False,
            activation="silu",
            need_trans=False,
            dynamic_eplb=False,
        )

        with (
            patch("vllm_ascend.ops.fused_moe.moe_mlp.quant_apply_mlp", return_value=expected) as mock_quant,
            patch("vllm_ascend.ops.fused_moe.moe_mlp.unquant_apply_mlp") as mock_unquant,
        ):
            output = unified_apply_mlp(mlp_compute_input=mlp_compute_input)

        self.assertTrue(output is expected)
        quant_kwargs = mock_quant.call_args.kwargs
        self.assertTrue(quant_kwargs["use_w4a8_per_channel_gmm_swiglu"])
        mock_unquant.assert_not_called()

    def test_request_quant_path_passes_swiglustep_activation(self):
        expected = torch.randn(1, 2)
        mlp_compute_input = MoEMlpComputeInput(
            hidden_states=torch.ones((1, 2), dtype=torch.float32),
            group_list=torch.tensor([1], dtype=torch.int64),
            group_list_type=1,
            dynamic_scale=None,
            topk_scales=None,
            weights=MoEWeights(
                w1=[torch.ones((1, 2, 4), dtype=torch.float32)],
                w2=[torch.ones((1, 2, 2), dtype=torch.float32)],
                w1_scale=[torch.ones((1,), dtype=torch.float32)],
                w2_scale=[torch.ones((1,), dtype=torch.float32)],
            ),
            quant=MoEQuantParams(quant_type=QuantType.W8A8),
            fusion=True,
            activation=MoEActivation.SWIGLUSTEP,
            swiglu_limit=5.0,
        )

        with (
            patch("vllm_ascend.ops.fused_moe.moe_mlp.quant_apply_mlp", return_value=expected) as mock_quant,
            patch("vllm_ascend.ops.fused_moe.moe_mlp.unquant_apply_mlp") as mock_unquant,
        ):
            output = unified_apply_mlp(mlp_compute_input=mlp_compute_input)

        self.assertTrue(output is expected)
        quant_kwargs = mock_quant.call_args.kwargs
        self.assertEqual(quant_kwargs["activation"], MoEActivation.SWIGLUSTEP)
        self.assertEqual(quant_kwargs["swiglu_limit"], 5.0)
        mock_unquant.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
