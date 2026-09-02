import subprocess
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


def _fake_grouped_matmul(**kwargs):
    """CPU stand-in for ``torch_npu.npu_grouped_matmul`` (split_item=2 semantics).

    Emulates the op behavior measured on Ascend 910: the output has exactly
    as many rows as x, rows beyond the dispatched tokens are zero-filled,
    and each group computes ``x_slice @ weight[group]``.
    """
    x = kwargs["x"][0]
    weight = kwargs["weight"][0]
    group_list = kwargs["group_list"].tolist()
    if kwargs["group_list_type"] == 0:
        counts = [group_list[0]] + [group_list[i] - group_list[i - 1] for i in range(1, len(group_list))]
    else:
        counts = group_list
    out = torch.zeros_like(x)
    start = 0
    for group_idx, count in enumerate(counts):
        if count > 0:
            out[start : start + count] = x[start : start + count] @ weight[group_idx]
        start += count
    return [out]


def _lht_loop_reference(hidden_states, transform_weight, group_list, group_list_type):
    """Pre-vectorization per-expert loop math (fp32 round-trip per segment)."""
    num_experts = transform_weight.shape[0]
    matrix_size = transform_weight.shape[-1]
    boundaries = cumsum_group_list(group_list, group_list_type, 0, expert_num=num_experts)
    outputs: list[torch.Tensor] = []
    start = 0
    for expert_idx, end_tensor in enumerate(boundaries):
        end = int(end_tensor.item())
        if end > start:
            segment = hidden_states[start:end]
            outputs.append(
                segment.to(torch.float32)
                .reshape(-1, matrix_size)
                .matmul(transform_weight[expert_idx].to(torch.float32))
                .reshape(segment.shape)
                .to(hidden_states.dtype)
            )
        start = end
    if start < hidden_states.shape[0]:
        outputs.append(hidden_states[start:])
    return torch.cat(outputs, dim=0) if outputs else hidden_states


def _flatquant_loop_reference(hidden_states, fc_state, group_list, group_list_type):
    """Pre-vectorization per-expert loop math (transform_flatquant_activation per segment)."""
    num_experts = fc_state["left_trans"].shape[0]
    boundaries = cumsum_group_list(group_list, group_list_type, 0, expert_num=num_experts)
    outputs: list[torch.Tensor] = []
    start = 0
    for expert_idx, end_tensor in enumerate(boundaries):
        end = int(end_tensor.item())
        if end > start:
            segment = hidden_states[start:end]
            left = fc_state["left_trans"][expert_idx]
            right = fc_state["right_trans"][expert_idx]
            left_dim = left.shape[0]
            right_dim = right.shape[0]
            diag = fc_state.get("diag_scale")
            diag = diag[expert_idx] if diag is not None else None
            reshaped = segment.reshape(-1, left_dim, right_dim)
            if diag is not None:
                reshaped = reshaped * diag.to(segment.dtype).reshape(1, left_dim, right_dim)
            transformed = torch.matmul(left.to(segment.dtype).transpose(0, 1), reshaped)
            transformed = torch.matmul(transformed, right.to(segment.dtype))
            outputs.append(transformed.reshape(segment.shape))
        start = end
    if start < hidden_states.shape[0]:
        outputs.append(hidden_states[start:])
    return torch.cat(outputs, dim=0) if outputs else hidden_states


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

        with patch(
            "vllm_ascend.ops.fused_moe.moe_mlp.torch_npu.npu_grouped_matmul",
            side_effect=_fake_grouped_matmul,
            create=True,
        ):
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
        cls.hidden_states = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 3.0], [9.0, 7.0]])

    def test_learned_hadamard_preserves_padding_for_supported_group_list_types(self):
        transforms = torch.stack(
            (
                torch.eye(2),
                torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
            )
        )
        expected = torch.tensor([[1.0, 0.0], [0.0, 1.0], [3.0, 2.0], [9.0, 7.0]])

        for group_list_type, group_list in self.group_lists.items():
            for has_padding in (False, True):
                with (
                    self.subTest(
                        group_list_type=group_list_type,
                        has_padding=has_padding,
                    ),
                    patch(
                        "vllm_ascend.ops.fused_moe.moe_mlp.torch_npu.npu_grouped_matmul",
                        side_effect=_fake_grouped_matmul,
                        create=True,
                    ),
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
        expected = torch.tensor([[1.0, 0.0], [0.0, 1.0], [4.0, 3.0], [9.0, 7.0]])

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
        expected = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 3.0], [9.0, 7.0]])

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


class TestVectorizedPerExpertTransforms(unittest.TestCase):
    """The vectorized implementations must reproduce the previous per-expert loop math."""

    lht_group_lists: ClassVar[dict[int, torch.Tensor]]
    flatquant_group_lists: ClassVar[dict[int, torch.Tensor]]

    @classmethod
    def setUpClass(cls):
        # counts [0, 2, 3, 0]: empty first/last experts, 5 dispatched rows.
        cls.lht_group_lists = {
            0: torch.tensor([0, 2, 5, 5]),
            1: torch.tensor([0, 2, 3, 0]),
        }
        cls.flatquant_group_lists = {
            0: torch.tensor([0, 2, 5, 5]),
            1: torch.tensor([0, 2, 3, 0]),
        }

    def _run_lht(self, hidden_states, transform_weight, group_list, group_list_type):
        with patch(
            "vllm_ascend.ops.fused_moe.moe_mlp.torch_npu.npu_grouped_matmul",
            side_effect=_fake_grouped_matmul,
            create=True,
        ):
            return _apply_expert_learned_hadamard(hidden_states, transform_weight, group_list, group_list_type)

    @staticmethod
    def _make_lht_case(rows):
        torch.manual_seed(7)
        # 4 experts, matrix_size 2, input_dim 4 -> blocks = 2 (sub-row reshape).
        hidden_states = torch.randn(rows, 4)
        # Asymmetric matrices: a transposed-weight bug would change the result.
        transform_weight = torch.randn(4, 2, 2) * 0.5
        return hidden_states, transform_weight

    @staticmethod
    def _make_flatquant_case(rows, with_diag=True):
        torch.manual_seed(9)
        num_experts, left_dim, right_dim = 4, 2, 2
        hidden_states = torch.randn(rows, left_dim * right_dim)
        fc_state = {
            "left_trans": torch.randn(num_experts, left_dim, left_dim) * 0.5,
            "right_trans": torch.randn(num_experts, right_dim, right_dim) * 0.5,
        }
        if with_diag:
            fc_state["diag_scale"] = torch.rand(num_experts, left_dim * right_dim) + 0.5
        return hidden_states, fc_state

    def test_learned_hadamard_matches_loop_reference(self):
        for group_list_type, group_list in self.lht_group_lists.items():
            for rows in (5, 6):  # exact dispatch vs one padding row
                with self.subTest(group_list_type=group_list_type, rows=rows):
                    hidden_states, transform_weight = self._make_lht_case(rows)
                    actual = self._run_lht(hidden_states, transform_weight, group_list, group_list_type)
                    expected = _lht_loop_reference(hidden_states, transform_weight, group_list, group_list_type)
                    torch.testing.assert_close(actual, expected)

    def test_learned_hadamard_all_zero_group_list_returns_input(self):
        hidden_states, transform_weight = self._make_lht_case(2)
        group_list = torch.zeros(4, dtype=torch.int64)
        actual = self._run_lht(hidden_states, transform_weight, group_list, 1)
        torch.testing.assert_close(actual, hidden_states)

    def test_learned_hadamard_single_expert(self):
        torch.manual_seed(8)
        hidden_states = torch.randn(3, 4)
        transform_weight = torch.randn(1, 2, 2) * 0.5
        group_list = torch.tensor([2])  # 2 dispatched rows, 1 padding row
        actual = self._run_lht(hidden_states, transform_weight, group_list, 1)
        expected = _lht_loop_reference(hidden_states, transform_weight, group_list, 1)
        torch.testing.assert_close(actual, expected)

    def test_learned_hadamard_accepts_int32_group_list(self):
        hidden_states, transform_weight = self._make_lht_case(3)
        actual = self._run_lht(hidden_states, transform_weight, torch.tensor([1, 1, 1, 0], dtype=torch.int32), 1)
        expected = _lht_loop_reference(hidden_states, transform_weight, torch.tensor([1, 1, 1, 0]), 1)
        torch.testing.assert_close(actual, expected)

    def test_learned_hadamard_accepts_non_contiguous_weight(self):
        hidden_states, transform_weight = self._make_lht_case(4)
        group_list = torch.tensor([1, 2, 1, 0])
        non_contiguous = transform_weight.transpose(1, 2).contiguous().transpose(1, 2)
        self.assertFalse(non_contiguous.is_contiguous())
        actual = self._run_lht(hidden_states, non_contiguous, group_list, 1)
        expected = _lht_loop_reference(hidden_states, transform_weight, group_list, 1)
        torch.testing.assert_close(actual, expected)

    def test_learned_hadamard_rejects_boundary_overflow(self):
        hidden_states, transform_weight = self._make_lht_case(3)
        with self.assertRaisesRegex(ValueError, "invalid expert token boundaries"):
            self._run_lht(hidden_states, transform_weight, torch.tensor([2, 2, 0, 0]), 1)

    def test_learned_hadamard_rejects_group_count_mismatch(self):
        hidden_states, transform_weight = self._make_lht_case(3)
        with self.assertRaisesRegex(ValueError, "does not match local transforms"):
            self._run_lht(hidden_states, transform_weight, torch.tensor([3]), 1)

    def test_learned_hadamard_rejects_non_square_weight(self):
        hidden_states, _ = self._make_lht_case(3)
        non_square = torch.randn(4, 2, 3)  # last two dims differ
        with self.assertRaisesRegex(ValueError, "must have shape \\[experts, K, K\\]"):
            self._run_lht(hidden_states, non_square, torch.tensor([1, 1, 1, 0]), 1)

    def test_learned_hadamard_rejects_non_floating_weight(self):
        hidden_states, _ = self._make_lht_case(3)
        integer_weight = torch.ones(4, 2, 2, dtype=torch.int64)
        with self.assertRaisesRegex(TypeError, "must be floating point"):
            self._run_lht(hidden_states, integer_weight, torch.tensor([1, 1, 1, 0]), 1)

    def test_learned_hadamard_rejects_type_2_group_list(self):
        hidden_states, transform_weight = self._make_lht_case(3)
        with (
            patch("vllm_ascend.ops.fused_moe.moe_mlp.logger.error") as mock_log_error,
            self.assertRaisesRegex(
                NotImplementedError,
                "do not support group_list_type=2",
            ),
        ):
            self._run_lht(hidden_states, transform_weight, torch.tensor([[1, 2], [2, 1]]), 2)
        mock_log_error.assert_called_once()

    def test_learned_hadamard_returns_empty_hidden_states_unchanged(self):
        hidden_states = torch.empty((0, 4))
        actual = self._run_lht(hidden_states, torch.eye(2).unsqueeze(0), torch.tensor([0]), 1)
        self.assertIs(actual, hidden_states)

    def test_flatquant_matches_loop_reference(self):
        for group_list_type, group_list in self.flatquant_group_lists.items():
            for rows in (5, 6):  # exact dispatch vs one padding row
                with self.subTest(group_list_type=group_list_type, rows=rows):
                    hidden_states, fc_state = self._make_flatquant_case(rows)
                    actual = _apply_expert_flatquant(hidden_states, fc_state, group_list, group_list_type)
                    expected = _flatquant_loop_reference(hidden_states, fc_state, group_list, group_list_type)
                    torch.testing.assert_close(actual, expected)

    def test_flatquant_without_diag_scale(self):
        hidden_states, fc_state = self._make_flatquant_case(6, with_diag=False)
        group_list = torch.tensor([2, 1, 2, 0])
        actual = _apply_expert_flatquant(hidden_states, fc_state, group_list, 1)
        expected = _flatquant_loop_reference(hidden_states, fc_state, group_list, 1)
        torch.testing.assert_close(actual, expected)

    def test_flatquant_chunks_do_not_change_result(self):
        hidden_states, fc_state = self._make_flatquant_case(7)
        group_list = torch.tensor([3, 1, 2, 1])
        expected = _apply_expert_flatquant(hidden_states, fc_state, group_list, 1)
        with patch("vllm_ascend.ops.fused_moe.moe_mlp._FLATQUANT_MAX_ROWS_PER_CHUNK", 2):
            actual = _apply_expert_flatquant(hidden_states, fc_state, group_list, 1)
        torch.testing.assert_close(actual, expected)

    def test_flatquant_accepts_int32_group_list(self):
        hidden_states, fc_state = self._make_flatquant_case(5)
        actual = _apply_expert_flatquant(hidden_states, fc_state, torch.tensor([0, 2, 3, 0], dtype=torch.int32), 1)
        expected = _flatquant_loop_reference(hidden_states, fc_state, torch.tensor([0, 2, 3, 0]), 1)
        torch.testing.assert_close(actual, expected)

    def test_flatquant_accepts_non_contiguous_state(self):
        hidden_states, fc_state = self._make_flatquant_case(6)
        group_list = torch.tensor([1, 3, 1, 1])
        # Transpose-then-view leaves non-contiguous parameter tensors; the
        # parameter tables are rebuilt via torch.cat so this must still work.
        non_contiguous_state = {
            "left_trans": fc_state["left_trans"].transpose(1, 2).contiguous().transpose(1, 2),
            "right_trans": fc_state["right_trans"].transpose(1, 2).contiguous().transpose(1, 2),
            "diag_scale": fc_state["diag_scale"].t().contiguous().t(),
        }
        self.assertFalse(non_contiguous_state["left_trans"].is_contiguous())
        self.assertFalse(non_contiguous_state["right_trans"].is_contiguous())
        self.assertFalse(non_contiguous_state["diag_scale"].is_contiguous())
        actual = _apply_expert_flatquant(hidden_states, non_contiguous_state, group_list, 1)
        expected = _flatquant_loop_reference(hidden_states, fc_state, group_list, 1)
        torch.testing.assert_close(actual, expected)

    def test_flatquant_all_zero_group_list_returns_input(self):
        hidden_states, fc_state = self._make_flatquant_case(2)
        group_list = torch.zeros(4, dtype=torch.int64)
        actual = _apply_expert_flatquant(hidden_states, fc_state, group_list, 1)
        torch.testing.assert_close(actual, hidden_states)

    def test_flatquant_rejects_boundary_overflow(self):
        hidden_states, fc_state = self._make_flatquant_case(3)
        with self.assertRaisesRegex(ValueError, "invalid expert token boundaries"):
            _apply_expert_flatquant(hidden_states, fc_state, torch.tensor([2, 2, 0, 0]), 1)

    def test_flatquant_rejects_type_2_group_list(self):
        hidden_states, fc_state = self._make_flatquant_case(3)
        with (
            patch("vllm_ascend.ops.fused_moe.moe_mlp.logger.error") as mock_log_error,
            self.assertRaisesRegex(
                NotImplementedError,
                "do not support group_list_type=2",
            ),
        ):
            _apply_expert_flatquant(hidden_states, fc_state, torch.tensor([[1, 2], [2, 1]]), 2)
        mock_log_error.assert_called_once()

    def test_flatquant_returns_empty_hidden_states_unchanged(self):
        hidden_states = torch.empty((0, 4))
        actual = _apply_expert_flatquant(hidden_states, self._make_flatquant_case(1)[1], torch.tensor([0, 0, 0, 0]), 1)
        self.assertIs(actual, hidden_states)


def _real_npu_available() -> bool:
    try:
        subprocess.run(["npu-smi", "info"], capture_output=True, check=True)
    except (OSError, subprocess.SubprocessError):
        return False
    return bool(torch.npu.is_available())


@unittest.skipUnless(_real_npu_available(), "requires a real NPU device")
class TestVectorizedPerExpertTransformsOnNpu(unittest.TestCase):
    """End-to-end checks against the real grouped-matmul op on NPU."""

    def test_learned_hadamard_matches_loop_reference_on_npu(self):
        torch.manual_seed(0)
        device = "npu"
        num_experts, matrix_size, input_dim = 4, 128, 512  # blocks = 4
        rows = 12  # 9 dispatched + 3 padding rows
        hidden_states = torch.randn(rows, input_dim, dtype=torch.bfloat16, device=device)
        transform_weight = torch.randn(num_experts, matrix_size, matrix_size, dtype=torch.bfloat16, device=device) * 0.1
        cases = {
            1: torch.tensor([3, 0, 5, 1], dtype=torch.int64, device=device),
            0: torch.tensor([3, 3, 8, 9], dtype=torch.int64, device=device),
        }
        for group_list_type, group_list in cases.items():
            with self.subTest(group_list_type=group_list_type):
                actual = _apply_expert_learned_hadamard(hidden_states, transform_weight, group_list, group_list_type)
                expected = _lht_loop_reference(hidden_states, transform_weight, group_list, group_list_type)
                torch.testing.assert_close(actual, expected)

    def test_flatquant_matches_loop_reference_on_npu(self):
        torch.manual_seed(1)
        device = "npu"
        num_experts, left_dim, right_dim = 4, 4, 32  # input_dim = 128
        rows = 10  # 8 dispatched + 2 padding rows
        hidden_states = torch.randn(rows, left_dim * right_dim, dtype=torch.bfloat16, device=device)
        fc_state = {
            "left_trans": torch.randn(num_experts, left_dim, left_dim, dtype=torch.bfloat16, device=device) * 0.2,
            "right_trans": torch.randn(num_experts, right_dim, right_dim, dtype=torch.bfloat16, device=device) * 0.2,
            "diag_scale": torch.rand(num_experts, left_dim * right_dim, dtype=torch.bfloat16, device=device) + 0.5,
        }
        cases = {
            1: torch.tensor([2, 0, 4, 2], dtype=torch.int64, device=device),
            0: torch.tensor([2, 2, 6, 8], dtype=torch.int64, device=device),
        }
        for group_list_type, group_list in cases.items():
            with self.subTest(group_list_type=group_list_type):
                actual = _apply_expert_flatquant(hidden_states, fc_state, group_list, group_list_type)
                expected = _flatquant_loop_reference(hidden_states, fc_state, group_list, group_list_type)
                torch.testing.assert_close(actual, expected)


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
