# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
#
"""Performance regression guard for the vectorized per-expert fake-MX transforms.

``_apply_expert_learned_hadamard`` (grouped matmul) and
``_apply_expert_flatquant`` (row-mapped batched matmul) replaced a per-expert
Python loop that issued one ``.item()`` sync and several small kernels per
expert (see .do/task/20260825-qwen36-moe-w4a4-code-plan/design-v-14.md).
These tests fail if the vectorized paths regress to launch-bound behavior.

Measured on Ascend 910 (E=128, K=2048, 2048 rows): LHT ~13x, FlatQuant ~10x
over the loop reference. The threshold is set conservatively to absorb
machine noise while still catching a regression back to the loop.
"""

import time

import pytest
import torch

from vllm_ascend.ops.fused_moe.moe_mlp import (
    _apply_expert_flatquant,
    _apply_expert_learned_hadamard,
)

# Production-like shapes: TP=2 + EP=2 leaves 128 local experts, FC1 input dim
# 2048, FlatQuant L=16/R=128, LHT matrix size 128.
NUM_EXPERTS = 128
INPUT_DIM = 2048
LEFT_DIM, RIGHT_DIM = 16, 128
MATRIX_SIZE = 128
DECODE_ROWS = 2048
MIN_SPEEDUP = 3.0
WARMUP_ROUNDS = 3
TIMING_ROUNDS = 10

MIN_SPEEDUP_REASON = (
    f"vectorized per-expert transform is slower than {MIN_SPEEDUP}x the loop "
    "reference; per-expert sync/launch regression suspected"
)


def _benchmark(fn) -> float:
    for _ in range(WARMUP_ROUNDS):
        fn()
    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(TIMING_ROUNDS):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - start) / TIMING_ROUNDS


def _make_group_list() -> torch.Tensor:
    counts = torch.full((NUM_EXPERTS,), DECODE_ROWS // NUM_EXPERTS, dtype=torch.int64, device="npu")
    counts[: DECODE_ROWS - (DECODE_ROWS // NUM_EXPERTS) * NUM_EXPERTS] += 1
    return counts


def _lht_loop_reference(hidden_states, transform_weight, counts):
    boundaries = counts.cumsum(0).tolist()
    outputs: list[torch.Tensor] = []
    start = 0
    for expert_idx, end in enumerate(boundaries):
        if end > start:
            segment = hidden_states[start:end]
            outputs.append(
                segment.to(torch.float32)
                .reshape(-1, MATRIX_SIZE)
                .matmul(transform_weight[expert_idx].to(torch.float32))
                .reshape(segment.shape)
                .to(hidden_states.dtype)
            )
        start = end
    return torch.cat(outputs, dim=0)


def _flatquant_loop_reference(hidden_states, left, right, diag, counts):
    boundaries = counts.cumsum(0).tolist()
    outputs: list[torch.Tensor] = []
    start = 0
    for expert_idx, end in enumerate(boundaries):
        if end > start:
            segment = hidden_states[start:end]
            reshaped = segment.reshape(-1, LEFT_DIM, RIGHT_DIM)
            reshaped = reshaped * diag[expert_idx].reshape(1, LEFT_DIM, RIGHT_DIM)
            transformed = torch.matmul(left[expert_idx].transpose(0, 1), reshaped)
            outputs.append(torch.matmul(transformed, right[expert_idx]).reshape(segment.shape))
        start = end
    return torch.cat(outputs, dim=0)


@pytest.mark.parametrize(
    ("fast_fn", "loop_fn", "make_inputs"),
    [
        (
            lambda x, tw, gl: _apply_expert_learned_hadamard(x, tw, gl, 1),
            lambda x, tw, gl: _lht_loop_reference(x, tw, gl),
            lambda: (
                torch.randn(DECODE_ROWS, INPUT_DIM, dtype=torch.bfloat16, device="npu"),
                torch.randn(NUM_EXPERTS, MATRIX_SIZE, MATRIX_SIZE, dtype=torch.bfloat16, device="npu")
                * 0.1,
            ),
        ),
        (
            lambda x, state, gl: _apply_expert_flatquant(x, state, gl, 1),
            lambda x, state, gl: _flatquant_loop_reference(
                x, state["left_trans"], state["right_trans"], state["diag_scale"], gl
            ),
            lambda: (
                torch.randn(DECODE_ROWS, INPUT_DIM, dtype=torch.bfloat16, device="npu"),
                {
                    "left_trans": torch.randn(NUM_EXPERTS, LEFT_DIM, LEFT_DIM, dtype=torch.bfloat16, device="npu")
                    * 0.1,
                    "right_trans": torch.randn(
                        NUM_EXPERTS, RIGHT_DIM, RIGHT_DIM, dtype=torch.bfloat16, device="npu"
                    )
                    * 0.1,
                    "diag_scale": torch.rand(NUM_EXPERTS, INPUT_DIM, dtype=torch.bfloat16, device="npu")
                    + 0.5,
                },
            ),
        ),
    ],
    ids=["learned_hadamard", "flatquant"],
)
def test_per_expert_transform_faster_than_loop(fast_fn, loop_fn, make_inputs):
    torch.manual_seed(0)
    hidden_states, params = make_inputs()
    group_list = _make_group_list()

    fast_seconds = _benchmark(lambda: fast_fn(hidden_states, params, group_list))
    loop_seconds = _benchmark(lambda: loop_fn(hidden_states, params, group_list))

    assert loop_seconds / fast_seconds >= MIN_SPEEDUP, (
        f"{MIN_SPEEDUP_REASON} (loop={loop_seconds * 1e3:.2f} ms, "
        f"fast={fast_seconds * 1e3:.2f} ms, "
        f"speedup={loop_seconds / fast_seconds:.2f}x)"
    )
