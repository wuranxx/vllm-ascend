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
from types import SimpleNamespace

import pytest
import torch

import vllm_ascend.quantization.fake_mx as fake_mx_module
from vllm_ascend.quantization.fake_mx import (
    fake_mx_quantize,
    get_fake_mx_backend,
    learned_hadamard_transform,
    maybe_fake_mx_quantize_attention_qkv,
    randomized_hadamard_transform,
)
from vllm_ascend.quantization.kernels import fake_mx as fake_mx_kernel


def test_fake_mx_backend_defaults_to_reference_and_validates_config():
    assert get_fake_mx_backend({}) == "reference"
    assert get_fake_mx_backend({"fake_mx_backend": "kernel"}) == "kernel"
    assert get_fake_mx_backend({"fake_mx_backend": "auto"}) == "auto"

    with pytest.raises(ValueError, match="fake_mx_backend"):
        get_fake_mx_backend({"fake_mx_backend": "unknown"})


def test_fake_mx_backend_is_resolved_inside_stable_wrapper_module(monkeypatch):
    vllm_config = SimpleNamespace(
        quant_config=SimpleNamespace(quant_description={"fake_mx_backend": "kernel"})
    )
    monkeypatch.setattr(fake_mx_module, "get_current_vllm_config_or_none", lambda: vllm_config)

    assert get_fake_mx_backend() == "kernel"


def test_fake_mx_kernel_backend_dispatches_through_adapter(monkeypatch):
    values = torch.ones(1, 4)
    expected = values + 1

    monkeypatch.setattr(fake_mx_kernel, "fake_mx_kernel_support_reason", lambda *args: None)
    monkeypatch.setattr(fake_mx_kernel, "fake_mx_quantize_kernel", lambda *args: expected)
    monkeypatch.setattr(fake_mx_module, "get_fake_mx_backend", lambda: "kernel")

    actual = fake_mx_quantize(values, "mxfp4", group_size=4)

    assert actual is expected


def test_fake_mx_kernel_backend_rejects_unsupported_request(monkeypatch):
    monkeypatch.setattr(
        fake_mx_kernel,
        "fake_mx_kernel_support_reason",
        lambda *args: "test kernel is unavailable",
    )
    monkeypatch.setattr(fake_mx_module, "get_fake_mx_backend", lambda: "kernel")

    with pytest.raises(RuntimeError, match="test kernel is unavailable"):
        fake_mx_quantize(torch.ones(1, 4), "mxfp4", group_size=4)


def test_fake_mx_auto_backend_falls_back_to_reference(monkeypatch):
    values = torch.tensor([[6.0, 5.0, 3.0, 0.25]])
    monkeypatch.setattr(
        fake_mx_kernel,
        "fake_mx_kernel_support_reason",
        lambda *args: "test kernel is unavailable",
    )

    monkeypatch.setattr(fake_mx_module, "get_fake_mx_backend", lambda: "reference")
    expected = fake_mx_quantize(values, "mxfp4", group_size=4)
    monkeypatch.setattr(fake_mx_module, "get_fake_mx_backend", lambda: "auto")
    actual = fake_mx_quantize(values, "mxfp4", group_size=4)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_fake_mx_kernel_adapter_enforces_output_contract(monkeypatch):
    values = torch.ones(1, 4)
    monkeypatch.setattr(
        fake_mx_kernel,
        "_load_external_fake_mx_kernel",
        lambda: (lambda *args: torch.ones(2, 2), None),
    )

    with pytest.raises(ValueError, match="changed shape"):
        fake_mx_kernel.fake_mx_quantize_kernel(values, "mxfp4", 4, 1.0)


def test_fake_mxfp4_matches_amct_e2m1_rounding_and_preserves_dtype():
    values = torch.tensor(
        [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 0.0],
        dtype=torch.float32,
    )
    expected = torch.tensor(
        [0.0, 0.5, 0.5, 1.0, 1.0, 1.5, 1.5, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 6.0, 6.0, 0.0],
        dtype=torch.float32,
    )

    actual = fake_mx_quantize(values, "mxfp4", group_size=values.numel())

    assert actual.dtype == values.dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_fake_mxfp4_applies_per_group_amct_shared_exponent():
    values = torch.tensor([[12.0, 10.0, 1.0, -0.75, 0.0, 0.0, 0.0, 0.0]], dtype=torch.bfloat16)

    actual = fake_mx_quantize(values, "mxfp4", group_size=4)

    expected = torch.tensor([[12.0, 12.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.bfloat16)
    assert actual.shape == values.shape
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_fake_mxfp8_matches_amct_half_away_rounding_without_fp8_dtype():
    values = torch.tensor([1.0, 1.0625, 1.125, 1.1875, 1.25, 1.875, 0.0, 0.0], dtype=torch.float16)

    actual = fake_mx_quantize(values, "mxfp8", group_size=values.numel())

    expected = torch.tensor([1.0, 1.125, 1.125, 1.25, 1.25, 1.875, 0.0, 0.0], dtype=torch.float16)
    assert actual.dtype == torch.float16
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_fake_mxfp8_uses_element_range_when_selecting_shared_scale():
    values = torch.tensor([448.0, 0.25] + [0.0] * 30, dtype=torch.float32)

    actual = fake_mx_quantize(values, "mxfp8", group_size=32)

    assert actual[0] == 448.0
    assert actual[1] == 0.25


def test_fake_mx_rejects_non_divisible_last_dimension_like_amct_unflatten():
    with pytest.raises(ValueError, match="divisible"):
        fake_mx_quantize(torch.ones(5), "mxfp4", group_size=4)


def test_fake_mx_supports_per_block_clip_ratio_tensor():
    values = torch.tensor([[6.0, 5.0, 3.0, 2.0, 6.0, 5.0, 3.0, 2.0]])
    clip_ratio = torch.tensor([[1.0, 0.5]])

    actual = fake_mx_quantize(values, "mxfp4", group_size=4, clip_ratio=clip_ratio)

    torch.testing.assert_close(actual[..., :4], torch.tensor([[6.0, 4.0, 3.0, 2.0]]))
    torch.testing.assert_close(actual[..., 4:], torch.tensor([[3.0, 3.0, 3.0, 2.0]]))


def test_randomized_hadamard_transform_applies_normalized_hd_by_group():
    values = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0]])
    signs = torch.ones(8, dtype=torch.int8)

    actual = randomized_hadamard_transform(values, signs, group_size=4)

    expected = torch.tensor([[2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0]])
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.linalg.vector_norm(actual), torch.linalg.vector_norm(values))


def test_learned_hadamard_transform_matches_amct_q_block_contract():
    values = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    transform = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    actual = learned_hadamard_transform(values, transform)

    expected = values.reshape(-1, 2).matmul(transform).reshape_as(values)
    torch.testing.assert_close(actual, expected)


def test_learned_hadamard_transform_rejects_misaligned_dimension():
    with pytest.raises(ValueError, match="divisible by matrix_size"):
        learned_hadamard_transform(torch.ones(1, 3), torch.eye(2))


@pytest.mark.parametrize("group_size", (0, 3))
def test_randomized_hadamard_transform_rejects_non_power_of_two_group(group_size):
    with pytest.raises(ValueError, match="power of two"):
        randomized_hadamard_transform(torch.ones(1, 4), torch.ones(4), group_size)


def test_attention_qkv_uses_projection_fake_mx_scheme():
    scheme = type(
        "FakeMXScheme",
        (),
        {
            "is_fake_mx": True,
            "mx_format": "mxfp4",
            "group_size": 4,
            "quant_targets": frozenset({"attn-cache"}),
        },
    )()
    projection = type("Projection", (), {})()
    projection.quant_method = type("LinearMethod", (), {"quant_method": scheme})()
    query = torch.tensor([[6.0, 5.0, 3.0, 0.25]])
    key = query + 1.0
    value = query + 2.0

    actual = maybe_fake_mx_quantize_attention_qkv(projection, query, key, value)

    expected = tuple(fake_mx_quantize(tensor, "mxfp4", 4) for tensor in (query, key, value))
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor)


def test_attention_qkv_is_unchanged_for_non_fake_projection():
    projection = type("Projection", (), {"quant_method": object()})()
    qkv = tuple(torch.randn(2, 4) for _ in range(3))

    actual = maybe_fake_mx_quantize_attention_qkv(projection, *qkv)

    assert all(actual_tensor is original_tensor for actual_tensor, original_tensor in zip(actual, qkv))


def test_attention_qkv_is_unchanged_when_attn_cache_target_is_disabled():
    scheme = type(
        "FakeMXScheme",
        (),
        {
            "is_fake_mx": True,
            "mx_format": "mxfp4",
            "group_size": 4,
            "quant_targets": frozenset(),
        },
    )()
    projection = type("Projection", (), {})()
    projection.quant_method = type("LinearMethod", (), {"quant_method": scheme})()
    qkv = tuple(torch.randn(2, 4) for _ in range(3))

    actual = maybe_fake_mx_quantize_attention_qkv(projection, *qkv)

    assert all(actual_tensor is original_tensor for actual_tensor, original_tensor in zip(actual, qkv))


def test_fake_mxfp4_matches_amct_half_away_from_zero_and_shared_exponent():
    values = torch.tensor([[0.25, -0.25, 1.25, -1.25, 7.0, 0.0, 0.0, 0.0]])

    actual = fake_mx_quantize(values, "mxfp4", group_size=4)

    expected = torch.tensor([[0.5, -0.5, 1.5, -1.5, 6.0, 0.0, 0.0, 0.0]])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("tensor", "mx_format", "group_size", "error"),
    [
        (torch.ones(2, dtype=torch.int32), "mxfp4", 32, TypeError),
        (torch.ones(2), "invalid", 32, ValueError),
        (torch.ones(2), "mxfp8", 0, ValueError),
        (torch.ones(2), "mxfp8", 32, ValueError),
    ],
)
def test_fake_mx_rejects_invalid_inputs(tensor, mx_format, group_size, error):
    with pytest.raises(error):
        clip_ratio = 0.0 if mx_format == "mxfp8" and group_size == 32 else 1.0
        fake_mx_quantize(tensor, mx_format, group_size, clip_ratio=clip_ratio)
