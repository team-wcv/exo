"""Numerical acceptance for the opt-in Qwen CUDA recurrent kernel."""

import math
import sys
from collections.abc import Callable
from typing import cast

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models.gated_delta import gated_delta_update as reference_update

from exo.worker.engines.mlx.gated_delta_cuda import (
    enable_qwen_cuda_prefill,
    gated_delta_update,
)


@pytest.mark.skipif(sys.platform != "linux", reason="Requires MLX CUDA")
@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
@pytest.mark.parametrize("batch,length", [(1, 1), (2, 7), (1, 256), (1, 1024)])
def test_recurrent_output_and_resumed_state(
    dtype: mx.Dtype, batch: int, length: int
) -> None:
    """Compare both outputs and carried state, including a second invocation."""
    mx.random.seed(73)
    q = mx.random.normal((batch, length, 2, 128))
    k = mx.random.normal(q.shape)
    q = (q / mx.sqrt(mx.sum(q * q, -1, keepdims=True) + 1e-6) / math.sqrt(128)).astype(
        dtype
    )
    k = (k / mx.sqrt(mx.sum(k * k, -1, keepdims=True) + 1e-6)).astype(dtype)
    v = mx.random.normal((batch, length, 6, 128)).astype(dtype)
    a = mx.random.normal((batch, length, 6)).astype(dtype)
    b = mx.random.normal(a.shape).astype(dtype)
    a_log = mx.zeros((6,), dtype=mx.float32) - 2
    dt_bias = mx.ones((6,), dtype=dtype)
    state = mx.random.normal((batch, 6, 128, 128)).astype(dtype) * 0.1
    actual_state = reference_state = state
    for _ in range(2):
        actual, actual_state = gated_delta_update(
            q, k, v, a, b, a_log, dt_bias, actual_state
        )
        expected, reference_state = reference_update(
            q, k, v, a, b, a_log, dt_bias, reference_state
        )
        mx.eval(actual, actual_state, expected, reference_state)
        atol = 2e-3 if dtype == mx.bfloat16 else 2e-4
        rtol = 0.02 if dtype == mx.bfloat16 else 0.002
        assert mx.allclose(actual, expected, atol=atol, rtol=rtol).item()
        assert mx.allclose(actual_state, reference_state, atol=atol, rtol=rtol).item()


def test_masked_inputs_keep_reference_behavior() -> None:
    """The optimized path must not change masked state semantics."""
    q = mx.ones((1, 3, 1, 128)) * 0.01
    v = mx.ones((1, 3, 1, 4))
    a = mx.zeros((1, 3, 1))
    mask = mx.array([[True, False, True]])
    args = (q, q, v, a, a, mx.zeros((1,)), mx.zeros((1,)), None, mask)
    actual = gated_delta_update(*args)
    expected = reference_update(*args)
    for left, right in zip(actual, expected, strict=True):
        assert mx.array_equal(left, right).item()


def test_opt_in_handles_unregistered_checkpoint_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trusted checkpoint execution in MLX LM does not populate sys.modules."""
    namespace: dict[str, object] = {
        "__name__": "exo_test_unregistered_qwen",
        "nn": nn,
        "gated_delta_update": reference_update,
    }
    exec(
        "class Model(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.model_type = 'qwen4_exp'\n"
        "    def __call__(self, x):\n"
        "        return x\n",
        namespace,
    )
    model = cast(Callable[[], nn.Module], namespace["Model"])()
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("EXO_QWEN_CUDA_GATED_DELTA", raising=False)
    assert not enable_qwen_cuda_prefill(model)
    monkeypatch.setenv("EXO_QWEN_CUDA_GATED_DELTA", "1")
    model.model_type = "other_model"
    assert not enable_qwen_cuda_prefill(model)
    model.model_type = "qwen4_exp"
    assert enable_qwen_cuda_prefill(model)
    assert namespace["gated_delta_update"] is gated_delta_update
    assert not enable_qwen_cuda_prefill(model)
