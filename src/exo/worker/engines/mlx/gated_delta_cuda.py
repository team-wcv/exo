"""Experimental opt-in CUDA recurrent attention for Qwen4.

MLX LM 0.31.3 uses a Python token loop on CUDA. Keeping the recurrent matrix
in registers avoids materializing that matrix between every token operation.
Unsupported shapes retain MLX LM's reference implementation. Kernel parity tests
do not establish end-to-end correctness: accelerated fleet split-route receipts
produced invalid text. Keep this disabled in normal service configurations.
"""

import os
import sys
from functools import lru_cache
from types import FunctionType
from typing import Protocol, cast

import mlx.core as mx
import mlx.core.fast as fast
import mlx.nn as nn
from mlx_lm.models.gated_delta import compute_g
from mlx_lm.models.gated_delta import gated_delta_update as reference_update

_SOURCE = r"""
const int lane = threadIdx.x;
const int dv = blockIdx.y * blockDim.y + threadIdx.y;
const int n = blockIdx.z;
const int batch = n / Hv;
const int head = n % Hv;
const int key_head = head / (Hv / Hk);
float state[4];
for (int i = 0; i < 4; ++i)
    state[i] = float(state_in[(n * Dv + dv) * 128 + lane + i * 32]);
for (int t = 0; t < T; ++t) {
    const int qbase = ((batch * T + t) * Hk + key_head) * 128;
    const int vbase = ((batch * T + t) * Hv + head) * Dv;
    const int gate = (batch * T + t) * Hv + head;
    float keys[4];
    float memory = 0.0f;
    for (int i = 0; i < 4; ++i) {
        keys[i] = float(k[qbase + lane + i * 32]);
        state[i] *= float(g[gate]);
        memory += state[i] * keys[i];
    }
    for (int offset = 16; offset > 0; offset /= 2)
        memory += __shfl_down_sync(0xffffffff, memory, offset);
    memory = __shfl_sync(0xffffffff, memory, 0);
    const float delta = (float(v[vbase + dv]) - memory) * float(beta[gate]);
    float output = 0.0f;
    for (int i = 0; i < 4; ++i) {
        state[i] += keys[i] * delta;
        output += state[i] * float(q[qbase + lane + i * 32]);
    }
    for (int offset = 16; offset > 0; offset /= 2)
        output += __shfl_down_sync(0xffffffff, output, offset);
    if (lane == 0) y[vbase + dv] = OutT(output);
}
for (int i = 0; i < 4; ++i)
    state_out[(n * Dv + dv) * 128 + lane + i * 32] =
        OutT(state[i]);
"""


class _CudaKernel(Protocol):
    def __call__(
        self,
        *,
        inputs: list[mx.array],
        template: list[tuple[str, int | mx.Dtype]],
        grid: tuple[int, int, int],
        threadgroup: tuple[int, int, int],
        output_shapes: list[tuple[int, ...]],
        output_dtypes: list[mx.Dtype],
    ) -> list[mx.array]: ...


class _CudaKernelFactory(Protocol):
    def __call__(
        self,
        *,
        name: str,
        input_names: list[str],
        output_names: list[str],
        source: str,
    ) -> _CudaKernel: ...


@lru_cache(maxsize=1)
def _kernel() -> _CudaKernel:
    factory = cast(_CudaKernelFactory | None, getattr(fast, "cuda_kernel", None))
    if factory is None:
        raise RuntimeError("Qwen CUDA prefill requires MLX custom CUDA kernel support")
    return factory(
        name="exo_qwen_scalar_gated_delta_v1",
        input_names=["q", "k", "v", "g", "beta", "state_in"],
        output_names=["y", "state_out"],
        source=_SOURCE,
    )


def gated_delta_update(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    a: mx.array,
    b: mx.array,
    A_log: mx.array,  # noqa: N803 - preserve the MLX LM call signature
    dt_bias: mx.array,
    state: mx.array | None = None,
    mask: mx.array | None = None,
    use_kernel: bool = True,
) -> tuple[mx.array, mx.array]:
    """Use a fused CUDA recurrence for scalar gates and 128-wide Qwen heads."""
    supported = (
        use_kernel
        and sys.platform == "linux"
        and mx.default_device() == mx.gpu
        and mask is None
        and q.ndim == 4
        and q.shape[0] > 0
        and q.shape[1] > 0
        and k.shape == q.shape
        and q.shape[-1] == 128
        and v.ndim == 4
        and v.shape[:2] == q.shape[:2]
        and v.shape[-1] % 4 == 0
        and v.shape[-1] > 0
        and v.shape[2] > 0
        and q.shape[2] > 0
        and v.shape[2] % q.shape[2] == 0
        and a.shape == v.shape[:3]
        and b.shape == a.shape
        and A_log.shape == (v.shape[2],)
        and dt_bias.shape == A_log.shape
        and (state is None or state.shape == (q.shape[0], v.shape[2], v.shape[3], 128))
    )
    if not supported:
        return reference_update(q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel)
    batch, length, key_heads, _ = q.shape
    value_heads, value_dim = v.shape[2:]
    if state is None:
        state = mx.zeros((batch, value_heads, value_dim, 128), dtype=mx.float32)
    outputs = _kernel()(
        inputs=[
            q,
            k,
            v,
            compute_g(A_log, a, dt_bias),
            mx.sigmoid(b.astype(mx.float32)),
            state,
        ],
        template=[
            ("T", length),
            ("Hk", key_heads),
            ("Hv", value_heads),
            ("Dv", value_dim),
            ("OutT", q.dtype),
        ],
        grid=(32, value_dim, batch * value_heads),
        threadgroup=(32, 4, 1),
        output_shapes=[v.shape, state.shape],
        output_dtypes=[q.dtype, q.dtype],
    )
    return outputs[0], outputs[1]


def enable_qwen_cuda_prefill(model: nn.Module) -> bool:
    """Install the opt-in kernel only in a Qwen4 module using the known helper."""
    if os.environ.get("EXO_QWEN_CUDA_GATED_DELTA") != "1" or sys.platform != "linux":
        return False
    if cast(object, getattr(model, "model_type", None)) != "qwen4_exp":
        return False
    # MLX LM executes trusted checkpoint modules without adding them to sys.modules.
    call = cast(object, type(model).__call__)
    if not isinstance(call, FunctionType):
        return False
    namespace = cast(dict[str, object], call.__globals__)
    if namespace.get("gated_delta_update") is not reference_update:
        return False
    namespace["gated_delta_update"] = gated_delta_update
    return True
