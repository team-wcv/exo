from exo.worker.engines.mlx.generator.generate import (
    _prefill_step_size,  # pyright: ignore[reportPrivateUsage]
)


def test_kv_only_prefill_keeps_large_chunks() -> None:
    assert _prefill_step_size(has_recurrent_state=False) == 4096


def test_recurrent_prefill_uses_bounded_chunks() -> None:
    assert _prefill_step_size(has_recurrent_state=True) == 512
