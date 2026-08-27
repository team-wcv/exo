# pyright: reportPrivateUsage=false

from typing import cast

import mlx.core as mx
import mlx.nn as nn

from exo.worker.engines.mlx.auto_parallel import (
    Qwen4ExpShardingStrategy,
    _is_qwen4_exp_model,
    _LayerCallable,
    _patch_qwen4_exp_pipeline_state,
    _Qwen4ExpHeadShardedEmbedding,
    _Qwen4ExpPle,
)


class _FakeGroup:
    def __init__(self, rank: int, size: int = 2):
        self._rank = rank
        self._size = size

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size


class _ModelArgs:
    def __init__(self, model_type: str):
        self.model_type = model_type


class _MetadataModel(nn.Module):
    def __init__(
        self, model_type: str | None = None, args_model_type: str | None = None
    ):
        super().__init__()
        self.model_type = model_type
        self.args = _ModelArgs(args_model_type) if args_model_type else None


class _CacheModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_type = "qwen4_exp"

    def make_cache(self) -> list[object]:
        return ["cache-0", "cache-1", "cache-2", "cache-3"]


class _PipelineInnerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.ple_layers = [1]


class _PipelineLayer(nn.Module):
    def __init__(self, ple: object | None):
        super().__init__()
        self.ple = ple

    def __call__(self, value: mx.array, *args: object, **kwargs: object) -> mx.array:
        return value


class _EmbeddingTable(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rows = 8
        self.n_shards = 8
        for index in range(self.n_shards):
            setattr(self, f"shard_{index}", nn.Module())


class _NGramEmbedding:
    def __init__(self, table: _EmbeddingTable):
        self.ngram_heads = 4
        self.head_vocab_sizes = [16, 16, 16, 16]
        self.ngram_embedding: nn.Module = table


class _Ple:
    def __init__(self, table: _EmbeddingTable):
        self.ple_embedding = _NGramEmbedding(table)
        self.key_proj: nn.Module = nn.Linear(1, 1, bias=False)
        self.value_proj: nn.Module = nn.Linear(1, 1, bias=False)


def _linear_identity(module: nn.Module) -> nn.Linear:
    return cast(nn.Linear, module)


def _in_place_identity(module: nn.Module) -> None:
    del module


def _strategy(rank: int) -> Qwen4ExpShardingStrategy:
    group = cast(mx.distributed.Group, cast(object, _FakeGroup(rank)))
    return Qwen4ExpShardingStrategy(
        group,
        _linear_identity,
        _linear_identity,
        _in_place_identity,
        _in_place_identity,
    )


def test_qwen4_exp_detection_uses_model_metadata() -> None:
    assert _is_qwen4_exp_model(_MetadataModel(model_type="qwen4_exp"))
    assert _is_qwen4_exp_model(_MetadataModel(args_model_type="qwen4_exp"))
    assert not _is_qwen4_exp_model(_MetadataModel(model_type="qwen3_5"))


def test_qwen4_exp_pipeline_cache_tracks_the_live_slice() -> None:
    model = _CacheModel()
    layers = [
        _PipelineLayer(ple=None),
        _PipelineLayer(ple=object()),
    ]
    inner = _PipelineInnerModel()

    _patch_qwen4_exp_pipeline_state(
        model,
        inner,
        cast(list[_LayerCallable], cast(object, layers)),
        start_layer=2,
        end_layer=4,
    )

    assert model.make_cache() == ["cache-2", "cache-3"]
    assert inner.ple_layers == [1]


def test_qwen4_exp_ple_keeps_only_the_local_head_shards() -> None:
    ple = _Ple(_EmbeddingTable())

    _strategy(rank=0)._shard_ple(cast(_Qwen4ExpPle, cast(object, ple)))

    wrapped = cast(
        _Qwen4ExpHeadShardedEmbedding,
        cast(object, ple.ple_embedding.ngram_embedding),
    )
    assert wrapped.head_start == 0
    assert wrapped.head_end == 2
    assert all(f"shard_{index}" in wrapped.embedding for index in range(4))
    assert all(f"shard_{index}" not in wrapped.embedding for index in range(4, 8))


def test_qwen4_exp_ple_assigns_the_other_heads_to_rank_one() -> None:
    ple = _Ple(_EmbeddingTable())

    _strategy(rank=1)._shard_ple(cast(_Qwen4ExpPle, cast(object, ple)))

    wrapped = cast(
        _Qwen4ExpHeadShardedEmbedding,
        cast(object, ple.ple_embedding.ngram_embedding),
    )
    assert wrapped.head_start == 2
    assert wrapped.head_end == 4
    assert all(f"shard_{index}" not in wrapped.embedding for index in range(4))
    assert all(f"shard_{index}" in wrapped.embedding for index in range(4, 8))
