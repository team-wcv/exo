import mlx.core as mx
from mlx_lm.models.cache import KVCache

from exo.worker.engines.mlx.cache import trim_trimmable_cache_entry


class _IndexerCache:
    def __init__(self, length: int) -> None:
        self.state = mx.zeros((1, length, 4))


class _KVCacheWithIndexer(KVCache):
    def __init__(self, length: int) -> None:
        super().__init__()
        self.keys = mx.zeros((1, 1, length, 4))
        self.values = mx.zeros((1, 1, length, 4))
        self.offset = length
        self.indexer = _IndexerCache(length)


def test_trim_keeps_auxiliary_indexer_aligned() -> None:
    cache = _KVCacheWithIndexer(length=10)

    assert trim_trimmable_cache_entry(cache, 3) == 3
    assert cache.offset == 7
    assert cache.indexer.state.shape[1] == 7


def test_trim_without_auxiliary_indexer_uses_native_behavior() -> None:
    cache = KVCache()
    cache.keys = mx.zeros((1, 1, 5, 4))
    cache.values = mx.zeros((1, 1, 5, 4))
    cache.offset = 5

    assert trim_trimmable_cache_entry(cache, 2) == 2
    assert cache.offset == 3
