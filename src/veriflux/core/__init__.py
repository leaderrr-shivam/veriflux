"""Veriflux core infrastructure layer.

Re-exports commonly used core components for convenient access::

    from veriflux.core import get_logger, get_config, configure
"""
from __future__ import annotations

from veriflux.core.cache import (
    ContentAddressableCache,
    DiskCache,
    LFUCache,
    LRUCache,
    TieredCache,
)
from veriflux.core.config import VerifluxConfigManager, configure, get_config
from veriflux.core.crypto import (
    constant_time_compare,
    generate_nonce,
    generate_trace_id,
    hash_content,
    hash_text,
    hmac_sign,
    hmac_verify,
)
from veriflux.core.logger import VerifluxLogger, get_logger
from veriflux.core.math_kernels import (
    batch_cosine_similarity,
    cosine_similarity,
    dot_product,
    euclidean_distance,
    l2_normalize,
    log_sum_exp,
    sigmoid,
    softmax,
    sparse_dot,
)
from veriflux.core.memory import ArenaAllocator, BufferPool, MemoryTracker, ObjectPool
from veriflux.core.pool import BatchProcessor, ParallelMapper, WorkerPool
from veriflux.core.registry import Registry
from veriflux.core.serialization import deserialize, from_json, serialize, to_json, to_msgpack_like
from veriflux.core.timer import Stopwatch, Timer, TimingStats, timer

__all__ = [
    # config
    "VerifluxConfigManager",
    "configure",
    "get_config",
    # logger
    "VerifluxLogger",
    "get_logger",
    # cache
    "LRUCache",
    "LFUCache",
    "ContentAddressableCache",
    "DiskCache",
    "TieredCache",
    # crypto
    "hash_content",
    "hmac_sign",
    "hmac_verify",
    "generate_trace_id",
    "generate_nonce",
    "constant_time_compare",
    "hash_text",
    # memory
    "ArenaAllocator",
    "BufferPool",
    "ObjectPool",
    "MemoryTracker",
    # timer
    "Timer",
    "timer",
    "TimingStats",
    "Stopwatch",
    # pool
    "WorkerPool",
    "ParallelMapper",
    "BatchProcessor",
    # registry
    "Registry",
    # math
    "dot_product",
    "cosine_similarity",
    "l2_normalize",
    "euclidean_distance",
    "softmax",
    "sigmoid",
    "log_sum_exp",
    "sparse_dot",
    "batch_cosine_similarity",
    # serialization
    "serialize",
    "deserialize",
    "to_json",
    "from_json",
    "to_msgpack_like",
]
