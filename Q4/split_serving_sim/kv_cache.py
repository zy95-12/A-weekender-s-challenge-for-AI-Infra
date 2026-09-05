from __future__ import annotations

import math
from dataclasses import dataclass

from .config import KVCacheConfig


@dataclass(frozen=True)
class KVAdmission:
    request_id: int
    allocated_blocks: int
    prefix_hit_blocks: int
    evicted_prefix_blocks: int


class PagedKVCache:
    """Logical block pool for the bottleneck model instance.

    Shared prefix blocks occupy the pool once. Request allocations contain only
    private blocks, so cache hits directly reduce admission pressure.
    """

    def __init__(self, config: KVCacheConfig):
        self.config = config
        self.allocations: dict[int, int] = {}
        self.shared_prefixes: dict[str, int] = {}
        self.prefix_last_used: dict[str, int] = {}
        self.request_shared_prefix: dict[int, str] = {}
        self._clock = 0

    @property
    def used_blocks(self) -> int:
        return sum(self.allocations.values()) + sum(self.shared_prefixes.values())

    @property
    def free_blocks(self) -> int:
        return self.config.num_blocks - self.used_blocks

    @property
    def watermark_blocks(self) -> int:
        return math.ceil(self.config.num_blocks * self.config.watermark)

    def blocks_for_tokens(self, tokens: int) -> int:
        return math.ceil(tokens / self.config.block_size_tokens)

    def cached_tokens(self, prefix_id: str | None, prefix_tokens: int) -> int:
        if not self.config.prefix_caching or prefix_id not in self.shared_prefixes:
            return 0
        return min(
            prefix_tokens,
            self.shared_prefixes[prefix_id] * self.config.block_size_tokens,
        )

    def reserve(
        self,
        request_id: int,
        total_tokens: int,
        *,
        prefix_id: str | None = None,
        prefix_tokens: int = 0,
        is_new_request: bool = True,
    ) -> KVAdmission | None:
        if request_id in self.allocations:
            return KVAdmission(request_id, 0, 0, 0)
        self._clock += 1
        hit_blocks = 0
        if self.config.prefix_caching and prefix_id in self.shared_prefixes:
            hit_blocks = min(
                self.shared_prefixes[prefix_id], self.blocks_for_tokens(prefix_tokens)
            )
            self.prefix_last_used[prefix_id] = self._clock
        needed = max(self.blocks_for_tokens(total_tokens) - hit_blocks, 0)
        reserve = self.watermark_blocks if is_new_request else 0
        evicted = self._evict_prefixes_until(needed + reserve, exclude=prefix_id)
        if self.free_blocks < needed + reserve:
            return None
        self.allocations[request_id] = needed
        if hit_blocks and prefix_id is not None:
            self.request_shared_prefix[request_id] = prefix_id
        return KVAdmission(request_id, needed, hit_blocks, evicted)

    def free_request(self, request_id: int) -> int:
        self.request_shared_prefix.pop(request_id, None)
        return self.allocations.pop(request_id, 0)

    def publish_prefix(
        self, request_id: int, prefix_id: str | None, prefix_tokens: int
    ) -> int:
        if (
            not self.config.prefix_caching
            or prefix_id is None
            or prefix_tokens <= 0
            or prefix_id in self.shared_prefixes
        ):
            return 0
        blocks = min(
            self.blocks_for_tokens(prefix_tokens), self.allocations.get(request_id, 0)
        )
        if blocks <= 0:
            return 0
        self.allocations[request_id] -= blocks
        self.shared_prefixes[prefix_id] = blocks
        self.request_shared_prefix[request_id] = prefix_id
        self._clock += 1
        self.prefix_last_used[prefix_id] = self._clock
        return blocks

    def _evict_prefixes_until(self, required_free: int, exclude: str | None) -> int:
        evicted = 0
        while self.free_blocks < required_free:
            pinned = set(self.request_shared_prefix.values())
            candidates = [
                key
                for key in self.shared_prefixes
                if key != exclude and key not in pinned
            ]
            if not candidates:
                break
            victim = min(candidates, key=lambda key: self.prefix_last_used[key])
            evicted += self.shared_prefixes.pop(victim)
            self.prefix_last_used.pop(victim, None)
        return evicted

