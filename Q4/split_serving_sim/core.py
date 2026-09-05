from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Phase(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"


class Stage(str, Enum):
    EDGE_FRONT = "edge_front"
    WAN_UP = "wan_up"
    CLOUD_MIDDLE = "cloud_middle"
    WAN_DOWN = "wan_down"
    EDGE_TAIL = "edge_tail"


GPU_STAGES = {Stage.EDGE_FRONT, Stage.CLOUD_MIDDLE, Stage.EDGE_TAIL}
NETWORK_STAGES = {Stage.WAN_UP, Stage.WAN_DOWN}


@dataclass(frozen=True)
class WorkItem:
    id: int
    request_id: int
    phase: Phase
    stage: Stage
    token_start: int
    token_count: int
    context_tokens: int
    chunk_index: int | None = None
    iteration: int | None = None
    pipeline_rank: int = 0
    produces_logits: bool = False
    dependencies: tuple[int, ...] = ()
    ready_time: float = 0.0

    def with_ready_time(self, ready_time: float) -> "WorkItem":
        return WorkItem(
            id=self.id,
            request_id=self.request_id,
            phase=self.phase,
            stage=self.stage,
            token_start=self.token_start,
            token_count=self.token_count,
            context_tokens=self.context_tokens,
            chunk_index=self.chunk_index,
            iteration=self.iteration,
            pipeline_rank=self.pipeline_rank,
            produces_logits=self.produces_logits,
            dependencies=self.dependencies,
            ready_time=ready_time,
        )


@dataclass(frozen=True)
class OperatorWorkload:
    flops: float
    memory_bytes: float
    communication_bytes: float = 0.0


@dataclass(frozen=True)
class SubOperation:
    name: str
    category: str
    duration_s: float
    input_shape: str
    dependencies: tuple[str, ...] = ()


@dataclass(frozen=True)
class PerformanceEstimate:
    flops: float
    memory_bytes: float
    communication_bytes: float
    compute_time_s: float
    memory_time_s: float
    collective_time_s: float
    overhead_time_s: float
    total_time_s: float
    input_shape: str = ""
    sub_operations: tuple[SubOperation, ...] = ()
