from __future__ import annotations

from collections import defaultdict

from .core import Phase, Stage, WorkItem


class ExecutionDAG:
    """Dependency graph for prefill chunks and lazily-created decode tokens."""

    def __init__(self, pipeline_degrees: dict[Stage, int] | None = None) -> None:
        self._next_id = 0
        self.items: dict[int, WorkItem] = {}
        self.remaining_dependencies: dict[int, int] = {}
        self.successors: dict[int, list[int]] = defaultdict(list)
        self.completed: set[int] = set()
        self.last_kv_transfer: dict[int, int] = {}
        self.final_prefill_tail: dict[int, int] = {}
        self.pipeline_degrees = pipeline_degrees or {}

    def _add(
        self,
        *,
        request_id: int,
        phase: Phase,
        stage: Stage,
        token_start: int,
        token_count: int,
        context_tokens: int,
        dependencies: tuple[int, ...] = (),
        chunk_index: int | None = None,
        iteration: int | None = None,
        pipeline_rank: int = 0,
        produces_logits: bool = False,
    ) -> int:
        item_id = self._next_id
        self._next_id += 1
        item = WorkItem(
            id=item_id,
            request_id=request_id,
            phase=phase,
            stage=stage,
            token_start=token_start,
            token_count=token_count,
            context_tokens=context_tokens,
            chunk_index=chunk_index,
            iteration=iteration,
            pipeline_rank=pipeline_rank,
            produces_logits=produces_logits,
            dependencies=dependencies,
        )
        self.items[item_id] = item
        remaining = sum(dependency not in self.completed for dependency in dependencies)
        self.remaining_dependencies[item_id] = remaining
        for dependency in dependencies:
            self.successors[dependency].append(item_id)
        return item_id

    def _add_pipeline(
        self,
        *,
        request_id: int,
        phase: Phase,
        stage: Stage,
        token_start: int,
        token_count: int,
        context_tokens: int,
        first_dependencies: tuple[int, ...] = (),
        previous_items: tuple[int, ...] = (),
        chunk_index: int | None = None,
        iteration: int | None = None,
        produces_logits: bool = False,
    ) -> list[int]:
        degree = self.pipeline_degrees.get(stage, 1)
        if previous_items and len(previous_items) != degree:
            raise ValueError(f"pipeline history does not match PP degree for {stage}")
        created: list[int] = []
        for pipeline_rank in range(degree):
            dependencies: list[int] = []
            if pipeline_rank == 0:
                dependencies.extend(first_dependencies)
            else:
                dependencies.append(created[-1])
            if previous_items:
                dependencies.append(previous_items[pipeline_rank])
            created.append(
                self._add(
                    request_id=request_id,
                    phase=phase,
                    stage=stage,
                    token_start=token_start,
                    token_count=token_count,
                    context_tokens=context_tokens,
                    dependencies=tuple(dependencies),
                    chunk_index=chunk_index,
                    iteration=iteration,
                    pipeline_rank=pipeline_rank,
                    produces_logits=produces_logits
                    and pipeline_rank == degree - 1,
                )
            )
        return created

    def add_prefill(
        self, request_id: int, input_tokens: int, chunk_size: int,
        ready_time: float, token_offset: int = 0,
    ) -> list[WorkItem]:
        previous_front: tuple[int, ...] = ()
        previous_cloud: tuple[int, ...] = ()
        previous_tail: tuple[int, ...] = ()
        chunk_index = 0
        token_start = token_offset
        created: list[int] = []

        while token_start < input_tokens:
            token_count = min(chunk_size, input_tokens - token_start)
            context = token_start + token_count
            front = self._add_pipeline(
                request_id=request_id,
                phase=Phase.PREFILL,
                stage=Stage.EDGE_FRONT,
                token_start=token_start,
                token_count=token_count,
                context_tokens=context,
                chunk_index=chunk_index,
                previous_items=previous_front,
            )
            up = self._add(
                request_id=request_id,
                phase=Phase.PREFILL,
                stage=Stage.WAN_UP,
                token_start=token_start,
                token_count=token_count,
                context_tokens=context,
                chunk_index=chunk_index,
                dependencies=(front[-1],),
            )
            cloud = self._add_pipeline(
                request_id=request_id,
                phase=Phase.PREFILL,
                stage=Stage.CLOUD_MIDDLE,
                token_start=token_start,
                token_count=token_count,
                context_tokens=context,
                chunk_index=chunk_index,
                first_dependencies=(up,),
                previous_items=previous_cloud,
            )
            down = self._add(
                request_id=request_id,
                phase=Phase.PREFILL,
                stage=Stage.WAN_DOWN,
                token_start=token_start,
                token_count=token_count,
                context_tokens=context,
                chunk_index=chunk_index,
                dependencies=(cloud[-1],),
            )
            tail = self._add_pipeline(
                request_id=request_id,
                phase=Phase.PREFILL,
                stage=Stage.EDGE_TAIL,
                token_start=token_start,
                token_count=token_count,
                context_tokens=context,
                chunk_index=chunk_index,
                produces_logits=token_start + token_count == input_tokens,
                first_dependencies=(down,),
                previous_items=previous_tail,
            )
            created.extend((*front, up, *cloud, down, *tail))
            previous_front = tuple(front)
            previous_cloud = tuple(cloud)
            previous_tail = tuple(tail)
            token_start += token_count
            chunk_index += 1

        if not previous_tail:
            raise ValueError("prefill must contain at least one token")
        self.final_prefill_tail[request_id] = previous_tail[-1]
        return self._newly_ready(created, ready_time)

    def add_decode(
        self, request_id: int, iteration: int, context_tokens: int, ready_time: float, token_start: int | None = None
    ) -> list[WorkItem]:
        position = context_tokens if token_start is None else token_start
        front = self._add_pipeline(
            request_id=request_id,
            phase=Phase.DECODE,
            stage=Stage.EDGE_FRONT,
            token_start=position,
            token_count=1,
            context_tokens=context_tokens,
            iteration=iteration,
        )
        up = self._add(
            request_id=request_id,
            phase=Phase.DECODE,
            stage=Stage.WAN_UP,
            token_start=position,
            token_count=1,
            context_tokens=context_tokens,
            iteration=iteration,
            dependencies=(front[-1],),
        )
        cloud = self._add_pipeline(
            request_id=request_id,
            phase=Phase.DECODE,
            stage=Stage.CLOUD_MIDDLE,
            token_start=position,
            token_count=1,
            context_tokens=context_tokens,
            iteration=iteration,
            first_dependencies=(up,),
        )
        down = self._add(
            request_id=request_id,
            phase=Phase.DECODE,
            stage=Stage.WAN_DOWN,
            token_start=position,
            token_count=1,
            context_tokens=context_tokens,
            iteration=iteration,
            dependencies=(cloud[-1],),
        )
        tail = self._add_pipeline(
            request_id=request_id,
            phase=Phase.DECODE,
            stage=Stage.EDGE_TAIL,
            token_start=position,
            token_count=1,
            context_tokens=context_tokens,
            iteration=iteration,
            produces_logits=True,
            first_dependencies=(down,),
        )
        return self._newly_ready([*front, up, *cloud, down, *tail], ready_time)

    def add_pd_kv_transfer(self, request_id: int, token_count: int, ready_time: float, dependency: int, token_start: int = 0) -> list[WorkItem]:
        dependencies = (dependency,) + ((self.last_kv_transfer[request_id],)
                                        if request_id in self.last_kv_transfer else ())
        common = dict(request_id=request_id, phase=Phase.PREFILL,
                      token_start=token_start, token_count=token_count,
                      context_tokens=token_start + token_count)
        start = self._add(**common, stage=Stage.PD_KV_START, dependencies=dependencies)
        transfer = self._add(**common, stage=Stage.PD_KV_TRANSFER, dependencies=(start,))
        commit = self._add(**common, stage=Stage.PD_KV_COMMIT, dependencies=(transfer,))
        self.last_kv_transfer[request_id] = commit
        return self._newly_ready([start, transfer, commit], ready_time)

    def _newly_ready(self, item_ids: list[int], ready_time: float) -> list[WorkItem]:
        ready: list[WorkItem] = []
        for item_id in item_ids:
            if self.remaining_dependencies[item_id] == 0:
                item = self.items[item_id].with_ready_time(ready_time)
                self.items[item_id] = item
                ready.append(item)
        return ready

    def mark_complete(self, item_id: int, completion_time: float) -> list[WorkItem]:
        if item_id in self.completed:
            raise RuntimeError(f"work item completed twice: {item_id}")
        self.completed.add(item_id)
        ready: list[WorkItem] = []
        for successor_id in self.successors[item_id]:
            self.remaining_dependencies[successor_id] -= 1
            if self.remaining_dependencies[successor_id] < 0:
                raise RuntimeError(f"negative dependency count: {successor_id}")
            if self.remaining_dependencies[successor_id] == 0:
                item = self.items[successor_id].with_ready_time(completion_time)
                self.items[successor_id] = item
                ready.append(item)
        return ready

    def is_final_prefill(self, item: WorkItem) -> bool:
        return self.final_prefill_tail.get(item.request_id) == item.id
