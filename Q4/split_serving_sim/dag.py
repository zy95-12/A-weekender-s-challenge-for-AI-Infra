from __future__ import annotations

from collections import defaultdict

from .core import Phase, Stage, WorkItem


class ExecutionDAG:
    """Dependency graph for prefill chunks and lazily-created decode tokens."""

    def __init__(self) -> None:
        self._next_id = 0
        self.items: dict[int, WorkItem] = {}
        self.remaining_dependencies: dict[int, int] = {}
        self.successors: dict[int, list[int]] = defaultdict(list)
        self.completed: set[int] = set()
        self.final_prefill_tail: dict[int, int] = {}

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
            produces_logits=produces_logits,
            dependencies=dependencies,
        )
        self.items[item_id] = item
        remaining = sum(dependency not in self.completed for dependency in dependencies)
        self.remaining_dependencies[item_id] = remaining
        for dependency in dependencies:
            self.successors[dependency].append(item_id)
        return item_id

    def add_prefill(
        self, request_id: int, input_tokens: int, chunk_size: int, ready_time: float
    ) -> list[WorkItem]:
        previous_front: int | None = None
        previous_cloud: int | None = None
        previous_tail: int | None = None
        chunk_index = 0
        token_start = 0
        created: list[int] = []

        while token_start < input_tokens:
            token_count = min(chunk_size, input_tokens - token_start)
            context = token_start + token_count
            front_deps = () if previous_front is None else (previous_front,)
            front = self._add(
                request_id=request_id,
                phase=Phase.PREFILL,
                stage=Stage.EDGE_FRONT,
                token_start=token_start,
                token_count=token_count,
                context_tokens=context,
                chunk_index=chunk_index,
                dependencies=front_deps,
            )
            up = self._add(
                request_id=request_id,
                phase=Phase.PREFILL,
                stage=Stage.WAN_UP,
                token_start=token_start,
                token_count=token_count,
                context_tokens=context,
                chunk_index=chunk_index,
                dependencies=(front,),
            )
            cloud_dependencies = (up,) if previous_cloud is None else (up, previous_cloud)
            cloud = self._add(
                request_id=request_id,
                phase=Phase.PREFILL,
                stage=Stage.CLOUD_MIDDLE,
                token_start=token_start,
                token_count=token_count,
                context_tokens=context,
                chunk_index=chunk_index,
                dependencies=cloud_dependencies,
            )
            down = self._add(
                request_id=request_id,
                phase=Phase.PREFILL,
                stage=Stage.WAN_DOWN,
                token_start=token_start,
                token_count=token_count,
                context_tokens=context,
                chunk_index=chunk_index,
                dependencies=(cloud,),
            )
            tail_dependencies = (down,) if previous_tail is None else (down, previous_tail)
            tail = self._add(
                request_id=request_id,
                phase=Phase.PREFILL,
                stage=Stage.EDGE_TAIL,
                token_start=token_start,
                token_count=token_count,
                context_tokens=context,
                chunk_index=chunk_index,
                produces_logits=token_start + token_count == input_tokens,
                dependencies=tail_dependencies,
            )
            created.extend((front, up, cloud, down, tail))
            previous_front, previous_cloud, previous_tail = front, cloud, tail
            token_start += token_count
            chunk_index += 1

        if previous_tail is None:
            raise ValueError("prefill must contain at least one token")
        self.final_prefill_tail[request_id] = previous_tail
        return self._newly_ready(created, ready_time)

    def add_decode(
        self, request_id: int, iteration: int, context_tokens: int, ready_time: float
    ) -> list[WorkItem]:
        front = self._add(
            request_id=request_id,
            phase=Phase.DECODE,
            stage=Stage.EDGE_FRONT,
            token_start=context_tokens,
            token_count=1,
            context_tokens=context_tokens,
            iteration=iteration,
        )
        up = self._add(
            request_id=request_id,
            phase=Phase.DECODE,
            stage=Stage.WAN_UP,
            token_start=context_tokens,
            token_count=1,
            context_tokens=context_tokens,
            iteration=iteration,
            dependencies=(front,),
        )
        cloud = self._add(
            request_id=request_id,
            phase=Phase.DECODE,
            stage=Stage.CLOUD_MIDDLE,
            token_start=context_tokens,
            token_count=1,
            context_tokens=context_tokens,
            iteration=iteration,
            dependencies=(up,),
        )
        down = self._add(
            request_id=request_id,
            phase=Phase.DECODE,
            stage=Stage.WAN_DOWN,
            token_start=context_tokens,
            token_count=1,
            context_tokens=context_tokens,
            iteration=iteration,
            dependencies=(cloud,),
        )
        tail = self._add(
            request_id=request_id,
            phase=Phase.DECODE,
            stage=Stage.EDGE_TAIL,
            token_start=context_tokens,
            token_count=1,
            context_tokens=context_tokens,
            iteration=iteration,
            produces_logits=True,
            dependencies=(down,),
        )
        return self._newly_ready([front, up, cloud, down, tail], ready_time)

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
