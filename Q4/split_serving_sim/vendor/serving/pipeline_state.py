"""CPU-only causal gate and conservative KV admission for opt-in pipelining."""
import threading
import time


class CausalGate:
    def __init__(self, timeout=30):
        self.condition = threading.Condition()
        self.positions = {}
        self.failure = None
        self.timeout = timeout

    def run(self, items, execute):
        deadline = time.monotonic()+self.timeout
        with self.condition:
            while True:
                if self.failure is not None:
                    raise RuntimeError(f"Cloud pipeline unhealthy: {self.failure}")
                expected = [self.positions.get(item["request_id"],0) for item in items]
                if any(item["position"] < pos for item,pos in zip(items,expected)):
                    raise ValueError("Replayed pipeline position")
                if all(item["position"] == pos for item,pos in zip(items,expected)):
                    break
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Missing predecessor chunk")
                # Releases the condition lock; an earlier chunk can execute.
                self.condition.wait(remaining)
            try:
                result = execute()
            except BaseException as error:
                self.failure = str(error)
                self.condition.notify_all()
                raise
            for item in items:
                self.positions[item["request_id"]] = item["position"]+item["query_len"]
            self.condition.notify_all()
            return result

    def release(self, ids, execute):
        with self.condition:
            result = execute()
            for rid in ids:
                self.positions.pop(rid,None)
            self.condition.notify_all()
            return result


class KVAdmission:
    """Reserve each admitted request's maximum blocks until final release."""
    def __init__(self, capacity, block_size=16):
        self.capacity, self.block_size = capacity, block_size
        self.reservations = {}

    def needed(self, job):
        return (len(job.ids)+job.limit-1+self.block_size-1)//self.block_size

    def admit(self, job):
        if job.id in self.reservations:
            raise ValueError("Request already admitted")
        blocks = self.needed(job)
        if blocks > self.capacity:
            raise ValueError("Request exceeds the KV capacity")
        if sum(self.reservations.values())+blocks > self.capacity:
            return False
        self.reservations[job.id] = blocks
        return True

    def release(self, job):
        self.reservations.pop(job.id,None)
