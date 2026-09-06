"""One host submission lane and one ordered GPU stream per resource.

Submission can run ahead of GPU completion. GPU dependencies constrain execution,
not host submission. This is a critical-rank approximation for a TP worker group;
it does not inject measured idle gaps or another rank-arrival penalty.
"""
from dataclasses import dataclass
from .config import HostSubmissionConfig
from .core import SubOperation


@dataclass(frozen=True)
class SubmissionSchedule:
    cpu_intervals: tuple[tuple[float, float], ...]
    gpu_intervals: tuple[tuple[float, float], ...]
    end_time: float
    cpu_end: float
    cpu_busy_s: float
    gpu_busy_s: float
    exposed_delay_s: float


def schedule_submissions(operations: tuple[SubOperation, ...], start: float,
                         cpu_available: float, config: HostSubmissionConfig) -> SubmissionSchedule:
    costs = dict(config.submit_us_by_type)
    cpu = max(start, cpu_available)
    gpu = start
    completed: dict[str, float] = {}
    cpu_intervals = []
    gpu_intervals = []
    cpu_busy = gpu_busy = 0.0
    for operation in operations:
        duration = costs.get(operation.profile_type, config.submit_us) * 1e-6
        cpu_intervals.append((cpu, cpu + duration))
        cpu += duration
        cpu_busy += duration
        dependency_end = max((completed[d] for d in operation.dependencies), default=start)
        gpu_start = max(cpu, gpu, dependency_end)
        gpu = gpu_start + operation.duration_s
        gpu_intervals.append((gpu_start, gpu))
        gpu_busy += operation.duration_s
        completed[operation.name] = gpu
    return SubmissionSchedule(tuple(cpu_intervals), tuple(gpu_intervals), gpu,
                              cpu, cpu_busy, gpu_busy, max(0.0, gpu - start - gpu_busy))
