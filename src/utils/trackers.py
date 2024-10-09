import time
from abc import ABC, abstractmethod

import torch
from memory_profiler import memory_usage


class IMemoryTracker(ABC):
    @abstractmethod
    def get_usage(self) -> dict:
        pass

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def stop(self):
        pass


class GPUTorchMemoryTracker(IMemoryTracker):
    def __init__(self):
        self.gpu_allocated_memory_before = 0.0
        self.gpu_allocated_memory_after = 0.0
        self.gpu_cached_memory_before = 0.0
        self.gpu_cached_memory_after = 0.0

    def start(self) -> None:
        self.gpu_allocated_memory_before = torch.cuda.memory_allocated()
        self.gpu_cached_memory_before = torch.cuda.memory_reserved()

    def stop(self) -> None:
        self.gpu_allocated_memory_after = torch.cuda.memory_allocated()
        self.gpu_cached_memory_after = torch.cuda.memory_reserved()

    def get_usage(self) -> dict:
        gpu_allocated_memory_used = self.gpu_allocated_memory_after - self.gpu_allocated_memory_before
        gpu_cached_memory_used = self.gpu_cached_memory_after - self.gpu_cached_memory_before
        return {"gpu_allocated_memory_used_mb": gpu_allocated_memory_used / 1024**2, "gpu_cached_memory_used_mb": gpu_cached_memory_used / 1024**2}


class RAMMemoryTracker:
    def __init__(self):
        self.ram_allocated_memory = 0.0

    def run_tracker(self, func, *args, **kwargs) -> None:
        self.ram_allocated_memory = memory_usage((func, args, kwargs))[-1]

    def get_usage(self) -> dict:
        return {
            "ram_mem_used_mb": self.ram_allocated_memory,
        }


class ITimeTracker(ABC):
    @abstractmethod
    def start(self) -> None:
        pass

    @abstractmethod
    def stop(self) -> None:
        pass

    @abstractmethod
    def get_duration(self) -> float:
        pass


class TimeTracker(ITimeTracker):
    def __init__(self):
        self.start_time = 0.0
        self.end_time = 0.0

    def start(self) -> None:
        self.start_time = time.time()

    def stop(self) -> None:
        self.end_time = time.time()

    def get_duration(self) -> float:
        return self.end_time - self.start_time
