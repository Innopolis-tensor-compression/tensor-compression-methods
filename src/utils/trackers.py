import time
from abc import ABC, abstractmethod

import tensorflow as tf
import torch
from memory_profiler import memory_usage


class IGPUMemoryTracker(ABC):
    @abstractmethod
    def get_usage(self) -> dict:
        pass

    @abstractmethod
    def start(self) -> None:
        pass

    @abstractmethod
    def stop(self) -> None:
        pass


class GPUTorchMemoryTracker(IGPUMemoryTracker):
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


class GPUTensorflowMemoryTracker(IGPUMemoryTracker):
    def __init__(self, tf_devices: list[str]):
        self.devices = tf_devices

        self.gpu_allocated_memory_before = 0.0
        self.gpu_allocated_memory_after = 0.0
        self.gpu_cached_memory_before = 0.0
        self.gpu_cached_memory_after = 0.0

    def start(self) -> None:
        devices_memory_info = list(map(tf.config.experimental.get_memory_info, self.devices))

        self.gpu_allocated_memory_before = sum(device_memory_info["current"] / (1024**2) for device_memory_info in devices_memory_info)
        self.gpu_cached_memory_before = sum(device_memory_info["peak"] / (1024**2) for device_memory_info in devices_memory_info)

    def stop(self) -> None:
        devices_memory_info = list(map(tf.config.experimental.get_memory_info, self.devices))

        self.gpu_allocated_memory_after = sum(device_memory_info["current"] / (1024**2) for device_memory_info in devices_memory_info)
        self.gpu_cached_memory_after = sum(device_memory_info["peak"] / (1024**2) for device_memory_info in devices_memory_info)

    def get_usage(self) -> dict:
        gpu_allocated_memory_used = self.gpu_allocated_memory_after - self.gpu_allocated_memory_before
        gpu_cached_memory_used = self.gpu_cached_memory_after - self.gpu_cached_memory_before
        return {"gpu_allocated_memory_used_mb": gpu_allocated_memory_used, "gpu_cached_memory_used_mb": gpu_cached_memory_used}


class IRAMMemoryTracker(ABC):
    @abstractmethod
    def run_tracker(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def get_usage(self) -> dict:
        pass


class RAMMemoryTracker(IRAMMemoryTracker):
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
