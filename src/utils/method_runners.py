import gc

import torch
from numba import cuda

from src.utils.trackers import GPUTorchMemoryTracker, ITimeTracker, RAMMemoryTracker


class MethodRunner:
    def __init__(self, func, gpu_memory_tracker: GPUTorchMemoryTracker, ram_memory_tracker: RAMMemoryTracker, time_tracker: ITimeTracker):
        self.result = None
        self.func = func
        self.gpu_memory_tracker = gpu_memory_tracker
        self.ram_memory_tracker = ram_memory_tracker
        self.time_tracker = time_tracker

    def run(self, *args, **kwargs) -> None:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        cuda.get_current_device().reset()
        gc.collect()

        self.gpu_memory_tracker.start()
        self.time_tracker.start()

        self.result = self.func(*args, **kwargs)

        torch.cuda.synchronize()
        self.time_tracker.stop()
        self.gpu_memory_tracker.stop()

        self.ram_memory_tracker.run_tracker(self.func, *args, **kwargs)

    def get_metrics(self) -> dict:
        metrics = {}
        metrics.update(self.gpu_memory_tracker.get_usage())
        metrics.update(self.ram_memory_tracker.get_usage())
        metrics["duration"] = self.time_tracker.get_duration()
        return metrics
