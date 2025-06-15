import gc

import tensorflow as tf
import torch

from src.utils.metrics_calculators import CompressionRationFactory, FrobeniusErrorFactory
from src.utils.tensor_reconstructors import TensorReconstructorFactory
from src.utils.trackers import IGPUMemoryTracker, IRAMMemoryTracker, ITimeTracker


class MethodRunner:
    def __init__(
        self,
        func,
        method_input_tensor,
        library_method_name: str,
        backend_name: str,
        gpu_memory_tracker: IGPUMemoryTracker,
        ram_memory_tracker: IRAMMemoryTracker,
        time_tracker: ITimeTracker,
    ):
        self.func = func
        self.method_input_tensor = method_input_tensor

        self.result = None
        self.reconstructed_tensor = None

        self.library_method_name = library_method_name
        self.backend_name = backend_name

        self.gpu_memory_tracker = gpu_memory_tracker
        self.ram_memory_tracker = ram_memory_tracker
        self.time_tracker = time_tracker

    def run(self, *args, **kwargs) -> None:
        # self.result = None
        # self.reconstructed_tensor = None
        if self.backend_name == "pytorch":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.reset_peak_memory_stats()
        elif self.backend_name == "tensorflow":
            tf.keras.backend.clear_session()
        gc.collect()

        self.gpu_memory_tracker.start()
        self.time_tracker.start()

        self.result = self.func(*args, **kwargs)

        if self.backend_name == "pytorch":
            torch.cuda.synchronize()
        elif self.backend_name == "tensorflow":
            tf.keras.backend.clear_session()
        self.time_tracker.stop()
        self.gpu_memory_tracker.stop()

        self.ram_memory_tracker.run_tracker(self.func, *args, **kwargs)

        self.reconstructed_tensor = self.calculate_reconstructed_tensor(library_method_name=self.library_method_name)

        if self.backend_name == "pytorch":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.reset_peak_memory_stats()
        elif self.backend_name == "tensorflow":
            tf.keras.backend.clear_session()
        gc.collect()

    def calculate_reconstructed_tensor(self, library_method_name: str):
        if self.result is None:
            return None
        return TensorReconstructorFactory.create_reconstructor(library_method_name=library_method_name).calculate(
            method_result=self.result
        )

    def calculate_frobenius_error(self, library_method_name: str) -> float | None:
        if self.result is None or self.reconstructed_tensor is None:
            return None
        return FrobeniusErrorFactory.create_calculators(library_method_name=library_method_name).calculate(
            original_tensor=self.method_input_tensor, reconstructed_tensor=self.reconstructed_tensor
        )

    def calculate_compression_ratio(self, library_method_name: str) -> float | None:
        if self.reconstructed_tensor is None:
            return None
        return CompressionRationFactory.create_calculators(library_method_name=library_method_name).calculate(
            original_tensor=self.method_input_tensor, method_result=self.result
        )

    def get_metrics(self, library_method_name: str) -> dict:
        metrics = {}
        metrics.update(self.gpu_memory_tracker.get_usage())
        metrics.update(self.ram_memory_tracker.get_usage())
        metrics["duration"] = self.time_tracker.get_duration()
        metrics["frobenius_error"] = self.calculate_frobenius_error(library_method_name=library_method_name)
        metrics["compression_ratio"] = self.calculate_compression_ratio(library_method_name=library_method_name)

        return metrics
