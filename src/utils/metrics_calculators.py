from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import tensorly as tl


class IMetricCalculator(ABC):
    @classmethod
    def get_tensors_size(cls, *args) -> float:
        total = 0
        for arg in args:
            partial = 1
            for d in arg.shape:
                partial *= d
            total += partial
        return total

    @abstractmethod
    def calculate(self, *args, **kwargs) -> float:
        pass


class FrobeniusErrorTensorLyCalculator(IMetricCalculator):
    @classmethod
    def calculate(cls, original_tensor, reconstructed_tensor) -> float:
        error = tl.norm(reconstructed_tensor - original_tensor) / tl.norm(original_tensor)
        return 100.0 * error.item()


class CompressionRatioCalculator(IMetricCalculator):
    @classmethod
    def calculate(cls, original_tensor, core, factors) -> float:
        original_size = IMetricCalculator.get_tensors_size(original_tensor)
        compressed_size = IMetricCalculator.get_tensors_size(core, *factors)
        return 100.0 * compressed_size / original_size


class MetricCalculatorFactory:
    @staticmethod
    def create_calculators(library_method_name: str) -> Any:
        if library_method_name == "TensorLy_Tucker":
            return FrobeniusErrorTensorLyCalculator(), CompressionRatioCalculator()
        error_message = f"Неизвестный метод: {library_method_name}"
        raise ValueError(error_message)


def compute_stats(data):
    data = np.array(data)

    if len(data) > 1:
        lower_bound, upper_bound = np.percentile(data, [5, 95])
        filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
    else:
        filtered_data = data

    return {
        "mean": np.mean(filtered_data),
        "min": np.min(filtered_data),
        "max": np.max(filtered_data),
    }
