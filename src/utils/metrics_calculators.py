from abc import ABC, abstractmethod

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


class FrobeniusErrorCalculator(IMetricCalculator):
    @classmethod
    def calculate(cls, original_tensor, reconstructed_tensor) -> float:
        error = tl.norm(reconstructed_tensor - original_tensor) / tl.norm(original_tensor)
        return 100.0 * error.item()


class FrobeniusErrorFactory:
    @staticmethod
    def create_calculators(library_method_name: str) -> IMetricCalculator:
        metrics_calculators = {
            "TensorLy_Tucker": FrobeniusErrorCalculator(),
            "TensorLy_TensorTrain": FrobeniusErrorCalculator(),
            "T3F_TensorTrain": FrobeniusErrorCalculator(),
        }

        if library_method_name in metrics_calculators:
            return metrics_calculators[library_method_name]
        raise ValueError(f"Unknown library method name: {library_method_name}")


class CompressionRatioTensorLyTuckerCalculator(IMetricCalculator):
    @classmethod
    def calculate(cls, original_tensor, method_result) -> float:
        core, factors = method_result

        original_size = IMetricCalculator.get_tensors_size(original_tensor)
        compressed_size = IMetricCalculator.get_tensors_size(core, *factors)
        return 100.0 * compressed_size / original_size


class CompressionRatioTensorLyTensorTrainCalculator(IMetricCalculator):
    @classmethod
    def calculate(cls, original_tensor, method_result) -> float:
        tt_factors = method_result

        original_size = IMetricCalculator.get_tensors_size(original_tensor)
        compressed_size = IMetricCalculator.get_tensors_size(*tt_factors)
        return 100.0 * compressed_size / original_size


class CompressionRatioT3FTensorTrainCalculator(IMetricCalculator):
    @classmethod
    def calculate(cls, original_tensor, method_result) -> float:
        tt_factors = method_result

        original_size = IMetricCalculator.get_tensors_size(original_tensor)
        compressed_size = IMetricCalculator.get_tensors_size(*tt_factors.tt_cores)
        return 100.0 * compressed_size / original_size


class CompressionRationFactory:
    @staticmethod
    def create_calculators(library_method_name: str) -> IMetricCalculator:
        metrics_calculators = {
            "TensorLy_Tucker": CompressionRatioTensorLyTuckerCalculator(),
            "TensorLy_TensorTrain": CompressionRatioTensorLyTensorTrainCalculator(),
            "T3F_TensorTrain": CompressionRatioT3FTensorTrainCalculator(),
        }

        if library_method_name in metrics_calculators:
            return metrics_calculators[library_method_name]
        raise ValueError(f"Unknown library method name: {library_method_name}")


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
