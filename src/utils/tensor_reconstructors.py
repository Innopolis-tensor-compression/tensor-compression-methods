from abc import ABC, abstractmethod

import t3f
import tensorly as tl


class ITensorReconstructor(ABC):
    @abstractmethod
    def calculate(self, *args, **kwargs) -> float:
        pass


class TensorLyTuckerTensorReconstructor(ITensorReconstructor):
    def calculate(self, method_result) -> float:
        core, factors = method_result
        return tl.tucker_tensor.tucker_to_tensor((core, factors))


class TensorLyTensorTrainTensorReconstructor(ITensorReconstructor):
    def calculate(self, method_result) -> float:
        tt_factors = method_result
        return tl.tt_to_tensor(tt_factors)


class T3FTensorTrainTensorReconstructor(ITensorReconstructor):
    def calculate(self, method_result) -> float:
        tt_factors = method_result
        return t3f.full(tt_factors)


class TensorReconstructorFactory:
    @staticmethod
    def create_reconstructor(library_method_name: str) -> ITensorReconstructor:
        reconstructors = {
            "TensorLy_Tucker": TensorLyTuckerTensorReconstructor(),
            "TensorLy_TensorTrain": TensorLyTensorTrainTensorReconstructor(),
            "T3F_TensorTrain": T3FTensorTrainTensorReconstructor(),
        }

        if library_method_name in reconstructors:
            return reconstructors[library_method_name]
        error_message = f"Unknown library method name: {library_method_name}"
        raise ValueError(error_message)
