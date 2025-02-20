from collections.abc import Callable
from functools import partial

import numpy as np
import tensorly as tl
from scipy.optimize import OptimizeResult, differential_evolution

from src.utils.benchmark.calculate_bounds import calculate_tt_bounds
from src.utils.metrics_calculators import (
    CompressionRatioTensorLyTensorTrainCalculator,
)


def loss_function_tensor_train(
    rank: list,
    tensor: np.ndarray,
    target_compression_ratio: float,
    tensor_train_args: dict[str, str | int] | None = None,
    frobenius_error_coef: float = 1.0,
    compression_ratio_coef: float = 10.0,
):
    """
    Computes the loss function for Tensor Train decomposition.

    Parameters
    ----------
    rank : list
        The ranks for Tensor Train decomposition.
    tensor : np.ndarray
        The input tensor to be decomposed.
    target_compression_ratio : float
        The desired compression ratio as a percentage.
    tensor_train_args : dict[str, str] | None, optional
        Arguments for Tensor Train decomposition, by default None.
    frobenius_error_coef : float, optional
        Coefficient for Frobenius norm error, by default 1.0.
    compression_ratio_coef : float, optional
        Coefficient for compression ratio penalty, by default 10.0.

    Returns
    -------
    float
        The computed loss value.

    """
    if tensor_train_args is None:
        tensor_train_args = {
            "svd": "truncated_svd",
        }

    try:
        tensor = tl.tensor(tensor).to("cuda") if tl.get_backend() == "pytorch" else tl.tensor(tensor)

        tt_factors = tl.decomposition.tensor_train(tensor, rank=rank, **tensor_train_args)
        reconstructed_tensor = tl.tt_to_tensor(tt_factors)

        frobenius_error = (tl.norm(reconstructed_tensor - tensor) / tl.norm(tensor)).item()
        compression_ratio = CompressionRatioTensorLyTensorTrainCalculator.calculate(tensor, tt_factors) / 100

        target_compression_ratio /= 100

        compression_penalty = (target_compression_ratio - compression_ratio) ** 2

        return frobenius_error_coef * frobenius_error + compression_ratio_coef * compression_penalty

    except Exception as e:
        print(e)
        return float("inf")


def loss_tensor_train_wrapper(
    tensor_train_rank: list[float],
    tensor: np.ndarray,
    target_compression_ratio: float,
    frobenius_error_coef: float,
    compression_ratio_coef: float,
    tensor_train_args: dict[str, str | int] | None,
) -> float:
    """
    Wrapper function to compute the loss for tensor train decomposition with rank constraints.

    Parameters
    ----------
    tensor_train_rank : list
        The list of tensor train ranks that are optimized.
    tensor : np.ndarray
        The input tensor to be decomposed.
    target_compression_ratio : float
        The desired compression ratio.
    frobenius_error_coef : float
        Coefficient for Frobenius norm error.
    compression_ratio_coef : float
        Coefficient for compression ratio penalty.
    tensor_train_args : dict[str, str]
        Arguments for tensor_train decomposition.

    Returns
    -------
    float
        The computed loss value.

    """
    loss = float("inf")
    tensor_train_rank = list(np.round(tensor_train_rank).astype(int))
    try:
        loss = loss_function_tensor_train(
            rank=tensor_train_rank,
            tensor=tensor,
            target_compression_ratio=target_compression_ratio,
            tensor_train_args=tensor_train_args,
            frobenius_error_coef=frobenius_error_coef,
            compression_ratio_coef=compression_ratio_coef,
        )
    except Exception as e:
        print(e)
    return loss


def global_optimize_tensor_train_rank(
    tensor: np.ndarray,
    target_compression_ratio: float,
    tensor_train_args: dict[str, str | int] | None = None,
    frobenius_error_coef: float = 1.0,
    compression_ratio_coef: float = 10.0,
    optimization_method: str = "differential_evolution",
    loss_function_fixed: Callable | None = None,
    verbose: bool = False,
):
    """
    Performs global optimization of Tensor Train decomposition ranks using evolutionary strategies.

    Function works with pytorch and numpy backend context of tensorly.

    Parameters
    ----------
    tensor : np.ndarray
        The input tensor to be decomposed.
    target_compression_ratio : float
        The desired compression ratio in percentage.
    tensor_train_args : dict[str, str] | None, optional
        Additional parameters for the Tensor Train decomposition algorithm (default: None).
    frobenius_error_coef : float, optional
        Weighting factor for the reconstruction error (default: 1.0).
    compression_ratio_coef : float, optional
        Weighting factor for the compression ratio penalty (default: 10.0).
    optimization_method : str, optional
        The optimization method used (default: "differential_evolution").
    loss_function_fixed : callable, optional
        The loss function to be minimized. If None, a default function is used (default: None).
    verbose : bool, optional
        If True, logs optimization steps and returns additional debugging information.

    Returns
    -------
    tuple | np.ndarray
        If `verbose` is True, returns a tuple containing:
        - reconstructed_tensor (np.ndarray): The reconstructed tensor after Tensor Train decomposition.
        - core_tensor (np.ndarray): The core tensor from the Tensor Train decomposition.
        - factors (list[np.ndarray]): The factor matrices of the Tensor Train decomposition.
        - optimal_rank (list[int]): The best-found Tensor Train ranks.
        - final_loss (float): The corresponding loss value.
        - optimization_result (OptimizeResult): The full optimization output.
        - logs (list[dict]): Detailed logs of optimization steps.

        If `verbose` is False, returns only:
        - reconstructed_tensor (np.ndarray): The reconstructed tensor after Tensor Train decomposition.

    """
    if loss_function_fixed is None:
        loss_function_fixed = partial(
            loss_tensor_train_wrapper,
            tensor=tensor,
            target_compression_ratio=target_compression_ratio,
            frobenius_error_coef=frobenius_error_coef,
            compression_ratio_coef=compression_ratio_coef,
            tensor_train_args=tensor_train_args,
        )

    if tensor_train_args is None:
        tensor_train_args = {
            "svd": "truncated_svd",
        }

    if verbose is True:

        class OptimizationLogger:
            def __init__(
                self,
                tensor: np.ndarray,
                tensor_train_args: dict[str, str | int],
                target_compression_ratio: float = 50.0,
                frobenius_error_coef: float = 1.0,
                compression_ratio_coef: float = 10.0,
            ):
                self.logs: list[str | float | list | dict[str, float] | OptimizeResult] = []
                self.current_iteration = -1

                if tl.get_backend() == "pytorch":
                    tl_tensor = tl.tensor(tensor).to("cuda")
                elif tl.get_backend() == "numpy":
                    tl_tensor = tl.tensor(tensor)
                else:
                    tl_tensor = tl.tensor(tensor)

                self.tensor = tl_tensor
                self.tensor_train_args = tensor_train_args
                self.target_compression_ratio = target_compression_ratio
                self.frobenius_error_coef = frobenius_error_coef
                self.compression_ratio_coef = compression_ratio_coef

            def calculate_metrics(
                self,
                rank: list,
            ) -> dict[str, float]:
                tt_factors = tl.decomposition.tensor_train(tensor, rank=rank, **tensor_train_args)
                reconstructed_tensor = tl.tt_to_tensor(tt_factors)

                target_compression_ratio = self.target_compression_ratio / 100

                frobenius_error = (tl.norm(reconstructed_tensor - self.tensor) / tl.norm(self.tensor)).item()
                compression_ratio = CompressionRatioTensorLyTensorTrainCalculator.calculate(tensor, tt_factors) / 100

                compression_penalty = (target_compression_ratio - compression_ratio) ** 2
                loss = frobenius_error_coef * frobenius_error + compression_ratio_coef * compression_penalty

                metrics = {
                    "frobenius_error": frobenius_error,
                    "compression_ratio": compression_ratio,
                    "compression_penalty": compression_penalty,
                    "loss": loss,
                }

                return metrics

            def callback(self, intermediate_result: OptimizeResult):
                self.current_iteration += 1

                rank = list(np.round(intermediate_result.x).astype(int))
                metrics = self.calculate_metrics(rank=rank)

                self.logs.append(
                    {
                        "step": self.current_iteration,
                        "rank": rank,
                        "metrics": metrics,
                        "raw_results": intermediate_result,
                    }
                )

        optimization_logger = OptimizationLogger(
            tensor=tensor,
            tensor_train_args=tensor_train_args,
            target_compression_ratio=target_compression_ratio,
            frobenius_error_coef=frobenius_error_coef,
            compression_ratio_coef=compression_ratio_coef,
        )

    # params
    tensor_train_bounds = (
        calculate_tt_bounds(tensor.shape) if optimization_method in ["differential_evolution"] else None
    )

    callback_param = optimization_logger.callback if optimization_method not in [] and verbose is True else None

    if optimization_method == "differential_evolution":
        optimization_kwargs_differential_evolution = {
            "func": loss_function_fixed,
            "bounds": tensor_train_bounds,
            "strategy": "best1bin",
            "maxiter": 50,
            "popsize": 10,
            "tol": 0.01,
            "atol": 0.001,
            "mutation": (0.3, 0.7),
            "recombination": 0.9,
            "init": "latinhypercube",
            "polish": True,
            "callback": callback_param,
            "disp": True,
        }

        if tl.get_backend() == "pytorch":
            optimization_kwargs_differential_evolution.update(
                {
                    "workers": 1,
                    "updating": "immediate",
                }
            )
        elif tl.get_backend() == "numpy":
            optimization_kwargs_differential_evolution.update(
                {
                    "workers": -1,
                    "updating": "deferred",
                }
            )

        result = differential_evolution(**optimization_kwargs_differential_evolution)

        optimal_rank = list(np.clip(np.round(result.x).astype(int), 1, None))
        final_loss = result.fun

        if tl.get_backend() == "pytorch":
            tl_tensor = tl.tensor(tensor).to("cuda")
        elif tl.get_backend() == "numpy":
            tl_tensor = tl.tensor(tensor)
        else:
            tl_tensor = tl.tensor(tensor)

        tt_factors = tl.decomposition.tensor_train(
            tl_tensor,
            rank=optimal_rank,
            **tensor_train_args,
        )
        reconstructed_tensor = tl.tt_to_tensor(tt_factors)

        if verbose is True:
            output = (reconstructed_tensor, tt_factors, optimal_rank, final_loss, result, optimization_logger.logs)
        else:
            output = reconstructed_tensor

    return output
