from collections.abc import Callable
from functools import partial

import numpy as np
import tensorly as tl
from scipy.optimize import OptimizeResult, differential_evolution

from src.model_compressor.calculate_bounds import calculate_tucker_bounds


def compression_ratio_nn(tensor, ranks: list[int] | tuple[int, int]) -> float:
    """
    Returns the custom compression ratio of the layer of neural network after Tucker decomposition.

    Parameters
    ----------
    tensor : np.ndarray
        The original tensor.
    ranks : list[int] | tuple[int, int]
        The Tucker ranks for decomposition.

    Returns
    -------
    float
        The computed compression ratio.

    """
    size = tensor.shape
    size1 = size[0] * ranks[0]
    size2 = ranks[0] * ranks[1] * size[2] ** 2
    size3 = size[1] * ranks[1]
    return (size1 + size2 + size3) / (size[0] * size[1] * size[2] ** 2)


def loss_function_tucker(
    rank: list,
    tensor: np.ndarray,
    target_compression_ratio: float,
    tucker_args: dict[str, str | int] | None = None,
    frobenius_error_coef: float = 1.0,
    compression_ratio_coef: float = 10.0,
):
    """
    Computes the loss function for Tucker decomposition.

    Parameters
    ----------
    rank : list
        The ranks for Tucker decomposition.
    tensor : np.ndarray
        The input tensor to be decomposed.
    target_compression_ratio : float
        The desired compression ratio as a percentage.
    tucker_args : dict[str, str] | None, optional
        Arguments for Tucker decomposition, by default None.
    frobenius_error_coef : float, optional
        Coefficient for Frobenius norm error, by default 1.0.
    compression_ratio_coef : float, optional
        Coefficient for compression ratio penalty, by default 10.0.

    Returns
    -------
    float
        The computed loss value.

    """
    if tucker_args is None:
        tucker_args = {
            "svd": "truncated_svd",
            "init": "random",
            "random_state": 42,
        }

    try:
        # TODO: закинуть tensor на нужный бэкенд
        weight, factors = tl.decomposition.tucker(tensor, rank=rank, **tucker_args)
        reconstructed_tensor = tl.tucker_to_tensor((weight, factors))

        frobenius_error = (tl.norm(reconstructed_tensor - tensor) / tl.norm(tensor)).item()
        compression_ratio = compression_ratio_nn(tensor=reconstructed_tensor, ranks=rank)

        target_compression_ratio /= 100

        compression_penalty = (target_compression_ratio - compression_ratio) ** 2

        return frobenius_error_coef * frobenius_error + compression_ratio_coef * compression_penalty

    except Exception as e:
        print(e)
        return float("inf")


def loss_tucker_wrapper(
    tucker_rank: list,
    tensor: np.ndarray,
    target_compression_ratio: float,
    frobenius_error_coef: float,
    compression_ratio_coef: float,
    tucker_args: dict[str, str | int] | None,
) -> float:
    """
    Wrapper function to compute the loss for Tucker decomposition with rank constraints.

    Parameters
    ----------
    tucker_rank : list
        The list of tucker ranks that are optimized.
    tensor : np.ndarray
        The input tensor to be decomposed.
    target_compression_ratio : float
        The desired compression ratio.
    frobenius_error_coef : float
        Coefficient for Frobenius norm error.
    compression_ratio_coef : float
        Coefficient for compression ratio penalty.
    tucker_args : dict[str, str]
        Arguments for Tucker decomposition.

    Returns
    -------
    float
        The computed loss value.

    """
    loss = float("inf")
    try:
        loss = loss_function_tucker(
            rank=tucker_rank,
            tensor=tensor,
            target_compression_ratio=target_compression_ratio,
            tucker_args=tucker_args,
            frobenius_error_coef=frobenius_error_coef,
            compression_ratio_coef=compression_ratio_coef,
        )
    except Exception as e:
        print(e)
    return loss


def global_optimize_tucker_rank(
    tensor: np.ndarray,
    target_compression_ratio: float,
    tucker_args: dict[str, str | int] | None = None,
    frobenius_error_coef: float = 1.0,
    compression_ratio_coef: float = 10.0,
    optimization_method: str = "differential_evolution",
    loss_function_fixed: Callable | None = None,
    verbose: bool = False,
):
    """
    Performs global optimization of Tucker decomposition ranks using evolutionary strategies.

    Parameters
    ----------
    tensor : np.ndarray
        The input tensor to be decomposed.
    target_compression_ratio : float
        The desired compression ratio in percentage.
    tucker_args : dict[str, str] | None, optional
        Additional parameters for the Tucker decomposition algorithm (default: None).
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
        - reconstructed_tensor (np.ndarray): The reconstructed tensor after Tucker decomposition.
        - core_tensor (np.ndarray): The core tensor from the Tucker decomposition.
        - factors (list[np.ndarray]): The factor matrices of the Tucker decomposition.
        - optimal_rank (list[int]): The best-found Tucker ranks.
        - final_loss (float): The corresponding loss value.
        - optimization_result (OptimizeResult): The full optimization output.
        - logs (list[dict]): Detailed logs of optimization steps.

        If `verbose` is False, returns only:
        - reconstructed_tensor (np.ndarray): The reconstructed tensor after Tucker decomposition.

    """
    if loss_function_fixed is None:
        loss_function_fixed = partial(
            loss_tucker_wrapper,
            tensor=tensor,
            target_compression_ratio=target_compression_ratio,
            frobenius_error_coef=frobenius_error_coef,
            compression_ratio_coef=compression_ratio_coef,
        )

    if tucker_args is None:
        tucker_args = {
            "svd": "truncated_svd",
            "init": "random",
            "random_state": 42,
        }

    if verbose is True:

        class OptimizationLogger:
            def __init__(
                self,
                tensor: np.ndarray,
                tucker_args: dict[str, str | int],
                target_compression_ratio: float = 50.0,
                frobenius_error_coef: float = 1.0,
                compression_ratio_coef: float = 10.0,
            ):
                self.logs: list[str | float | list | dict[str, float] | OptimizeResult] = []
                self.current_iteration = -1

                self.tensor = tensor
                self.tucker_args = tucker_args
                self.target_compression_ratio = target_compression_ratio
                self.frobenius_error_coef = frobenius_error_coef
                self.compression_ratio_coef = compression_ratio_coef

            def calculate_metrics(
                self,
                rank: list,
            ) -> dict[str, float]:
                weight, factors = tl.decomposition.tucker(tensor, rank=rank, **tucker_args)
                reconstructed_tensor = tl.tucker_to_tensor((weight, factors))

                target_compression_ratio = self.target_compression_ratio / 100

                frobenius_error = (tl.norm(reconstructed_tensor - tensor) / tl.norm(tensor)).item()
                compression_ratio = compression_ratio_nn(tensor=reconstructed_tensor, ranks=rank)
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
            tucker_args=tucker_args,
            target_compression_ratio=target_compression_ratio,
            frobenius_error_coef=frobenius_error_coef,
            compression_ratio_coef=compression_ratio_coef,
        )

    # params
    tucker_bounds = calculate_tucker_bounds(tensor.shape) if optimization_method in ["differential_evolution"] else None

    callback_param = optimization_logger.callback if optimization_method not in [] or verbose is True else None

    if optimization_method == "differential_evolution":
        optimization_kwargs_differential_evolution = {
            "func": loss_function_fixed,
            "bounds": tucker_bounds,
            "strategy": "best1bin",
            "maxiter": 50,
            "popsize": 10,
            "tol": 0.01,
            "atol": 0.001,
            "mutation": (0.3, 0.7),
            "recombination": 0.9,
            "init": "latinhypercube",
            "polish": True,
            "workers": -1,
            "updating": "deferred",
            "callback": callback_param,
            "disp": True,
        }

        result = differential_evolution(**optimization_kwargs_differential_evolution)

        optimal_rank = list(np.clip(np.round(result.x).astype(int), 1, None))
        final_loss = result.fun

        weight, factors = tl.decomposition.tucker(tensor, rank=optimal_rank, **tucker_args)
        reconstructed_tensor = tl.tucker_to_tensor((weight, factors))

        if verbose is True:
            output = (reconstructed_tensor, weight, factors, optimal_rank, final_loss, result, optimization_logger.logs)
        else:
            output = reconstructed_tensor

    return output
