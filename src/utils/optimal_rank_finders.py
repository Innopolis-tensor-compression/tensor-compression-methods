import gc
from typing import Any

import numpy as np
import psutil
import tensorly as tl
import torch

from src.utils.metrics_calculators import IMetricCalculator


def find_optimal_rank_tucker_by_frobenius_error(
    tensor: np.ndarray,
    tucker_args: dict[str, Any],
    frobenius_error_limit: float = 1.0,
    search_strategy: str = "binary",
) -> tuple[list[int], float, list[dict[str, Any]]]:
    """
    Finds the optimal rank for Tucker decomposition to ensure the Frobenius error does not exceed the specified limit.
    Supports binary and incremental search.

    :param tensor: Input tensor (np.array or torch.Tensor).
    :param tucker_args: Arguments for Tucker decomposition (e.g., SVD method).
    :param frobenius_error_limit: Frobenius error limit (%) between 1 and 100.
    :param search_strategy: Search strategy ("binary" or "incremental").
    :return: Optimal rank, minimum Frobenius error, compression ratio, and iteration logs.
    """
    if search_strategy not in ["binary", "incremental"]:
        raise ValueError("Invalid search strategy. Use 'binary' or 'incremental'.")  # noqa: EM101

    tensor_compression_logs: list[dict[str, Any]] = []
    tensor_shape = tensor.shape
    current_rank: list[int] = list(map(int, tensor_shape))
    search_index = -1
    previous_best_error = float("inf")

    print("Optimal rank search process:")
    print("iteration number | rank | frobenius error (%) | compression ratio (%)")

    with tl.backend_context("pytorch"):
        tensor_cuda = tl.tensor(tensor).to("cuda")

        while True:
            candidates: list[tuple[int, int]] = []
            best_rank: list[int] = []
            best_error: float = float("inf")
            prev_frobenius_error: float = float("inf")
            best_compression: float | None = None

            for dim in range(len(tensor_shape)):
                if current_rank[dim] > 1:
                    if search_strategy == "binary":
                        low, high = 1, current_rank[dim]
                        mid = (low + high) // 2
                        candidates.append((dim, mid))
                        candidates.append((dim, current_rank[dim] - 1))

                    if search_strategy == "incremental":
                        candidates.append((dim, current_rank[dim] - 1))

            candidates = list(set(candidates))

            next_iteration_errors: list[float] = []
            for dim, rank_value in candidates:
                test_rank = current_rank.copy()
                test_rank[dim] = rank_value

                try:
                    weight, factors = tl.decomposition.tucker(tensor_cuda, rank=test_rank, **tucker_args)
                    reconstructed_tensor = tl.tucker_to_tensor((weight, factors))
                    frobenius_error = (
                        100.0 * (tl.norm(reconstructed_tensor - tensor_cuda) / tl.norm(tensor_cuda)).item()
                    )
                    compression_ratio = (
                        100.0
                        * IMetricCalculator.get_tensors_size(weight, *factors)
                        / IMetricCalculator.get_tensors_size(tensor_cuda)
                    )

                    next_iteration_errors.append(frobenius_error)

                    if frobenius_error < prev_frobenius_error and frobenius_error <= frobenius_error_limit:
                        best_rank = test_rank.copy()
                        best_error = frobenius_error
                        best_compression = compression_ratio
                        prev_frobenius_error = frobenius_error

                    del weight, factors, reconstructed_tensor
                except Exception as e:
                    print(f"Error for rank {test_rank}: {e}")
                finally:
                    torch.cuda.empty_cache()
                    gc.collect()

            if (
                all(error > frobenius_error_limit for error in next_iteration_errors)
                and previous_best_error <= frobenius_error_limit
            ):
                print("Stopping: no further improvements possible or all next iterations exceed the error limit.")
                break

            current_rank = best_rank
            previous_best_error = best_error
            search_index += 1
            tensor_compression_logs.append(
                {
                    "rank": current_rank.copy(),
                    "frobenius_error": best_error,
                    "compression_ratio": best_compression,
                }
            )
            print(f"{search_index} | {current_rank} | {best_error:.6f} % | {best_compression:.6f} %")

    return current_rank, previous_best_error, tensor_compression_logs


def find_optimal_rank_tucker_by_compression_ratio(
    tensor: np.array,
    tucker_args: dict,
    target_compression_ratio: float,
    search_strategy: str = "binary",
):
    """
    Finds the optimal rank for Tucker decomposition to achieve a compression ratio as close as possible to the target,
    while minimizing the Frobenius error at each iteration.

    :param tensor: Input tensor (np.array or torch.Tensor).
    :param tucker_args: Arguments for Tucker decomposition (e.g., SVD method).
    :param target_compression_ratio: Target compression ratio (%) between 1 and 100.
    :param search_strategy: Search strategy ("binary" or "incremental").
    :return: Optimal rank, compression ratio, Frobenius error, and iteration logs.
    """
    if search_strategy not in ["binary", "incremental"]:
        raise ValueError("Invalid search strategy. Use 'binary' or 'incremental'.")  # noqa: EM101

    tensor_compression_logs = []
    tensor_shape = tensor.shape
    current_rank = list(tensor_shape)
    search_index = -1
    best_rank, best_compression, best_error = None, None, None

    print("Optimal rank search process:")
    print("iteration number | rank | compression ratio (%) | frobenius error (%)")

    with tl.backend_context("pytorch"):
        tensor_cuda = tl.tensor(tensor).to("cuda")

        while True:
            candidates = []
            next_best_rank = None
            next_best_compression = None
            next_best_error = float("inf")

            for dim in range(len(tensor_shape)):
                if current_rank[dim] > 1:
                    if search_strategy == "binary":
                        low, high = 1, current_rank[dim]
                        mid = (low + high) // 2
                        candidates.append((dim, mid))
                        candidates.append((dim, current_rank[dim] - 1))

                    if search_strategy == "incremental":
                        candidates.append((dim, current_rank[dim] - 1))

            candidates = list(set(candidates))

            for dim, rank_value in candidates:
                test_rank = current_rank.copy()
                test_rank[dim] = rank_value

                try:
                    weight, factors = tl.decomposition.tucker(tensor_cuda, rank=test_rank, **tucker_args)
                    reconstructed_tensor = tl.tucker_to_tensor((weight, factors))

                    frobenius_error = (
                        100.0 * (tl.norm(reconstructed_tensor - tensor_cuda) / tl.norm(tensor_cuda)).item()
                    )

                    compression_ratio = (
                        100.0
                        * IMetricCalculator.get_tensors_size(weight, *factors)
                        / IMetricCalculator.get_tensors_size(tensor_cuda)
                    )

                    # If we have reached or exceeded the target compression ratio
                    if compression_ratio >= target_compression_ratio and frobenius_error < next_best_error:
                        next_best_rank = test_rank.copy()
                        next_best_compression = compression_ratio
                        next_best_error = frobenius_error

                    del weight, factors, reconstructed_tensor
                except Exception as e:
                    print(f"Error for rank {test_rank}: {e}")
                finally:
                    torch.cuda.empty_cache()
                    gc.collect()

            # Stop if no further improvement is found
            if next_best_rank is None:
                print("Stopping: no further improvements possible.")
                break

            current_rank = next_best_rank
            best_rank, best_compression, best_error = (
                current_rank,
                next_best_compression,
                next_best_error,
            )

            search_index += 1
            tensor_compression_logs.append(
                {
                    "rank": best_rank.copy(),
                    "compression_ratio": best_compression,
                    "frobenius_error": best_error,
                }
            )
            print(f"{search_index} | {best_rank} | {best_compression:.6f} % | {best_error:.6f} %")

    return best_rank, best_compression, best_error, tensor_compression_logs


def find_optimal_rank_tensor_train_by_compression_ratio(
    tensor: np.array,
    target_compression_ratio: float,
    tensor_train_args: dict | None = None,
    initial_rank_arg: list | None = None,
    search_strategy: str = "custom",
):
    def check_memory_availability(tensor_cuda: tl.tensor) -> None:
        """Check if there is sufficient RAM available."""
        if psutil.virtual_memory().available < 2 * tensor_cuda.element_size() * tensor_cuda.numel():
            raise MemoryError("Insufficient RAM available to process this tensor.")  # noqa: EM101

    def calculate_tt_bounds(shape: tuple | list) -> list:
        """
        Calculates the bounds for TT-ranks of a tensor based on its shape.

        Parameters
        ----------
        shape : tuple or list
            List or tuple of tensor dimensions. Each element represents the size of the tensor along that dimension.

        Returns
        -------
        list
            List of rank bounds in the format [(1, 1), (1, r1_max), ..., (1, 1)].

        """
        d = len(shape)
        bounds = [(1, 1)]

        for k in range(1, d):
            prod_left = 1
            for i in range(k):
                prod_left *= shape[i]

            prod_right = 1
            for j in range(k, d):
                prod_right *= shape[j]

            rk_max = min(prod_left, prod_right)
            bounds.append((1, rk_max))

        bounds.append((1, 1))
        return bounds

    if search_strategy not in ["custom"]:
        raise ValueError("Invalid search strategy. Currently only 'custom' is supported.")  # noqa: EM101

    if tensor_train_args is None:
        tensor_train_args = {"svd": "truncated_svd"}

    tensor_compression_logs = []
    tensor_shape = tensor.shape
    initial_rank: list[int] = [1] * (len(tensor_shape) + 1) if initial_rank_arg is None else initial_rank_arg

    if len(initial_rank) != len(tensor_shape) + 1:
        raise ValueError(
            f"Length of initial rank must be equal to length of tensor shape plus 1. Length of initial rank is {len(initial_rank)} and length of tensor shape is {len(tensor_shape)}"
        )

    best_rank = initial_rank.copy()
    best_compression = 0.0
    best_error = float("inf")

    fixed_dimensions = set()

    print("Optimal rank search process for TensorTrain:")
    print("step | rank | compression ratio (%) | frobenius error (%)")

    with tl.backend_context("pytorch"):
        tensor_cuda = tl.tensor(tensor).to("cuda")

        step = 0
        while True:
            step += 1
            candidates = []

            current_best_rank = None
            current_best_compression = 0.0
            current_best_error = float("inf")

            secondary_best_rank = None
            secondary_best_compression = 0.0
            secondary_best_error = float("inf")

            # Generate candidates by adding 1 to each dimension's rank, excluding fixed dimensions
            for dim in range(1, len(tensor_shape)):
                if dim in fixed_dimensions:
                    continue
                new_rank = best_rank.copy()
                new_rank[dim] += 1
                candidates.append(new_rank)

            for test_rank in candidates:
                try:
                    check_memory_availability(tensor_cuda)

                    method_result = tl.decomposition.tensor_train(tensor_cuda, rank=test_rank, **tensor_train_args)
                    tt_factors = method_result
                    reconstructed_tensor = tl.tt_to_tensor(tt_factors)

                    frobenius_error = (
                        100.0 * (tl.norm(reconstructed_tensor - tensor_cuda) / tl.norm(tensor_cuda)).item()
                    )
                    compression_ratio = (
                        100.0
                        * IMetricCalculator.get_tensors_size(*tt_factors)
                        / IMetricCalculator.get_tensors_size(tensor_cuda)
                    )

                    # Primary priority: minimize error and increase compression
                    if compression_ratio > current_best_compression and frobenius_error < current_best_error:
                        current_best_rank = test_rank.copy()
                        current_best_compression = compression_ratio
                        current_best_error = frobenius_error

                    # Secondary priority: increase compression even if error grows
                    elif compression_ratio > secondary_best_compression and frobenius_error >= current_best_error:
                        secondary_best_rank = test_rank.copy()
                        secondary_best_compression = compression_ratio
                        secondary_best_error = frobenius_error

                    # Check if this dimension should be fixed
                    if compression_ratio == best_compression:
                        for dim in range(1, len(test_rank)):
                            if test_rank[dim] > best_rank[dim]:
                                fixed_dimensions.add(dim)

                    del tt_factors, reconstructed_tensor
                except MemoryError as mem_err:
                    print(f"MemoryError: {mem_err}. Skipping current tensor.")
                except Exception as e:
                    print(f"Error for rank {test_rank}: {e}")
                finally:
                    torch.cuda.empty_cache()
                    gc.collect()

            # Choose the best candidate based on priorities
            if current_best_rank is not None:
                best_rank = current_best_rank.copy()
                best_compression = current_best_compression
                best_error = current_best_error
            elif secondary_best_rank is not None:
                best_rank = secondary_best_rank.copy()
                best_compression = secondary_best_compression
                best_error = secondary_best_error
            else:
                print("Stopping: No further improvements or all candidates exceed compression ratio.")
                break

            print(f"{step} | {best_rank} | {best_compression:.6f} % | {best_error:.6f} %")

            tensor_compression_logs.append(
                {
                    "rank": best_rank.copy(),
                    "compression_ratio": best_compression,
                    "frobenius_error": best_error,
                }
            )

            if best_compression > target_compression_ratio:  # Compression limit met
                print("Target compression ratio reached. Stopping search.")
                break

    print(f"Optimal rank: {best_rank}, Compression: {best_compression}%, Error: {best_error}%")
    return best_rank, best_compression, best_error, tensor_compression_logs
