from typing import Any

import numpy as np


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
    import gc

    import tensorly as tl
    import torch

    from src.utils.metrics_calculators import IMetricCalculator

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
    import gc

    import tensorly as tl
    import torch

    from src.utils.metrics_calculators import IMetricCalculator

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
    search_strategy: str = "custom",
):
    import gc

    import psutil
    import tensorly as tl
    import torch

    from src.utils.metrics_calculators import IMetricCalculator

    def check_memory_availability() -> None:
        """Check if there is sufficient RAM available."""
        if psutil.virtual_memory().available < 2 * tensor_cuda.element_size() * tensor_cuda.numel():
            raise MemoryError("Insufficient RAM available to process this tensor.")  # noqa: EM101

    if search_strategy not in ["custom"]:
        raise ValueError("Invalid search strategy. Currently only 'custom' is supported.")  # noqa: EM101

    if tensor_train_args is None:
        tensor_train_args = {"svd": "randomized_svd"}

    tensor_compression_logs = []
    tensor_shape = tensor.shape
    initial_rank = [1] * (len(tensor_shape) + 1)

    search_index = -1
    best_rank, best_compression, best_error = None, None, None
    current_compression_ratio = 0.0

    print("Optimal rank search process for TensorTrain:")
    print("iteration number | rank | compression ratio (%) | frobenius error (%)")

    with tl.backend_context("pytorch"):
        tensor_cuda = tl.tensor(tensor).to("cuda")

        while current_compression_ratio < target_compression_ratio:
            candidates = []
            next_best_rank = None
            next_best_compression = None
            next_best_error = float("inf")

            for dim in range(1, len(tensor_shape)):
                step_sizes = [1, 2, 5, 10]
                for step in step_sizes:
                    new_rank = initial_rank.copy()
                    new_rank[dim] += step
                    candidates.append(new_rank)

            for test_rank in candidates:
                try:
                    check_memory_availability()

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

                    if compression_ratio > current_compression_ratio and frobenius_error < next_best_error:
                        next_best_rank = test_rank.copy()
                        next_best_compression = compression_ratio
                        next_best_error = frobenius_error
                        current_compression_ratio = compression_ratio

                    del tt_factors, reconstructed_tensor
                except MemoryError as mem_err:
                    print(f"MemoryError: {mem_err}. Skipping current tensor.")
                    best_rank = initial_rank
                    break
                except Exception as e:
                    print(f"Error for rank {test_rank}: {e}")
                finally:
                    torch.cuda.empty_cache()
                    gc.collect()

            if next_best_rank is None or best_rank is not None:
                print("Stopping: no further improvements possible or memory issue encountered.")
                break

            initial_rank = next_best_rank
            best_rank, best_compression, best_error = (
                initial_rank,
                next_best_compression,
                next_best_error,
            )

            search_index += 1
            if best_compression is not None and best_error is not None:
                tensor_compression_logs.append(
                    {
                        "rank": best_rank.copy(),
                        "compression_ratio": best_compression,
                        "frobenius_error": best_error,
                    }
                )
                print(f"{search_index} | {best_rank} | {best_compression:.6f} % | {best_error:.6f} %")
            else:
                print(f"{search_index} | {best_rank} | No valid compression or error values.")

    return best_rank, best_compression, best_error, tensor_compression_logs
