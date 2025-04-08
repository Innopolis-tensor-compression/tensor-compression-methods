"""
This module contains functions for compressing entire module and Conv2d, ConvTranspose2d, Linear layers.
"""

import time
from copy import deepcopy
from typing import Type
from warnings import warn

from tensorly import set_backend
from tensorly.decomposition import parafac, tucker
from tltorch import FactorizedLinear
import torch
from torch.nn import Conv2d, ConvTranspose2d, Linear, Module, Sequential
from torch.nn.modules.loss import _Loss
from torch.nn.parameter import Parameter
from torch.optim import Optimizer

from src.model_compressor.calculate_optimized_rank_for_nn_layer import global_optimize_tucker_rank

set_backend("pytorch")


def svd_conv2d(conv_layer: Conv2d, rank_cpd: int = None) -> Sequential:
    """
    Compresses Conv2d layer with kernel size of (1, 1) using SVD decomposition.

    Args:
        conv_layer: Conv2d layer to compress.
        rank_cpd: Rank of CPD decomposition. If not specified, it will be set to the smallest dimension of the kernel.

    Returns:
        Sequential: Compressed ConvTranspose2d layer.

    """
    # Params of source conv_layer
    out_channels = conv_layer.out_channels
    in_channels = conv_layer.in_channels
    stride = conv_layer.stride
    bias = conv_layer.bias is not None  # TODO check how to handle this
    conv_weight = conv_layer.weight.squeeze().squeeze()

    # Due to limitations of tensorly, only float32 is supported
    if conv_weight.dtype != torch.float32:
        warn("Currently only float32 supported. It will be converted to float32")

    # If rank_cpd is not specified, it will be set to the smallest dimension of the kernel
    if rank_cpd is None:
        rank_cpd = min(conv_weight.shape)

    # SVD decomposition (CPD with only 2 dimensions)
    _, factors = parafac(conv_weight, rank_cpd)

    # Reshape factors to fit ConvTranspose2d layer
    factor_cpd_input = factors[1].permute([1, 0]).unsqueeze(2).unsqueeze(3)
    factor_cpd_output = factors[0].unsqueeze(2).unsqueeze(3)

    # Create compressed ConvTranspose2d layer
    conv1 = Conv2d(in_channels, rank_cpd, 1, stride=stride, dtype=torch.float32, bias=bias)
    conv2 = Conv2d(rank_cpd, out_channels, 1, dtype=torch.float32, bias=bias)
    conv1.weight = Parameter(factor_cpd_input)
    conv2.weight = Parameter(factor_cpd_output)
    return Sequential(conv1, conv2)


def cpd_conv2d(conv_layer: Conv2d, rank_cpd: int = None, rank_tkd: list[int] | tuple[int, int] = None) -> Sequential:
    """
    Compresses Conv2d layer using CPD decomposition.

    Args:
        conv_layer: Conv2d layer to compress.
        rank_cpd: Rank of CPD decomposition. If not specified, it will be set to the smallest dimension of the kernel.
        rank_tkd: Unused. For compatibility reasons.

    Returns:
        Sequential: Compressed ConvTranspose2d layer.

    """
    if conv_layer.kernel_size == (1, 1):
        return svd_conv2d(conv_layer, rank_cpd)

    # Params of source conv_layer
    out_channels = conv_layer.out_channels
    in_channels = conv_layer.in_channels
    kernel_size_x = conv_layer.kernel_size[0]
    kernel_size_y = conv_layer.kernel_size[1]
    stride = conv_layer.stride
    padding = conv_layer.padding
    dilation = conv_layer.dilation
    bias = conv_layer.bias is not None
    conv_weight = conv_layer.weight.reshape(out_channels, in_channels, kernel_size_x * kernel_size_y)

    # Due to limitations of tensorly, only float32 is supported
    if conv_weight.dtype != torch.float32:
        warn("Currently only float32 supported. It will be converted to float32")

    # If rank_cpd is not specified, it will be set to the smallest dimension of the kernel
    if rank_cpd is None:
        rank_cpd = max(conv_weight.shape)

    # CPD decomposition
    _, factors_cpd = parafac(conv_weight, rank_cpd, init="random")

    # Reshape factors to fit Conv2d layer
    factor_cpd_input = factors_cpd[1].permute([1, 0]).unsqueeze(2).unsqueeze(3)
    factor_cpd_hidden = factors_cpd[2].permute([1, 0]).unsqueeze(1).reshape(rank_cpd, 1, kernel_size_x, kernel_size_y)
    factor_cpd_output = factors_cpd[0].unsqueeze(2).unsqueeze(3)

    # Create compressed Conv2d layer
    conv1_cpd = Conv2d(in_channels, rank_cpd, 1, dtype=torch.float32, bias=bias)
    conv2_cpd = Conv2d(
        rank_cpd,
        rank_cpd,
        (kernel_size_x, kernel_size_y),
        groups=rank_cpd,
        stride=stride,
        padding=padding,
        dilation=dilation,
        dtype=torch.float32,
        bias=bias,
    )
    conv3_cpd = Conv2d(rank_cpd, out_channels, 1, dtype=torch.float32, bias=bias)
    conv1_cpd.weight = Parameter(factor_cpd_input)
    conv2_cpd.weight = Parameter(factor_cpd_hidden)
    conv3_cpd.weight = Parameter(factor_cpd_output)

    return Sequential(conv1_cpd, conv2_cpd, conv3_cpd)


def tkd_conv2d(conv_layer: Conv2d, rank_cpd: int = None, rank_tkd: list[int] | tuple[int, int] = None) -> Sequential:
    """
    Compresses Conv2d layer using Tucker decomposition.

    Args:
        conv_layer: Conv2d layer to compress.
        rank_cpd: Unused. For compatibility reasons.
        rank_tkd: Rank of Tucker decomposition. If not specified, it will be set to the size of ConvTranspose2d layer.

    Returns:
        Sequential: Compressed ConvTranspose2d layer.

    """
    if conv_layer.kernel_size == (1, 1):
        return svd_conv2d(conv_layer, rank_cpd)

    # Params of source conv_layer
    out_channels = conv_layer.out_channels
    in_channels = conv_layer.in_channels
    kernel_size_x = conv_layer.kernel_size[0]
    kernel_size_y = conv_layer.kernel_size[1]
    stride = conv_layer.stride
    padding = conv_layer.padding
    dilation = conv_layer.dilation
    bias = conv_layer.bias is not None
    conv_weight = conv_layer.weight.reshape(out_channels, in_channels, kernel_size_x * kernel_size_y)

    # Due to limitations of tensorly, only float32 is supported
    if conv_weight.dtype != torch.float32:
        warn("Currently only float32 supported. It will be converted to float32")

    # If rank_tkd is not specified, it will be set to the dimensions of the Conv2d layer
    if rank_tkd is None:
        rank_tkd = [out_channels, in_channels]
    else:
        if rank_tkd[1] > in_channels:
            rank_tkd = [rank_tkd[0], in_channels]
            warn("rank_tkd[1] is bigger then in_channels")
        if rank_tkd[0] > out_channels:
            rank_tkd = [out_channels, rank_tkd[1]]
            warn("rank_tkd[0] is bigger then out_channels")

    # TKD decomposition
    core_tkd, factors_tkd = tucker(conv_weight, rank_tkd + [kernel_size_x * kernel_size_y])

    # Reshape factors to fit Conv2d layer
    factor_tkd_input = factors_tkd[1].permute([1, 0]).unsqueeze(2).unsqueeze(3)
    factor_tkd_hidden = (
        torch.tensordot(factors_tkd[2], core_tkd, dims=([1], [2]))
        .permute([1, 2, 0])
        .reshape(rank_tkd[0], rank_tkd[1], kernel_size_x, kernel_size_y)
    )
    factor_tkd_output = factors_tkd[0].unsqueeze(2).unsqueeze(3)

    # Create compressed Conv2d layer
    conv1_tkd = Conv2d(in_channels, rank_tkd[1], 1, dtype=torch.float32, bias=bias)
    conv2_tkd = Conv2d(
        rank_tkd[1],
        rank_tkd[0],
        (kernel_size_x, kernel_size_y),
        stride=stride,
        padding=padding,
        dilation=dilation,
        dtype=torch.float32,
        bias=bias,
    )
    conv3_tkd = Conv2d(rank_tkd[0], out_channels, 1, dtype=torch.float32, bias=bias)
    conv1_tkd.weight = Parameter(factor_tkd_input)
    conv2_tkd.weight = Parameter(factor_tkd_hidden)
    conv3_tkd.weight = Parameter(factor_tkd_output)
    torch.cuda.empty_cache()
    return Sequential(conv1_tkd, conv2_tkd, conv3_tkd)


def tkd_cpd_conv2d(
    conv_layer: Conv2d, rank_cpd: int = None, rank_tkd: list[int] | tuple[int, int] = None
) -> Sequential:
    """
    Compresses Conv2d layer using combination of Tucker decomposition and CPD decomposition.

    Args:
        conv_layer: Conv2d layer to compress.
        rank_cpd: Rank of CPD decomposition. If not specified, it will be set to the smallest dimension of the kernel.
        rank_tkd: Rank of Tucker decomposition. If not specified, it will be set to the size of ConvTranspose2d layer.

    Returns:
        Sequential: Compressed ConvTranspose2d layer.

    """
    if conv_layer.kernel_size == (1, 1):
        return svd_conv2d(conv_layer, rank_cpd)

    # Params of source conv_layer
    out_channels = conv_layer.out_channels
    in_channels = conv_layer.in_channels
    kernel_size_x = conv_layer.kernel_size[0]
    kernel_size_y = conv_layer.kernel_size[1]
    stride = conv_layer.stride
    padding = conv_layer.padding
    dilation = conv_layer.dilation
    bias = conv_layer.bias is not None
    conv_weight = conv_layer.weight.reshape(out_channels, in_channels, kernel_size_x * kernel_size_y)

    # Due to limitations of tensorly, only float32 is supported
    if conv_weight.dtype != torch.float32:
        warn("Currently only float32 supported. It will be converted to float32")

    # If rank_tkd is not specified, it will be set to the dimensions of the Conv2d layer
    if rank_tkd is None:
        rank_tkd = [out_channels, in_channels]
    else:
        if rank_tkd[1] > in_channels:
            rank_tkd = [rank_tkd[0], in_channels]
            warn("rank_tkd[1] is bigger then in_channels")
        if rank_tkd[0] > out_channels:
            rank_tkd = [out_channels, rank_tkd[1]]
            warn("rank_tkd[0] is bigger then out_channels")

    # TKD decomposition
    core_tkd, factors_tkd = tucker(conv_weight, rank_tkd + [kernel_size_x * kernel_size_y])

    # Reshape factors to fit Conv2d layer
    factor_tkd_input = factors_tkd[1].permute([1, 0]).unsqueeze(2).unsqueeze(3)
    factor_tkd_hidden = (
        torch.tensordot(factors_tkd[2], core_tkd, dims=([1], [2]))
        .permute([1, 2, 0])
        .reshape(rank_tkd[0], rank_tkd[1], kernel_size_x, kernel_size_y)
    )
    factor_tkd_output = factors_tkd[0].unsqueeze(2).unsqueeze(3)

    # CPD decomposition of middle Conv2d
    conv2_tkd = Conv2d(
        rank_tkd[1],
        rank_tkd[0],
        (kernel_size_x, kernel_size_y),
        stride=stride,
        padding=padding,
        dilation=dilation,
        dtype=torch.float32,
        bias=bias,
    )
    conv2_tkd.weight = Parameter(factor_tkd_hidden)
    conv2_tkd = cpd_conv2d(conv2_tkd, rank_cpd=rank_cpd)

    # Create compressed Conv2d layer
    conv1_tkd = Conv2d(in_channels, rank_tkd[1], 1, dtype=torch.float32, bias=bias)
    conv3_tkd = Conv2d(rank_tkd[0], out_channels, 1, dtype=torch.float32, bias=bias)
    conv1_tkd.weight = Parameter(factor_tkd_input)
    conv3_tkd.weight = Parameter(factor_tkd_output)

    return Sequential(conv1_tkd, conv2_tkd, conv3_tkd)


# TODO Check if correct for ConvTranspose2d
def svd_conv_transpose2d(conv_layer: ConvTranspose2d, rank_cpd: int = None) -> Sequential:
    """
    Compresses ConvTranspose2d layer with kernel size of (1, 1) using SVD decomposition.

    Args:
        conv_layer: ConvTranspose2d layer to compress.
        rank_cpd: Rank of CPD decomposition. If not specified, it will be set to the smallest dimension of the kernel.

    Returns:
        Sequential: Compressed ConvTranspose2d layer.

    """
    # Params of source conv_layer
    out_channels = conv_layer.out_channels
    in_channels = conv_layer.in_channels
    stride = conv_layer.stride
    # TODO check how to handle this
    bias = conv_layer.bias is not None
    conv_weight = conv_layer.weight.squeeze().squeeze()

    # Due to limitations of tensorly, only float32 is supported
    if conv_weight.dtype != torch.float32:
        warn("Currently only float32 supported. It will be converted to float32")

    # If rank_cpd is not specified, it will be set to the smallest dimension of the kernel
    if rank_cpd is None:
        rank_cpd = min(conv_weight.shape)

    # SVD decomposition (CPD with only 2 dimensions)
    _, factors = parafac(conv_weight, rank_cpd)

    # Reshape factors to fit ConvTranspose2d layer
    factor_svd_input = factors[1].permute([1, 0]).unsqueeze(2).unsqueeze(3)
    factor_svd_output = factors[0].unsqueeze(2).unsqueeze(3)

    # Create compressed ConvTranspose2d layer
    conv1 = ConvTranspose2d(in_channels, rank_cpd, 1, stride=stride, dtype=torch.float32, bias=bias)
    conv2 = ConvTranspose2d(rank_cpd, out_channels, 1, dtype=torch.float32, bias=bias)
    conv1.weight = Parameter(factor_svd_input)
    conv2.weight = Parameter(factor_svd_output)
    return Sequential(conv1, conv2)


def cpd_conv_transpose2d(
    conv_layer: ConvTranspose2d, rank_cpd: int = None, rank_tkd: list[int] | tuple[int, int] = None
) -> Sequential:
    """
    Compresses ConvTranspose2d layer using CPD decomposition.

    Args:
        conv_layer: ConvTranspose2d layer to compress.
        rank_cpd: Rank of CPD decomposition. If not specified, it will be set to the smallest dimension of the kernel.
        rank_tkd: Unused. For compatibility reasons.

    Returns:
        Sequential: Compressed ConvTranspose2d layer.

    """
    if conv_layer.kernel_size == (1, 1):
        return svd_conv_transpose2d(conv_layer, rank_cpd)

    # Params of source conv_layer
    out_channels = conv_layer.out_channels
    in_channels = conv_layer.in_channels
    kernel_size_x = conv_layer.kernel_size[0]
    kernel_size_y = conv_layer.kernel_size[1]
    stride = conv_layer.stride
    padding = conv_layer.padding
    dilation = conv_layer.dilation
    bias = conv_layer.bias is not None
    conv_weight = conv_layer.weight.reshape(out_channels, in_channels, kernel_size_x * kernel_size_y)

    # Due to limitations of tensorly, only float32 is supported
    if conv_weight.dtype != torch.float32:
        warn("Currently only float32 supported. It will be converted to float32")

    # If rank_cpd is not specified, it will be set to the smallest dimension of the kernel
    if rank_cpd is None:
        rank_cpd = max(conv_weight.size())

    if rank_cpd > max(conv_weight.size()):
        warn("rank_cpd is bigger than the size of largest dimension. Setting it equal to the size of largest dimension")
        rank_cpd = max(conv_weight.size())

    # CPD decomposition
    _, factors_cpd = parafac(conv_weight, rank_cpd, init="random")

    # Reshape factors to fit ConvTranspose2d layer
    factor_cpd_input = factors_cpd[0].unsqueeze(2).unsqueeze(3)
    factor_cpd_hidden = factors_cpd[2].permute([1, 0]).unsqueeze(1).reshape(rank_cpd, 1, kernel_size_x, kernel_size_y)
    factor_cpd_output = factors_cpd[1].permute([1, 0]).unsqueeze(2).unsqueeze(3)

    # Create compressed ConvTranspose2d layer
    conv1_cpd = ConvTranspose2d(in_channels, rank_cpd, 1, dtype=torch.float32, bias=bias)
    conv2_cpd = ConvTranspose2d(
        rank_cpd,
        rank_cpd,
        (kernel_size_x, kernel_size_y),
        groups=rank_cpd,
        stride=stride,
        padding=padding,
        dilation=dilation,
        dtype=torch.float32,
        bias=bias,
    )
    conv3_cpd = ConvTranspose2d(rank_cpd, out_channels, 1, dtype=torch.float32, bias=bias)
    conv1_cpd.weight = Parameter(factor_cpd_input)
    conv2_cpd.weight = Parameter(factor_cpd_hidden)
    conv3_cpd.weight = Parameter(factor_cpd_output)

    return Sequential(conv1_cpd, conv2_cpd, conv3_cpd)


def tkd_conv_transpose2d(
    conv_layer: ConvTranspose2d, rank_cpd: int = None, rank_tkd: list[int] | tuple[int, int] = None
) -> Sequential:
    """
    Compresses ConvTranspose2d layer using Tucker decomposition.

    Args:
        conv_layer: ConvTranspose2d layer to compress.
        rank_cpd: Unused. For compatibility reasons.
        rank_tkd: Rank of Tucker decomposition. If not specified, it will be set to the size of ConvTranspose2d layer.

    Returns:
        Sequential: Compressed ConvTranspose2d layer.

    """
    if conv_layer.kernel_size == (1, 1):
        return svd_conv_transpose2d(conv_layer, min(rank_tkd))

    # Params of source conv_layer
    out_channels = conv_layer.out_channels
    in_channels = conv_layer.in_channels
    kernel_size_x = conv_layer.kernel_size[0]
    kernel_size_y = conv_layer.kernel_size[1]
    stride = conv_layer.stride
    padding = conv_layer.padding
    dilation = conv_layer.dilation
    bias = conv_layer.bias is not None
    conv_weight = conv_layer.weight.reshape(out_channels, in_channels, kernel_size_x * kernel_size_y)

    # Due to limitations of tensorly, only float32 is supported
    if conv_weight.dtype != torch.float32:
        warn("Currently only float32 supported. It will be converted to float32")

    # If rank_tkd is not specified, it will be set to the dimensions of the ConvTranspose2d layer
    if rank_tkd is None:
        rank_tkd = [out_channels, in_channels]
    else:
        if rank_tkd[1] > in_channels:
            rank_tkd = (rank_tkd[0], in_channels)
            warn("rank_tkd[1] is bigger then in_channels")
        if rank_tkd[0] > out_channels:
            rank_tkd = (out_channels, rank_tkd[1])
            warn("rank_tkd[0] is bigger then out_channels")

    # TKD decomposition
    core_tkd, factors_tkd = tucker(conv_weight, rank_tkd + [kernel_size_y * kernel_size_x])

    # Reshape factors to fit ConvTranspose2d layer
    factor_tkd_input = factors_tkd[0].unsqueeze(2).unsqueeze(3)
    factor_tkd_hidden = torch.tensordot(core_tkd, factors_tkd[2], dims=([2], [1])).reshape(
        rank_tkd[0], rank_tkd[1], kernel_size_x, kernel_size_y
    )
    factor_tkd_output = factors_tkd[1].permute([1, 0]).unsqueeze(2).unsqueeze(3)

    # Create compressed ConvTranspose2d layer
    conv1_tkd = ConvTranspose2d(in_channels, rank_tkd[0], 1, dtype=torch.float32, bias=bias)
    conv2_tkd = ConvTranspose2d(
        rank_tkd[0],
        rank_tkd[1],
        (kernel_size_x, kernel_size_y),
        stride=stride,
        padding=padding,
        dilation=dilation,
        dtype=torch.float32,
        bias=bias,
    )
    conv3_tkd = ConvTranspose2d(rank_tkd[1], out_channels, 1, dtype=torch.float32, bias=bias)
    conv1_tkd.weight = Parameter(factor_tkd_input)
    conv2_tkd.weight = Parameter(factor_tkd_hidden)
    conv3_tkd.weight = Parameter(factor_tkd_output)

    return Sequential(conv1_tkd, conv2_tkd, conv3_tkd)


def tkd_cpd_conv_transpose2d(
    conv_layer: ConvTranspose2d, rank_cpd: int = None, rank_tkd: list[int] | tuple[int, int] = None
) -> Sequential:
    """
    Compresses ConvTranspose2d layer using combination of Tucker decomposition and CPD decomposition.

    Args:
        conv_layer: ConvTranspose2d layer to compress.
        rank_cpd: Rank of CPD decomposition. If not specified, it will be set to the smallest dimension of the kernel.
        rank_tkd: Rank of Tucker decomposition. If not specified, it will be set to the size of ConvTranspose2d layer.

    Returns:
        Sequential: Compressed ConvTranspose2d layer.

    """
    if conv_layer.kernel_size == (1, 1):
        return svd_conv_transpose2d(conv_layer, rank_cpd)

    # Params of source conv_layer
    out_channels = conv_layer.out_channels
    in_channels = conv_layer.in_channels
    kernel_size_x = conv_layer.kernel_size[0]
    kernel_size_y = conv_layer.kernel_size[1]
    stride = conv_layer.stride
    padding = conv_layer.padding
    dilation = conv_layer.dilation
    bias = conv_layer.bias is not None
    groups = conv_layer.groups
    conv_weight = conv_layer.weight.reshape(out_channels, in_channels, kernel_size_x * kernel_size_y)


    # Due to limitations of tensorly, only float32 is supported
    if conv_weight.dtype != torch.float32:
        warn("Currently only float32 supported. It will be converted to float32")

    # If rank_tkd is not specified, it will be set to the dimensions of the ConvTranspose2d layer
    if rank_tkd is None:
        rank_tkd = [out_channels, in_channels]
    else:
        if rank_tkd[1] > in_channels:
            rank_tkd = (rank_tkd[0], in_channels)
            warn("rank_tkd[1] is bigger then in_channels")
        if rank_tkd[0] > out_channels:
            rank_tkd = (out_channels, rank_tkd[1])
            warn("rank_tkd[0] is bigger then out_channels")

    # TKD decomposition
    core_tkd, factors_tkd = tucker(conv_weight, rank_tkd + [kernel_size_x * kernel_size_y])

    # Reshape factors to fit ConvTranspose2d layer
    factor_tkd_input = factors_tkd[0].unsqueeze(2).unsqueeze(3)
    factor_tkd_hidden = (
        torch.tensordot(core_tkd, factors_tkd[2], dims=([2], [1]))
        .permute([1, 2, 0])
        .reshape(rank_tkd[0], rank_tkd[1], kernel_size_x, kernel_size_y)
    )
    factor_tkd_output = factors_tkd[1].permute([1, 0]).unsqueeze(2).unsqueeze(3)

    # CPD decomposition of middle ConvTranspose2d
    conv2_tkd = ConvTranspose2d(
        rank_tkd[0],
        rank_tkd[1],
        (kernel_size_x, kernel_size_y),
        stride=stride,
        padding=padding,
        dilation=dilation,
        dtype=torch.float32,
        bias=bias,
        groups=groups
    )
    conv2_tkd.weight = Parameter(factor_tkd_hidden)
    conv2_tkd = cpd_conv_transpose2d(conv2_tkd, rank_cpd=rank_cpd)

    # Create compressed ConvTranspose2d layer
    conv1_tkd = ConvTranspose2d(in_channels, rank_tkd[0], 1, dtype=torch.float32, bias=bias)
    conv3_tkd = ConvTranspose2d(rank_tkd[1], out_channels, 1, dtype=torch.float32, bias=bias)
    conv1_tkd.weight = Parameter(factor_tkd_input)
    conv3_tkd.weight = Parameter(factor_tkd_output)

    return Sequential(conv1_tkd, conv2_tkd, conv3_tkd)


def compress_model(
    model: Module,
    layers: list[type[Conv2d | ConvTranspose2d | Linear]] = None,
    conv_compression_method: str = "TKD",
    conv_transpose_compression_method: str = "TKD",
    linear_compress_method: str = "None",
    target_compression_ratio: float = 50.0,
    frobenius_error_coef: float = 1.0,
    compression_ratio_coef: float = 10.0,
    rank_cpd: int|None = None,
    rank_tkd: list[int] | tuple[int, int] = None,
    finetune: bool = False,
    epochs: int = 10,
    optimizer: Type[Optimizer] = None,
    data_size: list[int] = None,
    lr: float = 10e-3,
    loss_function: Type[_Loss] = None,
    batch_size: int = 32,
    number_of_iterations: int = 100,
    finetune_device: torch.device | str = "cpu",
    task: str | None = None,
) -> None:
    """
    Compresses specified layers of the model using specified compression methods

    Args:
        model: Model to compress.
        layers: List of layer types to compress. Default: [torch.nn.Linear, torch.nn.Conv2d, torch.nn.ConvTranspose2d].
        conv_compression_method: Method for compressing ConvTranspose2d layers. Possible values: "CPD", "TKD", "TKDCPD".
        conv_transpose_compression_method: Method for compressing ConvTranspose2d layers. Possible values: "CPD", "TKD", "TKDCPD".
        linear_compress_method: Method for compressing Linear layers. Possible values: "CP", "TT", "Tucker".
        target_compression_ratio: Target ratio for compression. If rank is not None, it is ignored.
        frobenius_error_coef: Coefficient for optimizing rank, based on frobenius error.
        compression_ratio_coef: oefficient for optimizing rank, based on compression ratio.
        rank_cpd: Rank of CPD decomposition.
        rank_tkd: Rank of Tucker decomposition.
        finetune: If True, the model will be fine-tuned after compression. Default: False.
        epochs: Number of epochs for fine-tuning. Default: 10.
        optimizer: Optimizer for fine-tuning. Default: SGD.
        data_size: Size of data for fine-tuning. Default: [3, 224, 224].
        lr: Learning rate for fine-tuning. Default: 10e-3.
        loss_function: Loss function for fine-tuning.
        batch_size: Batch size for fine-tuning. Default: 32.
        number_of_iterations: Number of iterations for fine-tuning. Default: 100.
        finetune_device: Device for fine-tuning. Default: "cpu".
        task: Task for fine-tuning. Default: None.

    Raises:
        ValueError: If an unknown compression method is specified.

    """
    if finetune:
        if optimizer is None or loss_function is None:
            raise ValueError("Optimizer and loss function must be specified for fine-tuning")
        teacher_model = deepcopy(model)

    if layers is None:
        layers = [Linear, Conv2d, ConvTranspose2d]

    match conv_compression_method:
        case "CPD":
            conv_compression_func = cpd_conv2d
        case "TKD":
            conv_compression_func = tkd_conv2d
        case "TKDCPD":
            conv_compression_func = tkd_cpd_conv2d
        case _:
            raise ValueError(f"Unknown compression method: {conv_compression_method}")

    match conv_transpose_compression_method:
        case "CPD":
            conv_transpose_compression_func = cpd_conv_transpose2d
        case "TKD":
            conv_transpose_compression_func = tkd_conv_transpose2d
        case "TKDCPD":
            conv_transpose_compression_func = tkd_cpd_conv_transpose2d
        case _:
            raise ValueError(f"Unknown compression method: {conv_transpose_compression_method}")

    if isinstance(model, (Linear, Conv2d, ConvTranspose2d)):
        if isinstance(model, Conv2d) and Conv2d in layers:
            if model.groups > 1:
                warn("Depthwise convolution is not supported. This layer will not be compressed.")
                return
            if (conv_compression_method == "TKD" or conv_compression_method == "TKDCPD") and rank_tkd is None:
                tensor = model.weight.reshape(
                    model.weight.size()[0], model.weight.size()[1], model.weight.size()[2] * model.weight.size()[3]
                ).detach().numpy()

                method = "differential_evolution"
                try:
                    reconstructed_tensor, weight, factors, optimal_rank, final_loss_value, optimize_result, iteration_logs = (
                        global_optimize_tucker_rank(
                            optimization_method=method,
                            tensor=tensor,
                            target_compression_ratio=target_compression_ratio,
                            frobenius_error_coef=frobenius_error_coef,
                            compression_ratio_coef=compression_ratio_coef,
                            verbose=True,
                        )
                    )
                    rank_tkd = optimal_rank[0:2]
                except Exception as e:
                    print(f"Error with method {method}: {e}")
            model = conv_compression_func(model, rank_cpd, rank_tkd)
        elif isinstance(model, ConvTranspose2d) and ConvTranspose2d in layers:
            if model.groups > 1:
                warn("Depthwise convolution is not supported. This layer will not be compressed.")
                return
            if (conv_transpose_compression_method == "TKD" or conv_transpose_compression_method == "TKDCPD") and rank_tkd is None:
                tensor = model.weight.reshape(
                    model.weight.size()[0], model.weight.size()[1], model.weight.size()[2] * model.weight.size()[3]
                ).detach().numpy()

                method = "differential_evolution"

                try:
                    reconstructed_tensor, weight, factors, optimal_rank, final_loss_value, optimize_result, iteration_logs = (
                        global_optimize_tucker_rank(
                            optimization_method=method,
                            tensor=tensor,
                            target_compression_ratio=target_compression_ratio,
                            frobenius_error_coef=frobenius_error_coef,
                            compression_ratio_coef=compression_ratio_coef,
                            verbose=True,
                        )
                    )
                    rank_tkd = optimal_rank[0:2]
                except Exception as e:
                    print(f"Error with method {method}: {e}")

            model = conv_transpose_compression_func(model, rank_cpd, rank_tkd)
        elif isinstance(model, Linear) and Linear in layers and linear_compress_method != "None":
            model = FactorizedLinear.from_linear(model, factorization=linear_compress_method)
        return model

    for name, child in model.named_children():
        if isinstance(child, Conv2d) and Conv2d in layers:
            if child.groups > 1:
                warn("Depthwise convolution is not supported. This layer will not be compressed.")
                continue
            if (conv_compression_method == "TKD" or conv_compression_method == "TKDCPD") and rank_tkd is None:
                tensor = child.weight.reshape(
                    child.weight.size()[0], child.weight.size()[1], child.weight.size()[2] * child.weight.size()[3]
                ).detach().numpy()

                method = "differential_evolution"

                try:
                    reconstructed_tensor, weight, factors, optimal_rank, final_loss_value, optimize_result, iteration_logs = (
                        global_optimize_tucker_rank(
                            optimization_method=method,
                            tensor=tensor,
                            target_compression_ratio=target_compression_ratio,
                            frobenius_error_coef=frobenius_error_coef,
                            compression_ratio_coef=compression_ratio_coef,
                            verbose=True,
                        )
                    )
                    if child.groups > 1:
                        pass
                    setattr(model, name, conv_compression_func(child, rank_cpd, optimal_rank[0:2]))
                except Exception as e:
                    print(f"Error with method {method}: {e}")
            else:
                setattr(model, name, conv_compression_func(child, rank_cpd, rank_tkd))
        elif isinstance(child, ConvTranspose2d) and ConvTranspose2d in layers:
            if child.groups > 1:
                warn("Depthwise convolution is not supported. This layer will not be compressed.")
                continue
            if (conv_transpose_compression_method == "TKD" or conv_transpose_compression_method == "TKDCPD") and rank_tkd is None:
                tensor = child.weight.reshape(
                    child.weight.size()[0], child.weight.size()[1], child.weight.size()[2] * child.weight.size()[3]
                ).detach().numpy()

                method = "differential_evolution"

                try:
                    start_time = time.perf_counter()
                    reconstructed_tensor, weight, factors, optimal_rank, final_loss_value, optimize_result, iteration_logs = (
                        global_optimize_tucker_rank(
                            optimization_method=method,
                            tensor=tensor,
                            target_compression_ratio=target_compression_ratio,
                            frobenius_error_coef=frobenius_error_coef,
                            compression_ratio_coef=compression_ratio_coef,
                            verbose=True,
                        )
                    )
                    elapsed_time = time.perf_counter() - start_time
                    print(elapsed_time)
                    print(optimal_rank)
                    setattr(model, name, conv_transpose_compression_func(child, rank_cpd, optimal_rank[0:2]))
                except Exception as e:
                    print(f"Error with method {method}: {e}")
            else:
                setattr(model, name, conv_transpose_compression_func(child, rank_cpd, rank_tkd))
        elif isinstance(child, Linear) and Linear in layers and linear_compress_method != "None":
            setattr(model, name, FactorizedLinear.from_linear(child, factorization=linear_compress_method))
        else:
            compress_model(
                child,
                layers,
                conv_compression_method,
                conv_transpose_compression_method,
                linear_compress_method,
                target_compression_ratio,
                frobenius_error_coef,
                compression_ratio_coef,
                rank_cpd,
                rank_tkd,
            )

    if finetune:
        optim = optimizer(model.parameters(), lr=lr)
        teacher_model = teacher_model.to(finetune_device)
        teacher_model.eval()
        criterion = loss_function()
        model = model.to(finetune_device)
        model.train()

        for epoch in range(epochs):
            for _ in range(number_of_iterations):
                inputs = torch.randn(batch_size, *data_size, device=finetune_device)
                with torch.no_grad():
                    reference_outputs = teacher_model(inputs)
                model_outputs = model(inputs)
                if task == "classification":
                    _, target_indices = torch.max(reference_outputs, dim=1)
                    loss = criterion(model_outputs, target_indices)
                elif task == "regression":
                    loss = criterion(model_outputs, reference_outputs)
                elif task == None:
                    raise ValueError("Task must be specified for fine-tuning")
                else:
                    raise ValueError("Unknown task specified")
                optim.zero_grad()
                loss.backward()
                optim.step()
