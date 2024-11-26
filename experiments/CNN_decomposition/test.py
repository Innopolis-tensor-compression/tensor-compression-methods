import argparse

import torch
import tensorly as tl
from torchvision.models import resnet18
from tensorly.decomposition import tucker, parafac
from flopco import FlopCo
import warnings
import gc

tl.set_backend("pytorch")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

in_channels = 64
out_channels = 128
kernel_size = (3, 3)
tensor_size = 7
# rank_CPD = 9
# rank_TKD = (256, 101, 9)
number_of_images = 128

full_conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, dtype=torch.float32)

random_tensor = torch.rand(number_of_images, in_channels, tensor_size, tensor_size, dtype=torch.float32)

def SVD_conv(conv_layer: torch.nn.Conv2d, rank_CPD: int = None) -> (torch.nn.Sequential, float):
    out_channels = conv_layer.out_channels
    in_channels = conv_layer.in_channels
    stride = conv_layer.stride
    matrix = conv_layer.weight.squeeze().squeeze()
    if rank_CPD is None:
        rank_CPD = min(matrix.shape)

    core, factors = parafac(matrix, rank_CPD, init="random")
    norm = tl.norm(matrix - tl.cp_to_tensor((core, factors))) / tl.norm(matrix)
    print(f"SVD ({in_channels}, {out_channels}, (1, 1)): {norm}")

    factor_CPD_input = factors[1].permute([1, 0]).unsqueeze(2).unsqueeze(3)
    factor_CPD_output = factors[0].unsqueeze(2).unsqueeze(3)


    conv1 = torch.nn.Conv2d(in_channels, rank_CPD, 1, stride=stride, dtype=torch.float32)
    conv2 = torch.nn.Conv2d(rank_CPD, out_channels, 1, dtype=torch.float32)
    conv1.weight = torch.nn.parameter.Parameter(factor_CPD_input)
    conv2.weight = torch.nn.parameter.Parameter(factor_CPD_output)
    return torch.nn.Sequential(conv1, conv2), norm

def CPD_conv(conv_layer: torch.nn.Conv2d, rank_CPD: int = None) -> (torch.nn.Sequential, float):
    if conv_layer.kernel_size == (1, 1):
        return SVD_conv(conv_layer, rank_CPD)
    # Params of source conv_layer
    out_channels = conv_layer.out_channels
    in_channels = conv_layer.in_channels
    kernel_size_x = conv_layer.kernel_size[0]
    kernel_size_y = conv_layer.kernel_size[1]
    stride = conv_layer.stride
    padding = conv_layer.padding
    dilation = conv_layer.dilation
    conv_weight = conv_layer.weight.reshape(out_channels, in_channels, kernel_size_x * kernel_size_y)

    if rank_CPD is None:
        rank_CPD = sorted(conv_weight.size())[0]
    # elif rank_CPD > sorted(conv_weight.size())[0]:
    #     rank_CPD = sorted(conv_weight.size())[0]
    #     warnings.warn(
    #         f"""
    #         rank_CPD > min(f{conv_weight.size()})
    #         rank_CPD is bigger than the smallest size of tensor dimension
    #         rank_CPD is set to min(f{conv_weight.size()})
    #         """
    #     )

    core_CPD, factors_CPD = parafac(conv_weight, rank_CPD, init="random", svd="randomized_svd")
    norm = tl.norm(conv_weight - tl.cp_to_tensor((core_CPD, factors_CPD))) / tl.norm(conv_weight)
    print(f"CPD ({in_channels}, {out_channels}, ({kernel_size_x}, {kernel_size_y})): {norm}")

    factor_CPD_input = factors_CPD[1].permute([1, 0]).unsqueeze(2).unsqueeze(3)
    factor_CPD_hidden = factors_CPD[2].permute([1, 0]).unsqueeze(1).reshape(rank_CPD, 1, kernel_size_x, kernel_size_y)
    factor_CPD_output = factors_CPD[0].unsqueeze(2).unsqueeze(3)

    conv1_CPD = torch.nn.Conv2d(in_channels, rank_CPD, 1, dtype=torch.float32)
    conv2_CPD = torch.nn.Conv2d(rank_CPD, rank_CPD, (kernel_size_x, kernel_size_y), groups=rank_CPD, stride=stride, padding=padding, dilation=dilation, dtype=torch.float32)
    conv3_CPD = torch.nn.Conv2d(rank_CPD, out_channels, 1, dtype=torch.float32)
    conv1_CPD.weight = torch.nn.parameter.Parameter(factor_CPD_input)
    conv2_CPD.weight = torch.nn.parameter.Parameter(factor_CPD_hidden)
    conv3_CPD.weight = torch.nn.parameter.Parameter(factor_CPD_output)

    return torch.nn.Sequential(conv1_CPD, conv2_CPD, conv3_CPD), norm

def TKD_conv(conv_layer: torch.nn.Conv2d, rank_TKD: tuple[int, int, int]=None) -> (torch.nn.Sequential, float):
    if conv_layer.kernel_size == (1, 1):
        return SVD_conv(conv_layer, min(rank_TKD))
    # Params of source conv_layer
    out_channels = conv_layer.out_channels
    in_channels = conv_layer.in_channels
    kernel_size_x = conv_layer.kernel_size[0]
    kernel_size_y = conv_layer.kernel_size[1]
    stride = conv_layer.stride
    padding = conv_layer.padding
    dilation = conv_layer.dilation
    conv_weight = conv_layer.weight.reshape(out_channels, in_channels, kernel_size_x * kernel_size_y)

    if rank_TKD is None:
        rank_TKD = (out_channels, in_channels, kernel_size_x * kernel_size_y)
    else:
        if rank_TKD[0] > out_channels:
            rank_TKD = (out_channels, rank_TKD[1], rank_TKD[2])
            warnings.warn("rank_TKD[0] is bigger then out_channels")
        if rank_TKD[1] > in_channels:
            rank_TKD = (rank_TKD[0], in_channels, rank_TKD[2])
            warnings.warn("rank_TKD[1] is bigger then in_channels")
        if rank_TKD[2] > kernel_size_x * kernel_size_y:
            rank_TKD = (rank_TKD[0], rank_TKD[1], kernel_size_x * kernel_size_y)
            warnings.warn("rank_TKD[2] is bigger then kernel_size_x * kernel_size_y")

    core_TKD, factors_TKD = tucker(conv_weight, rank_TKD)
    norm = tl.norm(conv_weight - tl.tucker_to_tensor((core_TKD, factors_TKD))) / tl.norm(conv_weight)
    print(f"TKD ({in_channels}, {out_channels}, ({kernel_size_x}, {kernel_size_y})): {norm}")

    factor_TKD_input = factors_TKD[1].permute([1, 0]).unsqueeze(2).unsqueeze(3)
    factor_TKD_hidden = torch.tensordot(factors_TKD[2], core_TKD, dims=([1], [2])).permute([1, 2, 0]).reshape(rank_TKD[0], rank_TKD[1], kernel_size_x, kernel_size_y)
    factor_TKD_output = factors_TKD[0].unsqueeze(2).unsqueeze(3)

    conv1_TKD = torch.nn.Conv2d(in_channels, rank_TKD[1], 1, dtype=torch.float32)
    conv2_TKD = torch.nn.Conv2d(rank_TKD[1], rank_TKD[0], (kernel_size_x, kernel_size_y), stride=stride, padding=padding, dilation=dilation, dtype=torch.float32)
    conv3_TKD = torch.nn.Conv2d(rank_TKD[0], out_channels, 1, dtype=torch.float32)
    conv1_TKD.weight = torch.nn.parameter.Parameter(factor_TKD_input)
    conv2_TKD.weight = torch.nn.parameter.Parameter(factor_TKD_hidden)
    conv3_TKD.weight = torch.nn.parameter.Parameter(factor_TKD_output)

    return torch.nn.Sequential(conv1_TKD, conv2_TKD, conv3_TKD), norm

def TKDCPD_conv(conv_layer: torch.nn.Conv2d, rank_TKD:tuple[int, int, int] = None, rank_CPD: int = None) -> (torch.nn.Sequential, float):
    if conv_layer.kernel_size == (1, 1):
        return SVD_conv(conv_layer, rank_CPD)
    # Params of source conv_layer
    out_channels = conv_layer.out_channels
    in_channels = conv_layer.in_channels
    kernel_size_x = conv_layer.kernel_size[0]
    kernel_size_y = conv_layer.kernel_size[1]
    stride = conv_layer.stride
    padding = conv_layer.padding
    dilation = conv_layer.dilation
    conv_weight = conv_layer.weight.reshape(out_channels, in_channels, kernel_size_x * kernel_size_y)

    if rank_TKD is None:
        rank_TKD = (out_channels, in_channels, kernel_size_x * kernel_size_y)
    else:
        if rank_TKD[0] > out_channels:
            rank_TKD = (out_channels, rank_TKD[1], rank_TKD[2])
            warnings.warn(f"rank_TKD[0] is bigger then out_channels\n\nrank_TKD[0]={rank_TKD[0]}\nout_channels={out_channels}")
        if rank_TKD[1] > in_channels:
            rank_TKD = (rank_TKD[0], in_channels, rank_TKD[2])
            warnings.warn(f"rank_TKD[1] is bigger then in_channels\n\nrank_TKD[1]={rank_TKD[1]}\nin_channels={in_channels}")
        if rank_TKD[2] > kernel_size_x * kernel_size_y:
            rank_TKD = (rank_TKD[0], rank_TKD[1], kernel_size_x * kernel_size_y)
            warnings.warn(f"rank_TKD[2] is bigger then kernel_size_x * kernel_size_y\nrank_TKD[2]={rank_TKD[2]}\nkernel_size_x * kernel_size_y={kernel_size_x * kernel_size_y}")

    core_TKD, factors_TKD = tucker(conv_weight, rank_TKD)
    norm = tl.norm(conv_weight - tl.tucker_to_tensor((core_TKD, factors_TKD))) / tl.norm(conv_weight)
    print(f"TKDCPD ({in_channels}, {out_channels}, ({kernel_size_x}, {kernel_size_y})): {norm}")

    factor_TKD_input = factors_TKD[1].permute([1, 0]).unsqueeze(2).unsqueeze(3)
    factor_TKD_hidden = torch.tensordot(factors_TKD[2], core_TKD, dims=([1], [2])).permute([1, 2, 0]).reshape(rank_TKD[0], rank_TKD[1], kernel_size_x, kernel_size_y)
    factor_TKD_output = factors_TKD[0].unsqueeze(2).unsqueeze(3)

    conv2_TKD = torch.nn.Conv2d(rank_TKD[1], rank_TKD[0], (kernel_size_x, kernel_size_y), stride=stride, padding=padding, dilation=dilation, dtype=torch.float32)
    conv2_TKD.weight = torch.nn.parameter.Parameter(factor_TKD_hidden)
    conv2_TKD = CPD_conv(conv2_TKD)
    norm = conv2_TKD[1]
    conv2_TKD = conv2_TKD[0]

    conv1_TKD = torch.nn.Conv2d(in_channels, rank_TKD[1], 1, dtype=torch.float32)
    conv3_TKD = torch.nn.Conv2d(rank_TKD[0], out_channels, 1, dtype=torch.float32)
    conv1_TKD.weight = torch.nn.parameter.Parameter(factor_TKD_input)
    conv3_TKD.weight = torch.nn.parameter.Parameter(factor_TKD_output)

    return torch.nn.Sequential(conv1_TKD, conv2_TKD, conv3_TKD), norm

def compress_resnet(resnet, conv_func, rank: int = None):
    layer_norms = {}
    for name, module in resnet.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            parent_name = ".".join(name.split(".")[:-1])
            attr_name = name.split(".")[-1]
            print(parent_name, end=": ")
            # Access the parent module
            parent_module = resnet
            if parent_name:
                parent_module = dict(resnet.named_modules())[parent_name]

            # Replace the old layer with the new one
            result = conv_func(module, rank_CPD=rank)
            setattr(parent_module, attr_name, result[0])
            layer_norms[parent_name + "." + attr_name] = result[1]
            del result
            gc.collect()
            torch.cuda.empty_cache()
    return resnet, layer_norms

import os
import json

def save_results(conv_func_name, rank, stats, norms):
    # Define the directory structure
    dir_path = f"./{conv_func_name}/{rank}/"

    # Create the directory if it doesn't exist
    os.makedirs(dir_path, exist_ok=True)

    # Save FLOPS and parameter counts
    stats_data = {
        "total_flops": stats.total_flops,
        "total_params": stats.total_params,
        "relative_flops": stats.relative_flops,
        "relative_params": stats.relative_params
    }
    with open(f"{dir_path}flops_params.json", "w") as f:
        json.dump(stats_data, f, indent=4)

    # Save norms of each layer
    with open(f"{dir_path}layer_norms.txt", "w") as f:
        for layer_name, norm_value in norms.items():
            f.write(f"{layer_name}: {norm_value}\n")
    print(f"Results saved in {dir_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='ProgramName',
        description='What the program does',
        epilog='Text at the bottom of help')
    parser.add_argument('--rank', type=int)

    rank = parser.parse_args().rank

    # Define the range of ranks you want to iterate over
    rank_range = range(rank, rank+1)

    # Initialize your ResNet model
    resnet_model = resnet18(weights='DEFAULT')

    # Loop over the specified ranks and compress the ResNet with each rank
    for rank in rank_range:
        print(f"Compressing with rank_CPD = {rank}")

        # Clone the model to avoid overwriting the original
        model_copy = resnet18(weights='DEFAULT')

        # Compress the model, storing norms of each layer
        compressed_model, layer_norms = compress_resnet(model_copy, CPD_conv, rank)
        compressed_model = compressed_model.cpu()

        # Capture statistics with FlopCo
        stats_CPD = FlopCo(compressed_model, device="cpu")

        # Save the results
        save_results("CPD_conv", rank, stats_CPD, layer_norms)

        # Clear cache
        del compressed_model
        torch.cuda.empty_cache()
        gc.collect()
