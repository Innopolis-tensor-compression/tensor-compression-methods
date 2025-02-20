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
