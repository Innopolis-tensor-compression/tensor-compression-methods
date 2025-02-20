def calculate_tucker_bounds(shape: tuple | list) -> list:
    """
    Calculate the bounds for Tucker ranks of a tensor based on its shape.

    Parameters
    ----------
    shape : tuple[int, ...] | list[int]
        The shape of the tensor as a list or tuple of integers.
        Each element represents the size of the tensor along a corresponding dimension.

    Returns
    -------
    list[tuple[int, int]]
        A list of rank bounds for the Tucker decomposition.
        Each element is a tuple (r_min, r_max), where:
        - r_min is always 1 for all modes except the last.
        - r_max is the upper bound for the Tucker rank along the corresponding mode.

    Example
    -------
    >>> res = calculate_tucker_bounds((3, 4, 5))
    [(1, 3), (1, 4), (1, 5)]

    """
    return [(1, dim) for dim in shape]


def calculate_tt_bounds(shape: tuple | list) -> list:
    """
    Calculate the bounds for TT-ranks of a tensor based on its shape.

    Parameters
    ----------
    shape : tuple[int, ...] | list[int]
        The shape of the tensor as a list or tuple of integers.
        Each element represents the size of the tensor along a corresponding dimension.

    Returns
    -------
    list[tuple[int, int]]
        A list of rank bounds for the Tensor Train (TT) decomposition.
        Each element is a tuple (r_min, r_max), where:
        - r_min is always 1.
        - r_max is the upper bound for the TT-rank at the corresponding position.

    Examples
    --------
    >>> res = calculate_tt_bounds((3, 4, 5))
    [(1, 1), (1, 3), (1, 12), (1, 1)]

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
