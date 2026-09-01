"""Pure parameter-sharding policies shared by MLX tensor strategies."""


def should_replicate_tensor_parameter(path: str) -> bool:
    """Return whether a tensor parameter must be replicated across ranks.

    Vector-quantized experts store integer codes that index a shared codebook.
    Sharding that lookup table gives each rank only part of the vocabulary and
    makes otherwise valid codes decode against the wrong vectors.

    Args:
        path: Fully qualified parameter path supplied by MLX sharding.

    Returns:
        True when the parameter is a vector-quantized codebook.
    """
    return path.endswith("codebook")
