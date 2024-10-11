import gc
from contextlib import contextmanager

import numpy as np
import torch


def normalize_frames(frame):
    frame = (frame - np.min(frame)) / (np.max(frame) - np.min(frame))
    frame = np.clip(frame * 255, 0, 255)
    return frame.astype(np.uint8)


def free_cuda_memory() -> None:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()

    for obj in gc.get_objects():
        try:
            if isinstance(obj, torch.Tensor) and obj.is_cuda:
                obj.cpu()
                del obj
        except (RuntimeError, ReferenceError, TypeError):
            pass

    torch.cuda.empty_cache()
    gc.collect()


@contextmanager
def gpu_torch_memory_manager():
    try:
        torch.cuda.synchronize()
        yield
    finally:
        free_cuda_memory()
