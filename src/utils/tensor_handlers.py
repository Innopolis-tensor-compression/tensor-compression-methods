import numpy as np


def normalize_frames(frame):
    frame = (frame - np.min(frame)) / (np.max(frame) - np.min(frame))
    frame = np.clip(frame * 255, 0, 255)
    return frame.astype(np.uint8)
