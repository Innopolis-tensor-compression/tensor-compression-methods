from abc import ABC, abstractmethod
from pathlib import Path

import cv2


class SaveFrames(ABC):
    output_folder = Path(__file__).parent.parent / ".cache" / "output"

    @abstractmethod
    def save_frames(self, *args, **kwargs):
        pass


class SaveFramesFactory:
    @staticmethod
    def get_save_methods(frame_name: str) -> SaveFrames:
        save_methods = {
            "video": SaveFramesAsVideo(),
        }

        if frame_name in save_methods:
            return save_methods[frame_name]
        raise ValueError(f"Unknown frame name: {frame_name}")


class SaveFramesAsVideo(SaveFrames):
    @staticmethod
    def save_frames(name, frames, fps, frame_size):
        output_path = SaveFrames.output_folder / f"{name}.mp4"

        SaveFrames.output_folder.mkdir(parents=True, exist_ok=True)

        width, height = frame_size
        size = (width, height)

        out = cv2.VideoWriter(output_path.__str__(), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

        for frame in frames:
            out.write(frame)

        out.release()
        print(f"Видео сохранено как {output_path}")
