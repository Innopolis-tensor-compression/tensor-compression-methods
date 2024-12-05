from abc import ABC, abstractmethod
from pathlib import Path

import cv2


class SaveFrames(ABC):
    output_folder = Path(__file__).parent.parent.parent / ".cache" / "output"

    @abstractmethod
    def save_frames(self, *args, **kwargs):
        pass


class SaveFramesFactory:
    @staticmethod
    def get_save_methods(frame_name: str) -> SaveFrames:
        save_methods = {
            "image": SaveFramesAsImage(),
            "video": SaveFramesAsVideo(),
            "eeg": SaveFramesAsEEG(),
        }

        if frame_name in save_methods:
            return save_methods[frame_name]
        raise ValueError(f"Unknown frame name: {frame_name}")


class SaveFramesAsImage(SaveFrames):
    @staticmethod
    def save_frames(name, frames):
        output_path = SaveFrames.output_folder / f"{name}.jpg"

        SaveFrames.output_folder.mkdir(parents=True, exist_ok=True)

        frames_bgr = cv2.cvtColor(frames, cv2.COLOR_RGB2BGR)

        cv2.imwrite(output_path.__str__(), frames_bgr)

        print(f"Изображение сохранено как {output_path}")


class SaveFramesAsEEG(SaveFrames):
    @staticmethod
    def save_frames(name, frames):  # noqa: ARG004
        print("Зачем сейвить ЭЭГ?")


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
