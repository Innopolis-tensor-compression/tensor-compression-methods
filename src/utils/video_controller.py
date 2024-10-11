import re
import tempfile
from pathlib import Path

import cv2
import numpy as np
import yt_dlp


def download_progress_hook(d):
    if d["status"] == "downloading":
        print(f"Downloading: {d['_percent_str']} at {d['_speed_str']} ETA: {d['_eta_str']}")
    elif d["status"] == "finished":
        print("Download complete!")


def extract_video_id(video_url: str):
    video_id_match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", video_url)
    if video_id_match:
        return video_id_match.group(1)
    error_message = "Не удалось извлечь ID видео из URL"
    raise ValueError(error_message)


def download_youtube_video(video_url, cache_dir=None, proxy_url=None):
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        video_id = extract_video_id(video_url)
        cache_video_path = Path(cache_dir) / f"{video_id}.mp4"
    else:
        cache_video_path = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name)

    if cache_video_path.exists():
        print(f"Видео уже загружено и закешировано: {cache_video_path}")
        return cache_video_path

    ydl_opts = {
        "format": "134",
        "outtmpl": str(cache_video_path),
        "progress_hooks": [download_progress_hook],
    }

    if proxy_url:
        ydl_opts["proxy"] = proxy_url

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

    print(f"Видео загружено и сохранено: {cache_video_path}")
    return cache_video_path


def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    return np.array(frames), fps, (width, height)


def process_frames(frames):
    processed_frames = []

    for frame in frames:
        b_channel, g_channel, r_channel = cv2.split(frame)

        merged_frame = cv2.merge((b_channel, g_channel, r_channel))

        processed_frames.append(merged_frame)

    return np.array(processed_frames)


def show_frames_as_video(frames):
    for frame in frames:
        cv2.imshow("Downloaded Video", frame)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()
