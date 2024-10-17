import tempfile
from io import BytesIO
from pathlib import Path

import numpy as np
import requests  # type: ignore
from PIL import Image


def download_image(image_url, cache_dir=None):
    """
    Загружает изображение и сохраняет его в файл с кешированием.

    :param image_url: URL изображения для загрузки
    :param cache_dir: Директория для кеширования, если не указано - используется временный файл
    :param proxy_url: URL прокси, если необходимо использовать прокси
    :return: Путь до закешированного файла изображения
    """
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        image_name = Path(image_url).name
        cache_image_path = Path(cache_dir) / image_name
    else:
        cache_image_path = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name)

    if cache_image_path.exists():
        print(f"Изображение уже загружено и закешировано: {cache_image_path}")
        return cache_image_path

    response = requests.get(image_url, timeout=10)
    image = Image.open(BytesIO(response.content)).convert("RGB")

    image.save(cache_image_path)
    print(f"Изображение загружено и сохранено: {cache_image_path}")

    return cache_image_path


def extract_image_frames(image_path):
    """
    Преобразует загруженное изображение в массив NumPy.

    :param image_path: Путь до изображения
    :return: Массив изображения NumPy
    """
    image = Image.open(image_path).convert("RGB")

    image_frames = np.array(image, np.uint8)

    return image_frames
