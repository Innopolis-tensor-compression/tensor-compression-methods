import tempfile
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import requests  # type: ignore
from scipy.io import loadmat
from tqdm import tqdm

YANDEX_DISK_API_URL = "https://cloud-api.yandex.net/v1/disk/public/resources/download"


def get_yandex_disk_download_link(public_key: str):
    """
    Получает прямую ссылку для скачивания файла с Яндекс Диска по публичному ключу.

    :param public_key: Публичная ссылка или ключ на файл
    :return: Прямая ссылка для скачивания файла
    """
    try:
        params = {"public_key": public_key}
        response = requests.get(YANDEX_DISK_API_URL, params=params, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Ошибка при получении ссылки для скачивания: {e}")
        return None
    else:
        download_url = response.json()["href"]
        return download_url


def download_mat_file(mat_url: str, cache_dir=None):
    """
    Загружает .mat файл с Яндекс Диска и сохраняет его с кешированием.
    Отображает статус бар загрузки.

    :param mat_url: URL публичного ключа или файла для загрузки
    :param cache_dir: Директория для кеширования, если не указано - используется временный файл
    :return: Путь до закешированного файла .mat
    """
    download_url = get_yandex_disk_download_link(mat_url)
    filename = parse_qs(urlparse(download_url).query).get("filename", download_url)[0]
    if not download_url:
        return None

    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        mat_name = Path(filename).name
        cache_mat_path = Path(cache_dir) / mat_name
    else:
        cache_mat_path = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".mat").name)

    if cache_mat_path.exists():
        print(f"Файл уже загружен и закеширован: {cache_mat_path}")
        return cache_mat_path

    try:
        response = requests.get(download_url, stream=True, timeout=10)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024

        with tqdm(total=total_size, unit="iB", unit_scale=True) as progress_bar, cache_mat_path.open("wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)

    except requests.RequestException as e:
        print(f"Ошибка при загрузке файла .mat: {e}")
        return None

    return cache_mat_path


def extract_mat_data(mat_path: str):
    """
    Извлекает данные из .mat файла.

    :param mat_path: Путь до .mat файла
    :return: Данные из .mat файла
    """
    return loadmat(mat_path)
