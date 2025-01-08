import tempfile
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import mne
import numpy as np
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

        with (
            tqdm(total=total_size, unit="iB", unit_scale=True) as progress_bar,
            cache_mat_path.open("wb") as file,
        ):
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


def get_eegbci_dataset(cache_dir_eeg: str):
    eeg_raw_list: dict[int, dict] = {}
    subjects = list(range(1, 5))  # 110
    runs = list(range(3, 15))  # 15

    for subject in subjects:
        eeg_raw_list[subject] = {}

    for subject in subjects:
        data_path = mne.datasets.eegbci.load_data(subject=subject, runs=runs, path=cache_dir_eeg)
        raw_fnames = data_path

        for run_idx, f in enumerate(raw_fnames, start=3):
            raw = mne.io.read_raw_edf(f, preload=True)

            raw.rename_channels(
                {
                    ch: ch.replace(".", "").upper().replace("Z", "z").replace("FP1", "Fp1").replace("FP2", "Fp2")
                    for ch in raw.ch_names
                }
            )

            eeg_raw_list[subject][run_idx] = raw

    montage_copy = mne.channels.make_standard_montage("biosemi64").copy()

    coords_t9 = np.array([-0.08869014, -0.0, -0.04014873])
    coords_t10 = np.array([0.08869014, 0.0, -0.04014873])

    def edit_montage_dig(montage, old_name, new_name, new_coords=None) -> None:
        if old_name in montage.ch_names:
            idx = montage.ch_names.index(old_name)

            if new_coords is not None:
                montage.dig[idx + 3]["r"] = new_coords
            montage.ch_names[idx] = new_name

    edit_montage_dig(montage_copy, "P9", "T9", coords_t9)
    edit_montage_dig(montage_copy, "P10", "T10", coords_t10)
    edit_montage_dig(montage_copy, "Fpz", "FPz")

    update_info = [
        {"name": "Fp1", "order": 22},
        {"name": "FPz", "order": 23},
        {"name": "Fp2", "order": 24},
        {"name": "AF7", "order": 25},
        {"name": "AF3", "order": 26},
        {"name": "AFz", "order": 27},
        {"name": "AF4", "order": 28},
        {"name": "AF8", "order": 29},
        {"name": "F7", "order": 30},
        {"name": "F5", "order": 31},
        {"name": "F3", "order": 32},
        {"name": "F1", "order": 33},
        {"name": "Fz", "order": 34},
        {"name": "F2", "order": 35},
        {"name": "F4", "order": 36},
        {"name": "F6", "order": 37},
        {"name": "F8", "order": 38},
        {"name": "FT7", "order": 39},
        {"name": "FC5", "order": 1},
        {"name": "FC3", "order": 2},
        {"name": "FC1", "order": 3},
        {"name": "FCz", "order": 4},
        {"name": "FC2", "order": 5},
        {"name": "FC4", "order": 6},
        {"name": "FC6", "order": 7},
        {"name": "FT8", "order": 8},
        {"name": "T9", "order": 43},
        {"name": "T7", "order": 41},
        {"name": "C5", "order": 8},
        {"name": "C3", "order": 9},
        {"name": "C1", "order": 10},
        {"name": "Cz", "order": 11},
        {"name": "C2", "order": 12},
        {"name": "C4", "order": 13},
        {"name": "C6", "order": 14},
        {"name": "T8", "order": 42},
        {"name": "T10", "order": 44},
        {"name": "TP7", "order": 45},
        {"name": "CP5", "order": 15},
        {"name": "CP3", "order": 16},
        {"name": "CP1", "order": 17},
        {"name": "CPz", "order": 18},
        {"name": "CP2", "order": 19},
        {"name": "CP4", "order": 20},
        {"name": "CP6", "order": 21},
        {"name": "TP8", "order": 46},
        {"name": "P7", "order": 47},
        {"name": "P5", "order": 48},
        {"name": "P3", "order": 49},
        {"name": "P1", "order": 50},
        {"name": "Pz", "order": 51},
        {"name": "P2", "order": 52},
        {"name": "P4", "order": 53},
        {"name": "P6", "order": 54},
        {"name": "P8", "order": 55},
        {"name": "PO7", "order": 56},
        {"name": "PO3", "order": 57},
        {"name": "POz", "order": 58},
        {"name": "PO4", "order": 59},
        {"name": "PO8", "order": 60},
        {"name": "O1", "order": 61},
        {"name": "Oz", "order": 62},
        {"name": "O2", "order": 63},
        {"name": "Iz", "order": 64},
    ]

    def update_montage_points_with_offset(montage, update_info, offset=3) -> None:
        for item in update_info:
            name = item["name"]
            order = item["order"] - 1 + offset

            if order < 0 or order >= len(montage.dig):
                raise IndexError(f"Порядковый номер {order} выходит за пределы dig")

            if name not in montage.ch_names:
                montage.ch_names.append(name)
            else:
                idx = montage.ch_names.index(name)
                montage.ch_names[idx] = name

    update_montage_points_with_offset(montage_copy, update_info)

    for raw_list_by_runs in eeg_raw_list.values():
        for raw in raw_list_by_runs.values():
            raw.set_montage(montage_copy)

    desired_annotations = ["T1", "T2"]  # Указываем нужные аннотации
    tmin, tmax = -4.0, 4.0
    baseline = (0.0, 0.0)

    first_raw = next(iter(next(iter(eeg_raw_list.values())).values()))

    n_channels = len(first_raw.info["ch_names"])

    events, event_id = mne.events_from_annotations(first_raw)
    filtered_event_id = {key: value for key, value in event_id.items() if key in desired_annotations}
    epochs = mne.Epochs(
        first_raw,
        events,
        filtered_event_id,
        tmin,
        tmax,
        baseline=baseline,
        preload=True,
    )

    n_times = epochs.get_data().shape[2]

    event_types = sorted(desired_annotations)
    n_events = len(event_types)

    eeg_data_tensor = []
    event_to_index = {event: idx for idx, event in enumerate(event_types)}

    total_epochs = 0

    for subject_idx, runs_dict in eeg_raw_list.items():
        subject_data = []

        for run_idx, raw in runs_dict.items():
            events, event_id = mne.events_from_annotations(raw)
            filtered_event_id = {key: value for key, value in event_id.items() if key in desired_annotations}

            if not filtered_event_id:
                print(f"Subject {subject_idx}, Run {run_idx}: No desired annotations found")
                continue

            epochs = mne.Epochs(
                raw,
                events,
                filtered_event_id,
                tmin,
                tmax,
                baseline=baseline,
                preload=True,
            )

            num_epochs = len(epochs)
            total_epochs += num_epochs
            print(
                f"Subject {subject_idx}, Run {run_idx}: {num_epochs} epochs for {list(filtered_event_id.keys())}",
                end="\n\n",
            )

            run_data = np.zeros((n_events, len(epochs), n_channels, n_times))

            for event_name, event_code in filtered_event_id.items():
                if event_name in event_types:
                    event_idx = event_to_index[event_name]
                    event_epochs = epochs[event_name].get_data()

                    run_data[event_idx, : event_epochs.shape[0], :, :] = event_epochs

            subject_data.append(run_data)

        eeg_data_tensor.append(subject_data)

    return np.array(eeg_data_tensor, dtype=np.float32)


def create_eeg_limo_data_tensor(cache_dir_eeg: str):
    limo_raw_list = {}
    subjects = list(range(2, 5))

    for subject in subjects:
        limo_raw_list[subject] = mne.datasets.limo.load_data(path=cache_dir_eeg, subject=subject)

    # Список типов событий
    event_types = sorted({event for epochs in limo_raw_list.values() for event in epochs.event_id})

    # Создаём маппинг событий на индексы
    event_to_index = {event: idx for idx, event in enumerate(event_types)}

    # Определяем минимальное количество эпох среди всех субъектов
    min_epochs = min(len(epochs.get_data()) for epochs in limo_raw_list.values())
    n_channels = len(next(iter(limo_raw_list.values())).info["ch_names"])
    n_times = next(iter(limo_raw_list.values())).get_data().shape[2]

    eeg_limo_data_tensor = []  # Список данных для всех субъектов

    # Создание данных для каждого субъекта
    for subject_idx, epochs in limo_raw_list.items():
        epoch_objs = epochs.get_data()[:min_epochs]  # Убираем лишние эпохи
        run_data = np.zeros(
            (min_epochs, len(event_types), n_channels, n_times),
            dtype=np.float32,
        )

        # Заполняем тензор по событиям
        for event_name, event_code in epochs.event_id.items():
            if event_name in event_types:
                event_idx = event_to_index[event_name]

                # Заполняем данные для всех эпох
                for epoch_idx, epoch_data in enumerate(epoch_objs):
                    run_data[epoch_idx, event_idx, :, :] = epoch_data

        eeg_limo_data_tensor.append(run_data)

    # Преобразуем в numpy массив и возвращаем
    eeg_limo_data_tensor = np.array(eeg_limo_data_tensor, dtype=np.float32)
    return eeg_limo_data_tensor
