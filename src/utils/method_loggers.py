import gc
from pathlib import Path
from pprint import pprint
from typing import Any

import json5
import numpy as np
import torch
from numba import cuda
from tqdm import tqdm

from src.utils.method_runners import MethodRunner


class MethodLogger:
    experiments_count = 1
    log_dir_path = Path("../../.cache")

    def __init__(
        self, method_name: str, method_input_tensor: np.ndarray, qualitative_metrics: dict[str, str], runner: MethodRunner, method_args: dict[str, Any]
    ):
        self.name = method_name
        self.method_args: dict[str, Any] = method_args
        self.qualitative_metrics = qualitative_metrics

        self.method_input_tensor = method_input_tensor
        self.runner = runner

        self.quantitative_metrics: dict[str, list[float]] = {}

    def run_experiments(self, *args, **kwargs):
        for _ in tqdm(range(MethodLogger.experiments_count), desc="Эксперимент набора параметров"):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            cuda.get_current_device().reset()
            gc.collect()

            self.runner.run(*args, **kwargs)

            metrics = self.runner.get_metrics()
            for metric_name, metric_value in metrics.items():
                if metric_name in self.qualitative_metrics:
                    self.quantitative_metrics[metric_name].append(metric_value)
                else:
                    self.quantitative_metrics[metric_name] = [metric_value]

    def save_logs_to_file(self, is_test: bool = False):
        method_logs = {
            "method_name": self.name,
            "method_args": dict(self.method_args.copy()),
            "qualitative_metrics": self.qualitative_metrics.copy(),
            "quantitative_metrics": self.quantitative_metrics.copy(),
        }

        excluded_data = ["tensor", "input_tensor", "tens", "sites"]
        method_logs["method_args"] = {k: v for k, v in method_logs["method_args"].items() if k not in excluded_data}  # type: ignore

        if is_test:
            pprint(method_logs)
        else:
            MethodLogger.log_dir_path.mkdir(parents=True, exist_ok=True)

            log_file_path = MethodLogger.log_dir_path / "method_logs.json"

            if not log_file_path.exists() or log_file_path.stat().st_size == 0:
                with log_file_path.open("w+", encoding="utf-8") as f:
                    json5.dump([method_logs], f, ensure_ascii=False, indent=4)
            else:
                with log_file_path.open("w+", encoding="utf-8") as f:
                    logs = json5.load(f)

                    existing_log = next(
                        (log for log in logs if log["method_name"] == method_logs["method_name"] and log["method_args"] == method_logs["method_args"]),
                        None,
                    )

                    if existing_log:
                        existing_log.update(method_logs)
                    else:
                        logs.append(method_logs)

                    f.seek(0)
                    json5.dump(logs, f, ensure_ascii=False, indent=4)
                    f.truncate()

    @classmethod
    def load_logs_from_file(cls, log_file_path):
        log_file = Path(log_file_path)
        if log_file.exists():
            with log_file.open(encoding="utf-8") as f:
                logs = json5.load(f)
                # print(f"Загруженные логи:")
                # pprint(logs, indent=4, width=100)
                return logs
        else:
            print(f"Файл {log_file_path} не найден.")
            return None
