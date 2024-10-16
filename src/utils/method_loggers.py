from pathlib import Path
from pprint import pprint
from typing import Any

import json5
from tqdm import tqdm

from src.utils.method_runners import MethodRunner
from src.utils.tensor_handlers import gpu_torch_memory_manager


class MethodLogger:
    experiments_count = 10
    log_dir_path = Path(__file__).parent.parent / ".cache"

    def __init__(self, method_name: str, qualitative_metrics: dict[str, str], runner: MethodRunner, method_args: dict[str, Any]):
        self.name = method_name
        self.method_args: dict[str, Any] = method_args
        self.qualitative_metrics = qualitative_metrics
        self.runner = runner

        self.quantitative_metrics: dict[str, list[float]] = {}
        self.library_method_name = self.runner.library_method_name

        self.run_experiments()

    def run_experiments(self) -> None:
        for _ in tqdm(range(MethodLogger.experiments_count), desc="Эксперимент набора параметров"):
            with gpu_torch_memory_manager():
                self.runner.run(**self.method_args)

            metrics = self.runner.get_metrics(library_method_name=self.library_method_name)

            for metric_name, metric_value in metrics.items():
                if metric_name in self.quantitative_metrics:
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
                with log_file_path.open("r+", encoding="utf-8") as f:
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
