from pathlib import Path

import json5


class LogReader:
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
