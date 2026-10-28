"""JSON persistence helpers."""

import json
from pathlib import Path


def load_json(file_path):
    with Path(file_path).open(encoding="utf-8") as source:
        return json.load(source)


def save_to_json(data_dict, save_path):
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        json.dump(data_dict, output, indent=4, ensure_ascii=False)
