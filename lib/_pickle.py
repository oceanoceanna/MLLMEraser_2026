"""Persistence for locally generated Python objects."""

import pickle
from pathlib import Path


def load_from_pickle(file_path):
    path = Path(file_path)
    if not path.exists():
        return None
    with path.open("rb") as source:
        return pickle.load(source)


def save_to_pickle(data, file_path):
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        pickle.dump(data, output, protocol=pickle.HIGHEST_PROTOCOL)
