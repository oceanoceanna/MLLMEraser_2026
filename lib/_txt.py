"""Text loading helper."""

from pathlib import Path


def load_txt(filename):
    return [line.strip() for line in Path(filename).read_text(encoding="utf-8").splitlines()]
