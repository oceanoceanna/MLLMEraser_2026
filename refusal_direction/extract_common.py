"""Estimate r_l = mean(h_l^positive) - mean(h_l^negative).

All states are decoder INPUT states, matching the forget/retain embeddings
and the location of steering. JSONL row i is paired with image ID i.
"""

import argparse
import json
import logging
import re
from pathlib import Path

from PIL import Image

from lib._pickle import save_to_pickle
from lib.embeddings import EmbeddingExtractor, image_prompt

ROOT = Path(__file__).resolve().parents[1]
REFUSAL_DIR = Path(__file__).resolve().parent


def image_paths_by_id(folder, *, attack=False):
    pattern = r"bad_prompt_(\d+)_.*" if attack else r"(\d+)"
    paths = {}
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".gif"}:
            continue
        match = re.fullmatch(pattern, path.stem)
        if match is None:
            continue
        index = int(match.group(1))
        if index in paths:
            raise ValueError(f"Duplicate image ID {index} in {folder}")
        paths[index] = path
    return paths


def load_pairs(
    data_path,
    positive_images,
    negative_images,
    num_samples=17,
    positive_key="en",
    negative_key="zh",
):
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    with Path(data_path).open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    if len(rows) < num_samples:
        raise ValueError(f"Requested {num_samples} pairs but JSONL contains {len(rows)}")
    positive_paths = image_paths_by_id(positive_images, attack=True)
    negative_paths = image_paths_by_id(negative_images)
    texts, images = [[], []], [[], []]
    for index, row in enumerate(rows[:num_samples]):
        for group, (key, paths) in enumerate(
            ((positive_key, positive_paths), (negative_key, negative_paths))
        ):
            if index not in paths:
                raise FileNotFoundError(
                    f"Missing image ID {index} for {key}; JSONL/image pairs must align"
                )
            text = row[key].strip()
            if not text:
                raise ValueError(f"Empty {key} text in JSONL row {index}")
            with Image.open(paths[index]) as image:
                images[group].append(image.convert("RGB"))
            texts[group].append(text)
    return texts, images


def parse_args(model_type, argv=None):
    parser = argparse.ArgumentParser(description=f"Extract {model_type} refusal directions")
    parser.add_argument(
        "--model_path",
        default=str(ROOT / ("LLaVA_Vanilla" if model_type == "llava" else "Qwen_Vanilla")),
    )
    parser.add_argument("--data_path", type=Path, default=REFUSAL_DIR / "test.jsonl")
    parser.add_argument("--positive_images", type=Path, default=REFUSAL_DIR / "constrain_16")
    parser.add_argument("--negative_images", type=Path, default=REFUSAL_DIR / "add_images")
    parser.add_argument("--positive_key", default="en")
    parser.add_argument("--negative_key", default="zh")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=17,
        help="First N aligned pairs (original experiment: 17)",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=REFUSAL_DIR / f"refusal_{model_type}.pkl")
    return parser.parse_args(argv)


def main(model_type, argv=None):
    args = parse_args(model_type, argv)
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    texts, images = load_pairs(
        args.data_path,
        args.positive_images,
        args.negative_images,
        args.num_samples,
        args.positive_key,
        args.negative_key,
    )
    extractor = EmbeddingExtractor(args.model_path, args.device, model_type=model_type)
    means = []
    for group in range(2):
        prompts = [image_prompt(text, model_type, extractor.processor) for text in texts[group]]
        states = extractor.extract_embeddings(
            prompts, images[group], args.batch_size, range(extractor.num_layers)
        )
        means.append(states.float().mean(dim=0))
    direction = (means[0] - means[1]).numpy()
    save_to_pickle(direction, args.output)
    logging.getLogger(__name__).info(
        "Saved decoder-input directions %s to %s", direction.shape, args.output
    )
