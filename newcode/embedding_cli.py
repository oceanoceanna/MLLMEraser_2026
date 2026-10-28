"""Shared CLI for forget/retain embedding extraction."""

import argparse
import logging
import random
from io import BytesIO
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from lib.embeddings import EmbeddingExtractor, formulate_prompt_with_options, image_prompt

ROOT = Path(__file__).resolve().parents[1]


def build_samples(frame, model_type, processor):
    """Collect image-textual classification, generation and cloze prompts."""
    samples = []
    for _, row in frame.iterrows():
        image_data = row["image"]
        source = (
            BytesIO(image_data["bytes"])
            if image_data.get("bytes") is not None
            else image_data["path"]
        )
        with Image.open(source) as opened:
            image = opened.convert("RGB")
        questions = []
        for task in row["Classification_Task"].get("Image_Textual_Questions", []):
            questions.append(
                formulate_prompt_with_options(task["Question"], task["Options"])
                + "\nJust give ONE letter representing the answer directly."
            )
        for task in row["Generation_Task"]:
            if task["Type"] == "Image_Textual":
                questions.append(
                    task["Question"]
                    + "\nAnswer the question based on your trained knowledge in one sentence in ENGLISH."
                )
        for task in row["Mask_Task"]:
            if task["Type"] == "Image_Textual":
                questions.append(
                    task["Question"].replace("__", "[Blank]")
                    + "\nPlease **ONLY** provide the correct answer that should replace the [Blank]."
                )
        samples.extend(
            (image_prompt(question, model_type, processor), image) for question in questions
        )
    return samples


def parse_args(model_type, argv=None):
    parser = argparse.ArgumentParser(description=f"Extract {model_type} decoder-input embeddings")
    parser.add_argument("--dataset", choices=["forget", "retain"], required=True)
    parser.add_argument(
        "--forget_ratio", type=int, choices=range(1, 100), default=10, metavar="1..99"
    )
    parser.add_argument(
        "--model_path",
        default=str(ROOT / ("LLaVA_Vanilla" if model_type == "llava" else "Qwen_Vanilla")),
    )
    parser.add_argument("--data_dir", type=Path, default=ROOT / "data" / "MLLMU-Bench")
    parser.add_argument(
        "--output", type=Path, help="Default: data_dir/embeddings/MODEL/DATASET_RATIO.pt"
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--device", default=None, help="Default: cuda when available, otherwise cpu"
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(model_type, argv=None):
    args = parse_args(model_type, argv)
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    folders = (
        [f"forget_{args.forget_ratio}"]
        if args.dataset == "forget"
        else [f"retain_{100 - args.forget_ratio}", "Retain_Set"]
    )
    frames = []
    for folder in folders:
        paths = sorted((args.data_dir / folder).glob("*.parquet"))
        if not paths:
            raise FileNotFoundError(f"No parquet files in {args.data_dir / folder}")
        frames.extend(pd.read_parquet(path) for path in paths)
    extractor = EmbeddingExtractor(args.model_path, args.device, model_type=model_type)
    samples = build_samples(pd.concat(frames, ignore_index=True), model_type, extractor.processor)
    if not samples:
        raise ValueError("The selected data contains no image-textual questions")
    random.Random(args.seed).shuffle(samples)
    prompts, images = map(list, zip(*samples))
    embeddings = extractor.extract_embeddings(
        prompts, images, args.batch_size, range(extractor.num_layers)
    )
    output = (
        args.output
        or args.data_dir / "embeddings" / model_type / f"{args.dataset}_{args.forget_ratio}.pt"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, output)
    logging.getLogger(__name__).info("Saved embeddings %s to %s", tuple(embeddings.shape), output)
