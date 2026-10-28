"""Fit layer-wise steering matrices from saved decoder-input activations."""

import argparse
import logging
import pickle
from pathlib import Path

import torch
from tqdm import tqdm

if __package__:
    from .null_space_util import fit_steering_matrix
else:
    from null_space_util import fit_steering_matrix

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--this_ratio",
        type=float,
        default=0.3,
        help="Fraction of smallest retain singular directions (0 to 1).",
    )
    parser.add_argument(
        "--lambda_reg_para",
        type=float,
        default=0.1,
        help="Nonnegative ridge coefficient; objective uses a sum over samples.",
    )
    parser.add_argument("--model", choices=("llava", "qwen"), default="llava")
    parser.add_argument(
        "--direction",
        type=Path,
        help="Refusal pickle; defaults to refusal_direction/refusal_MODEL.pkl.",
    )
    parser.add_argument(
        "--forget_ratio", type=int, default=10, help="Forget percentage, 1 through 99."
    )
    parser.add_argument("--data_dir", type=Path, default=PROJECT_ROOT / "data" / "MLLMU-Bench")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--layers", type=int, nargs="+", help="Zero-based decoder layers; default: all layers."
    )
    parser.add_argument("--output", type=Path, help="Optional output .pt path.")
    return parser


def validate_embeddings(retain, forget, directions):
    for name, value in (("retain", retain), ("forget", forget)):
        if not isinstance(value, torch.Tensor) or value.ndim != 3:
            raise ValueError(f"{name} embeddings must have shape (samples, layers, hidden_size)")
        if min(value.shape) == 0:
            raise ValueError(f"{name} embeddings cannot contain empty dimensions")
    if retain.shape[1:] != forget.shape[1:]:
        raise ValueError("Retain and forget layer/hidden dimensions must match")
    if directions.shape != retain.shape[1:]:
        raise ValueError(
            f"Refusal directions {tuple(directions.shape)} do not match embeddings "
            f"{tuple(retain.shape[1:])}; regenerate aligned decoder-input artifacts"
        )


def main(argv=None):
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not 0 <= args.this_ratio <= 1:
        raise ValueError("--this_ratio must be between 0 and 1")
    if not 0 <= args.lambda_reg_para < float("inf"):
        raise ValueError("--lambda_reg_para must be finite and nonnegative")
    if not 1 <= args.forget_ratio <= 99:
        raise ValueError("--forget_ratio must be between 1 and 99")
    embeds_dir = args.data_dir / "embeddings" / args.model
    # Keep the complete datasets and output on CPU; only one layer is moved
    # to the solver device.  Avoid three full (layers, d, d) GPU allocations.
    forget = torch.load(
        embeds_dir / f"forget_{args.forget_ratio}.pt", map_location="cpu", weights_only=True
    )
    retain = torch.load(
        embeds_dir / f"retain_{args.forget_ratio}.pt", map_location="cpu", weights_only=True
    )
    direction_path = (
        args.direction or PROJECT_ROOT / "refusal_direction" / f"refusal_{args.model}.pkl"
    )
    logger.info("Loading refusal directions from %s", direction_path)
    with direction_path.open("rb") as handle:
        directions = torch.as_tensor(pickle.load(handle), dtype=torch.float64)
    validate_embeddings(retain, forget, directions)
    num_layers, hidden_size = directions.shape
    layers = list(range(num_layers)) if args.layers is None else args.layers
    if len(set(layers)) != len(layers) or any(layer < 0 or layer >= num_layers for layer in layers):
        raise ValueError(f"--layers must contain unique indices in [0, {num_layers - 1}]")
    matrices = torch.zeros(num_layers, hidden_size, hidden_size, dtype=torch.float32)
    for layer in tqdm(layers, desc="Calculating steering matrices"):
        matrix = fit_steering_matrix(
            retain[:, layer].to(args.device),
            forget[:, layer].to(args.device),
            directions[layer].to(args.device),
            args.this_ratio,
            args.lambda_reg_para,
        )
        matrices[layer].copy_(matrix.to(device="cpu", dtype=torch.float32))
        del matrix
    output = (
        args.output
        or args.data_dir
        / "steering_matrix"
        / f"{args.model}_{args.this_ratio}_{args.lambda_reg_para}_{args.forget_ratio}.pt"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(matrices, output)
    logger.info("Saved steering matrices %s to %s", tuple(matrices.shape), output)


if __name__ == "__main__":
    main()
