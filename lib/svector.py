"""Optional PCA steering-vector utilities; no external lixiaolong package required."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm


@dataclass
class ContrastivePair:
    positive: str
    negative: str


class SteeringDataset:
    def __init__(self, tokenizer, examples, suffixes=None):
        self.tokenizer = tokenizer
        pairs = [ContrastivePair(f"USER: {a}", f"USER: {b}") for a, b in examples]
        self.formatted_dataset_pre_populated = pairs
        self.formatted_dataset = []
        self.suffixes = [] if suffixes else None
        if suffixes:
            for positive_suffix, negative_suffix in suffixes:
                for pair in pairs:
                    self.formatted_dataset.append(
                        ContrastivePair(
                            pair.positive + "\nASSISTANT: " + positive_suffix,
                            pair.negative + "\nASSISTANT: " + negative_suffix,
                        )
                    )
                    self.suffixes.append((positive_suffix, negative_suffix))
        else:
            self.formatted_dataset = list(pairs)


def get_model_layer_list(model):
    """Find decoder layers in LLaVA, Qwen or a wrapper exposing .model."""
    candidates = [model]
    seen = set()
    while candidates:
        current = candidates.pop(0)
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(getattr(current, "layers", None), torch.nn.ModuleList):
            return current.layers
        candidates.extend(
            getattr(current, name) for name in ("language_model", "model") if hasattr(current, name)
        )
    raise TypeError("Cannot locate the model's decoder layers")


@dataclass
class SteeringVector:
    model_type: str
    directions: dict[int, np.ndarray]
    explained_variances: dict[int, float]

    @classmethod
    def train(cls, model, tokenizer, steering_dataset, **kwargs):
        directions, variances = read_representations(
            model,
            tokenizer,
            steering_dataset.formatted_dataset,
            suffixes=steering_dataset.suffixes,
            **kwargs,
        )
        return cls(model.config.model_type, directions, variances)

    def save(self, file_path):
        path = Path(
            str(file_path) if str(file_path).endswith(".svec") else str(file_path) + ".svec"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "model_type": self.model_type,
            "directions": {k: v.tolist() for k, v in self.directions.items()},
            "explained_variances": {k: float(v) for k, v in self.explained_variances.items()},
        }
        path.write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, file_path):
        path = Path(
            str(file_path) if str(file_path).endswith(".svec") else str(file_path) + ".svec"
        )
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            data["model_type"],
            {int(k): np.asarray(v) for k, v in data["directions"].items()},
            {int(k): float(v) for k, v in data["explained_variances"].items()},
        )


def read_representations(
    model,
    tokenizer,
    inputs,
    hidden_layer_ids=None,
    batch_size=32,
    method="pca_center",
    accumulate_last_x_tokens=1,
    suffixes=None,
):
    """Fit one PCA direction per decoder output; orient toward positive examples.

    This optional PCA utility uses decoder OUTPUTS (hidden_states[l + 1]); it is
    independent of the main null-space pipeline, which uses decoder INPUTS.
    """
    if not inputs:
        raise ValueError("At least one contrastive pair is required")
    count = len(get_model_layer_list(model))
    layers = list(range(count)) if hidden_layer_ids is None else list(hidden_layer_ids)
    layers = [i if i >= 0 else count + i for i in layers]
    if not layers or any(i < 0 or i >= count for i in layers) or len(set(layers)) != len(layers):
        raise ValueError("Decoder layer indices must be nonempty, unique and in range")
    texts = [s for pair in inputs for s in (pair.positive, pair.negative)]
    hiddens = batched_get_hiddens(
        model, tokenizer, texts, layers, batch_size, accumulate_last_x_tokens, suffixes
    )
    directions, variances = {}, {}
    for layer in layers:
        hidden = hiddens[layer]
        if method == "pca_diff":
            training = hidden[::2] - hidden[1::2]
        elif method == "pca_center":
            training = hidden.copy()
            center = (hidden[::2] + hidden[1::2]) / 2
            training[::2] -= center
            training[1::2] -= center
        else:
            raise ValueError(f"Unknown method: {method}")
        if len(training) < 2 or not np.any(np.var(training, axis=0) > 0):
            raise ValueError("PCA needs at least two non-identical training vectors")
        fitted = PCA(n_components=1).fit(training)
        direction = fitted.components_[0].astype(np.float32)
        projections = project_onto_direction(hidden, direction)
        if np.mean(projections[::2] < projections[1::2]) > np.mean(
            projections[::2] > projections[1::2]
        ):
            direction *= -1
        directions[layer] = direction
        variances[layer] = float(fitted.explained_variance_ratio_[0])
    return directions, variances


def batched_get_hiddens(
    model,
    tokenizer,
    inputs,
    hidden_layer_ids,
    batch_size,
    accumulate_last_x_tokens=1,
    suffixes=None,
):
    if not inputs or batch_size <= 0:
        raise ValueError("Nonempty inputs and positive batch_size required")
    mode = accumulate_last_x_tokens
    if not ((isinstance(mode, int) and mode > 0) or mode in ("all", "suffix-only")):
        raise ValueError("accumulate_last_x_tokens must be positive, 'all', or 'suffix-only'")
    if mode == "suffix-only" and (not suffixes or len(suffixes) not in (1, len(inputs) // 2)):
        raise ValueError("suffix-only needs one shared suffix pair or one per contrastive pair")
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer requires a pad or EOS token for batching")
        tokenizer.pad_token = tokenizer.eos_token
    hiddens = {layer: [] for layer in hidden_layer_ids}
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for start in tqdm(range(0, len(inputs), batch_size), desc="Extracting PCA states"):
                encoded = tokenizer(
                    inputs[start : start + batch_size], padding=True, return_tensors="pt"
                ).to(model.device)
                output = model(**encoded, output_hidden_states=True, use_cache=False)
                for layer in hidden_layer_ids:
                    for offset, hidden in enumerate(
                        output.hidden_states[layer + 1 if layer >= 0 else layer]
                    ):
                        mask = encoded["attention_mask"][offset].to(
                            device=hidden.device, dtype=torch.bool
                        )
                        valid = hidden[mask]
                        if not len(valid):
                            raise ValueError("Cannot average an all-padding sequence")
                        if mode == "suffix-only":
                            absolute = start + offset
                            pair = suffixes[0 if len(suffixes) == 1 else absolute // 2]
                            size = len(
                                tokenizer.encode(pair[absolute % 2], add_special_tokens=False)
                            )
                            if size <= 0:
                                raise ValueError("Suffix cannot be empty")
                        else:
                            size = len(valid) if mode == "all" else mode
                        hiddens[layer].append(valid[-size:].float().mean(dim=0).cpu().numpy())
                del output, encoded
    finally:
        model.train(was_training)
    return {layer: np.stack(values) for layer, values in hiddens.items()}


def project_onto_direction(hidden, direction):
    magnitude = np.linalg.norm(direction)
    if not np.isfinite(magnitude) or magnitude == 0:
        raise ValueError("Direction must have finite nonzero norm")
    return (hidden @ direction) / magnitude
