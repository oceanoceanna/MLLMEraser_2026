"""Shared MLLMU datasets and multimodal training collators."""

import json
import warnings
from io import BytesIO
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset


def json2token(obj: Any, sort_json_key: bool = True):
    """Serialize structured question/answer fields using the existing token format."""
    if isinstance(obj, dict):
        if len(obj) == 1 and "text_sequence" in obj:
            return obj["text_sequence"]
        keys = sorted(obj, reverse=True) if sort_json_key else obj
        return "".join(f"<s_{key}>{json2token(obj[key], sort_json_key)}</s_{key}>" for key in keys)
    if isinstance(obj, list):
        return "<sep/>".join(json2token(item, sort_json_key) for item in obj)
    return str(obj)


def load_image(value):
    """Decode a Parquet/Hugging Face image cell, path, bytes, or PIL image."""
    if value is None:
        return None
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        value = value.get("bytes") if value.get("bytes") is not None else value.get("path")
    if isinstance(value, (bytes, bytearray, memoryview)):
        value = BytesIO(bytes(value))
    if value is None:
        raise ValueError("Image cell has neither bytes nor a path")
    with Image.open(value) as image:
        return image.convert("RGB")


def load_metadata(value):
    metadata = json.loads(value) if isinstance(value, str) else value
    if not isinstance(metadata, (list, tuple)) or not all(isinstance(qa, dict) for qa in metadata):
        raise ValueError("metadata must be a list of question/answer objects")
    return metadata


class LLAVA_multimodal_Dataset(Dataset):
    def __init__(self, df, target_size=None, sort_json_key: bool = True):
        self.df = df
        self.target_size = target_size
        self.sort_json_key = sort_json_key
        self.dataset = self.flatten_dataset()

    def keep_question(self, question):
        return True

    def flatten_dataset(self):
        flattened_data = []
        for index, row in self.df.iterrows():
            try:
                image = load_image(row["image"])
                metadata = load_metadata(row["metadata"])
            except (OSError, TypeError, ValueError) as error:
                warnings.warn(f"Skipping invalid row {index}: {error}", stacklevel=2)
                continue
            for qa_pair in metadata:
                question = qa_pair.get("Question", "")
                answer = qa_pair.get("Answer", "")
                if question and answer and self.keep_question(question):
                    flattened_data.append({"image": image, "question": question, "answer": answer})
        return flattened_data

    def resize_image(self, image):
        if image is not None and self.target_size is not None:
            return image.resize(self.target_size, Image.Resampling.LANCZOS)
        return image

    json2token = staticmethod(json2token)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        return {
            "image": self.resize_image(sample["image"]),
            "question": self.json2token(sample["question"], self.sort_json_key),
            "answer": self.json2token(sample["answer"], self.sort_json_key),
        }


class LLAVA_customized_Dataset(LLAVA_multimodal_Dataset):
    """Keep the historical question filter while sharing image and JSON handling."""

    def keep_question(self, question):
        return "is" in question


class Vanilla_LLaVA_Dataset(LLAVA_multimodal_Dataset):
    """Backward-compatible name for the same Parquet QA dataset."""


def _image_token_ids(processor):
    token_ids = set()
    tokenizer = processor.tokenizer
    for owner in (processor, tokenizer):
        for attribute in ("image_token_id", "video_token_id"):
            token_id = getattr(owner, attribute, None)
            if isinstance(token_id, int):
                token_ids.add(token_id)
        for attribute in ("image_token", "video_token"):
            token = getattr(owner, attribute, None)
            if isinstance(token, str):
                token_id = tokenizer.convert_tokens_to_ids(token)
                if token_id is not None and token_id != tokenizer.unk_token_id:
                    token_ids.add(token_id)
    return token_ids


def _answer_span(full_ids, empty_ids):
    """Locate the inserted assistant answer, accounting for contextual tokenization.

    Both sequences contain the same expanded image tokens and assistant template.
    Comparing to an empty assistant avoids matching answer text in the question or
    assuming that every tokenizer inserts a BOS token. Boundary-merged tokens are
    included; the shared assistant suffix (e.g. EOS) is excluded.
    """
    start = 0
    while start < min(len(full_ids), len(empty_ids)) and full_ids[start] == empty_ids[start]:
        start += 1
    end = len(full_ids)
    empty_end = len(empty_ids)
    while end > start and empty_end > start and full_ids[end - 1] == empty_ids[empty_end - 1]:
        end -= 1
        empty_end -= 1
    if end <= start:
        raise ValueError("Assistant answer contains no trainable tokens")
    return start, end


def _finish_batch(batch, processor, device, train_flag, empty_batch=None, max_length=None):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"].bool()
    if max_length is not None:
        if max_length < 2:
            raise ValueError("max_length must be at least 2")
        if torch.any(attention_mask.sum(dim=1) > max_length):
            raise ValueError(
                f"A processed sequence exceeds max_length={max_length}. Increase the limit; "
                "silently truncating multimodal inputs can remove image or answer tokens."
            )
    labels = input_ids.clone()
    labels[~attention_mask] = -100
    if empty_batch is not None:
        labels.fill_(-100)
        for index in range(len(input_ids)):
            positions = attention_mask[index].nonzero(as_tuple=True)[0]
            full_ids = input_ids[index, positions].tolist()
            empty_ids = empty_batch["input_ids"][index][
                empty_batch["attention_mask"][index].bool()
            ].tolist()
            start, end = _answer_span(full_ids, empty_ids)
            selected = positions[start:end]
            labels[index, selected] = input_ids[index, selected]
    for token_id in _image_token_ids(processor):
        labels[input_ids == token_id] = -100
    if torch.any((labels[:, 1:] != -100).sum(dim=1) == 0):
        raise ValueError("Every example must contain at least one causal target token")
    batch["labels"] = labels
    if device is not None:
        batch = {key: value.to(device) for key, value in batch.items()}
    return dict(batch) if train_flag else tuple(batch.values())


def collate_multimodal(
    examples, processor, device=None, train_flag=True, *, answer_only=False, max_length=None
):
    if not examples:
        raise ValueError("Cannot collate an empty batch")
    images, texts, empty_texts = [], [], []
    for example in examples:
        question, answer = example.get("question"), example.get("answer")
        if not isinstance(question, str) or not isinstance(answer, str) or not answer.strip():
            raise ValueError("Each example requires a string question and nonempty string answer")
        content = []
        if example.get("image") is not None:
            content.append({"type": "image"})
            images.append(example["image"])
        content.append({"type": "text", "text": question})
        messages = [
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ]
        texts.append(
            processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        )
        if answer_only:
            messages[-1]["content"][0]["text"] = ""
            empty_texts.append(
                processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            )
    kwargs = dict(
        images=images or None,
        padding=True,
        truncation=False,
        add_special_tokens=False,
        return_tensors="pt",
    )
    batch = processor(text=texts, **kwargs)
    empty_batch = processor(text=empty_texts, **kwargs) if answer_only else None
    return _finish_batch(batch, processor, device, train_flag, empty_batch, max_length)


def train_collate_fn_mllmu(examples, processor, device=None, train_flag=True, max_length=None):
    return collate_multimodal(examples, processor, device, train_flag, max_length=max_length)


def train_collate_mllmu_ansonly(examples, processor, device=None, train_flag=True, max_length=None):
    return collate_multimodal(
        examples, processor, device, train_flag, answer_only=True, max_length=max_length
    )


def train_collate_fn_llava(examples, processor, train_flag=True, max_length=None):
    """Legacy raw LLaVA prompt format; new callers should use chat templates."""
    if not examples or any(example.get("image") is None for example in examples):
        raise ValueError("Legacy LLaVA batches require an image for every example")
    texts = [
        f"USER: <image>\n{example['question']}\nASSISTANT: {example['answer']}"
        for example in examples
    ]
    batch = processor(
        text=texts,
        images=[example["image"] for example in examples],
        padding=True,
        truncation=False,
        return_tensors="pt",
    )
    return _finish_batch(batch, processor, None, train_flag, max_length=max_length)
