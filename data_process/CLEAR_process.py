"""CLEAR dataset adapters using the shared multimodal collators."""

import random

from torch.utils.data import Dataset

from data_process.data_preprocess import collate_multimodal

IMAGE_CAPTION_QUESTIONS = [
    "What can you see in this picture?",
    "Tell me about the content of this image",
    "Can you give a description of the image?",
    "What is depicted in the image?",
    "Explain what you observe in the picture.",
    "Describe the image in detail.",
    "What is the main subject of this image?",
    "Can you describe the scene or objects in the image?",
    "What is happening in this image?",
]
CAPTION_MODE = "CAP"
RECOGNITION_MODE = "REC"
NONE_MODE = "NONE"


class CLEAR_Dataset(Dataset):
    def __init__(self, data, mode=CAPTION_MODE):
        if mode not in (CAPTION_MODE, RECOGNITION_MODE, NONE_MODE):
            raise ValueError(f"Unknown CLEAR mode: {mode}")
        self.data = data
        self.mode = mode
        self.dataset = self.process_dataset()

    def process_dataset(self):
        dataset = []
        # Do not reset the caller's global RNG merely by importing this module.
        rng = random.Random(42)
        for index, sample in enumerate(self.data):
            image = sample.get("image")
            if image is not None and self.mode == NONE_MODE:
                continue
            question = sample.get("question", "")
            if question:
                answer = sample.get("answer", "")
            elif self.mode == CAPTION_MODE:
                question = rng.choice(IMAGE_CAPTION_QUESTIONS)
                answer = sample.get("caption", "")
            elif self.mode == RECOGNITION_MODE:
                question = "The name of the person on the image is "
                answer = sample.get("name", "")
            else:
                raise ValueError(f"CLEAR text sample {index} has no question")
            if not answer:
                raise ValueError(f"CLEAR sample {index} has no answer for mode {self.mode}")
            dataset.append({"image": image, "question": question, "answer": answer})
        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


def train_collate_clear(examples, processor, device=None, train_flag=True, max_length=None):
    return collate_multimodal(examples, processor, device, train_flag, max_length=max_length)


def train_collate_clear_w_example(
    examples, processor, device=None, train_flag=True, max_length=None
):
    return train_collate_clear(examples, processor, device, train_flag, max_length), examples


def train_collate_clear_ansonly(examples, processor, device=None, train_flag=True, max_length=None):
    return collate_multimodal(
        examples, processor, device, train_flag, answer_only=True, max_length=max_length
    )
