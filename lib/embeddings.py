"""Extract decoder-input states at the last non-padding token.

For N samples, L decoder layers, hidden size D the result is (N, L, D).
Layer l means hidden_states[l], the INPUT of decoder l.
"""

import logging

import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


def formulate_prompt_with_options(question, options):
    return question + "\n" + "\n".join(f"{key}: {value}" for key, value in options.items())


def image_prompt(question, model_type, processor):
    if model_type == "llava":
        return f"USER: <image>\n{question}\nASSISTANT:"
    if model_type != "qwen":
        raise ValueError(f"Unsupported model type: {model_type}")
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}
    ]
    # Match evaluation's exact final prompt token (Qwen templates end in a newline).
    return processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    ).strip()


def last_nonpadding_hidden(hidden, attention_mask):
    """Gather the final real token for either left- or right-padded batches."""
    if hidden.shape[:2] != attention_mask.shape:
        raise ValueError("Hidden states and mask must share batch/sequence dimensions")
    valid = attention_mask.to(device=hidden.device, dtype=torch.bool)
    if not valid.any(dim=1).all():
        raise ValueError("Cannot extract a state from an all-padding sequence")
    positions = torch.arange(hidden.shape[1], device=hidden.device)
    last = positions.expand_as(valid).masked_fill(~valid, -1).max(dim=1).values
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), last]


class EmbeddingExtractor:
    """Load model, config and processor from the same requested checkpoint."""

    def __init__(self, model_name_or_path, device=None, *, model_type="llava", model_class=None):
        from transformers import (
            AutoProcessor,
            LlavaForConditionalGeneration,
            Qwen2_5_VLForConditionalGeneration,
        )

        if model_type not in {"llava", "qwen"}:
            raise ValueError(f"Unsupported model type: {model_type}")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        dtype = torch.float32
        if self.device.type == "cuda":
            dtype = (
                torch.bfloat16
                if model_type == "qwen" and torch.cuda.is_bf16_supported()
                else torch.float16
            )
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer requires a pad or EOS token for batching")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if model_class is None:
            model_class = (
                LlavaForConditionalGeneration
                if model_type == "llava"
                else Qwen2_5_VLForConditionalGeneration
            )
        self.model = model_class.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            device_map=str(self.device),
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.config = self.model.config
        if model_type == "llava":
            self.processor.patch_size = self.config.vision_config.patch_size
            self.processor.vision_feature_select_strategy = (
                self.config.vision_feature_select_strategy
            )
            self.processor.num_additional_image_tokens = 1
        text_config = getattr(self.config, "text_config", self.config)
        self.num_layers = text_config.num_hidden_layers
        logger.info(
            "Loaded %s on %s (%s decoder layers)", model_name_or_path, self.device, self.num_layers
        )

    def extract_embeddings(self, prompts, images, batch_size, layers):
        if len(images) != len(prompts):
            raise ValueError("Each prompt must have exactly one image")
        return self._extract(prompts, batch_size, layers, images=images)

    def extract_embeddings_text(self, prompts, batch_size, layers):
        return self._extract(prompts, batch_size, layers)

    def _extract(self, prompts, batch_size, layers, images=None):
        layers = list(layers)
        if not prompts or batch_size <= 0:
            raise ValueError("Prompts must be nonempty and batch_size must be positive")
        if (
            not layers
            or len(set(layers)) != len(layers)
            or any(i < 0 or i >= self.num_layers for i in layers)
        ):
            raise ValueError(f"layers must be unique decoder indices in [0, {self.num_layers})")
        batches = []
        self.model.eval()
        with torch.inference_mode():
            for start in tqdm(range(0, len(prompts), batch_size), desc="Extracting embeddings"):
                kwargs = dict(
                    text=prompts[start : start + batch_size], padding=True, return_tensors="pt"
                )
                if images is not None:
                    kwargs["images"] = images[start : start + batch_size]
                    inputs = self.processor(**kwargs).to(self.device)
                else:
                    inputs = self.tokenizer(**kwargs).to(self.device)
                outputs = self.model(**inputs, output_hidden_states=True, use_cache=False)
                states = [
                    last_nonpadding_hidden(outputs.hidden_states[i], inputs["attention_mask"]).cpu()
                    for i in layers
                ]
                batches.append(torch.stack(states, dim=1))
                del outputs, inputs, states
        return torch.cat(batches, dim=0)
