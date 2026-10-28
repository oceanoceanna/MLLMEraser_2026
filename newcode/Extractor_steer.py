"""Qwen extractor whose model supports set_steering_parameters()."""

from lib.embeddings import EmbeddingExtractor as BaseEmbeddingExtractor
from lib.embeddings import formulate_prompt_with_options  # noqa: F401
from newcode.qwen_steering import QwenForSteering


class EmbeddingExtractor(BaseEmbeddingExtractor):
    def __init__(self, model_name_or_path, device=None):
        super().__init__(model_name_or_path, device, model_type="qwen", model_class=QwenForSteering)
