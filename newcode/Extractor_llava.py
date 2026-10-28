"""llava embedding extractor; stable import path."""

from lib.embeddings import EmbeddingExtractor as BaseEmbeddingExtractor
from lib.embeddings import formulate_prompt_with_options  # noqa: F401


class EmbeddingExtractor(BaseEmbeddingExtractor):
    def __init__(self, model_name_or_path, device=None):
        super().__init__(model_name_or_path, device, model_type="llava")
