"""Qwen2.5-VL steering using native Transformers decoder forwards and caches."""

from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLModel
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLDecoderLayer

if __package__:
    from .steering_utils import initialize_steering, set_layer_steering, set_model_steering
else:
    from steering_utils import initialize_steering, set_layer_steering, set_model_steering


class AlphaQwen2_5_VLDecoderLayer(Qwen2_5_VLDecoderLayer):
    """Compatibility decoder class; its native forward runs after the hook."""

    def __init__(self, config, layer_idx, steering_matrix=None, strength=0.0):
        super().__init__(config, layer_idx)
        initialize_steering(self)
        self.set_steering_parameters(steering_matrix, strength)

    def set_steering_parameters(self, steering_matrix=None, strength=0.0, device=None):
        set_layer_steering(self, steering_matrix, strength, device)


class Qwen2_5_VLModel_for_Steering(Qwen2_5_VLModel):
    """Standalone Qwen decoder with optional input steering."""

    def __init__(self, config):
        super().__init__(config)
        for layer in self.layers:
            initialize_steering(layer)

    def set_steering_parameters(self, steering_matrix=None, strength=None, device=None):
        set_model_steering(self.layers, steering_matrix, strength, device)


class QwenForSteering(Qwen2_5_VLForConditionalGeneration):
    """Attach steering to pretrained layers without rebuilding model weights."""

    def __init__(self, config):
        super().__init__(config)
        for layer in self.model.layers:
            initialize_steering(layer)

    def set_steering_parameters(self, steering_matrix=None, strength=None):
        set_model_steering(self.model.layers, steering_matrix, strength)
