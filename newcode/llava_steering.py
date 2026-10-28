"""LLaVA/Llama steering without copying Transformers decoder implementations."""

from transformers import LlamaModel, LlavaForConditionalGeneration
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

if __package__:
    from .steering_utils import initialize_steering, set_layer_steering, set_model_steering
else:
    from steering_utils import initialize_steering, set_layer_steering, set_model_steering


class AlphaLlamaDecoderLayer(LlamaDecoderLayer):
    """Compatibility decoder class; its native forward runs after the hook."""

    def __init__(self, config, layer_idx, steering_matrix=None, strength=0.0):
        super().__init__(config, layer_idx)
        initialize_steering(self)
        self.set_steering_parameters(steering_matrix, strength)

    def set_steering_parameters(self, steering_matrix=None, strength=0.0, device=None):
        set_layer_steering(self, steering_matrix, strength, device)


class LlamaModel_for_Steering(LlamaModel):
    """Standalone Llama model with the same steering API as LLaVA."""

    def __init__(self, config):
        super().__init__(config)
        for layer in self.layers:
            initialize_steering(layer)

    def set_steering_parameters(self, steering_matrix=None, strength=None, device=None):
        set_model_steering(self.layers, steering_matrix, strength, device)


class LlavaForSteering(LlavaForConditionalGeneration):
    """Preserve pretrained modules, parameter names, initialization and caches."""

    def __init__(self, config):
        super().__init__(config)
        for layer in self.language_model.model.layers:
            initialize_steering(layer)

    def set_steering_parameters(self, steering_matrix=None, strength=None):
        set_model_steering(self.language_model.model.layers, steering_matrix, strength)
