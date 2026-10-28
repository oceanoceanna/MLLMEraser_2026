"""Shared decoder-input steering hooks, independent of model architecture."""

import inspect
import math
from numbers import Real

import torch


def initialize_steering(layer):
    """Attach a nonpersistent matrix buffer and preserve the native forward."""
    layer.register_buffer("steering_matrix", None, persistent=False)
    layer.strength = 0.0
    layer._steering_forward_signature = inspect.signature(layer.forward)
    layer.register_forward_pre_hook(_steer_decoder_input, with_kwargs=True)


def _execution_device(layer):
    """Resolve Accelerate's execution target, including composed hooks."""
    hooks = [getattr(layer, "_hf_hook", None)]
    while hooks:
        hook = hooks.pop(0)
        target = getattr(hook, "execution_device", None)
        if target is not None:
            return target
        hooks.extend(getattr(hook, "hooks", ()))
    return None


def set_layer_steering(layer, steering_matrix=None, strength=0.0, device=None):
    """Replace or clear a layer's steering parameters, casting once on setup."""
    strength = float(strength)
    if not math.isfinite(strength):
        raise ValueError("Steering strength must be finite")
    reference = next(layer.parameters())
    if steering_matrix is not None:
        hidden_size = layer.input_layernorm.weight.numel()
        if steering_matrix.shape != (hidden_size, hidden_size):
            raise ValueError(f"Layer steering matrix must have shape {(hidden_size, hidden_size)}")
        target = device if device is not None else _execution_device(layer)
        if target is None:
            # Offloaded weights can be meta tensors; retain real matrix data.
            target = "cpu" if reference.device.type == "meta" else reference.device
        steering_matrix = steering_matrix.detach().to(device=target, dtype=reference.dtype)
        if not torch.isfinite(steering_matrix).all():
            raise ValueError(
                "Steering matrix contains nonfinite values or overflows the model dtype"
            )
    layer.steering_matrix = steering_matrix
    layer.strength = strength


def set_model_steering(layers, steering_matrix=None, strength=None, device=None):
    """Validate and distribute (layers, d, d) matrices to each layer's device."""
    num_layers = len(layers)
    hidden_size = layers[0].input_layernorm.weight.numel()
    if steering_matrix is not None:
        if not isinstance(steering_matrix, torch.Tensor) or steering_matrix.shape != (
            num_layers,
            hidden_size,
            hidden_size,
        ):
            raise ValueError(
                f"Steering matrices must have shape {(num_layers, hidden_size, hidden_size)}"
            )
    if strength is None:
        strengths = [0.0] * num_layers
    elif isinstance(strength, Real):
        strengths = [float(strength)] * num_layers
    else:
        strengths = list(strength)
    if len(strengths) != num_layers or not all(math.isfinite(float(value)) for value in strengths):
        raise ValueError(f"strength must contain {num_layers} finite values")
    for index, layer in enumerate(layers):
        set_layer_steering(
            layer,
            None if steering_matrix is None else steering_matrix[index],
            strengths[index],
            device,
        )


def _is_prefill(layer, arguments):
    cache = arguments.get("past_key_value")
    if hasattr(cache, "get_seq_length"):
        cached_length = cache.get_seq_length(layer.self_attn.layer_idx)
        if isinstance(cached_length, int) and cached_length > 0:
            # DynamicCache exposes a Python length: avoid a GPU synchronization
            # on cache_position[0] in every layer of every decode step.
            return False
    cache_position = arguments.get("cache_position")
    if cache_position is not None and cache_position.numel():
        return int(cache_position.reshape(-1)[0]) == 0
    cache = arguments.get("past_key_value")
    if cache is None:
        return True
    if hasattr(cache, "get_seq_length"):
        return cache.get_seq_length(layer.self_attn.layer_idx) == 0
    # Legacy per-layer (key, value) caches use (..., sequence, head_dim).
    return len(cache) == 0 or cache[0].shape[-2] == 0


def _steer_decoder_input(layer, args, kwargs):
    """On prefill apply h[b,t,:] += alpha * (h[b,last,:] @ W).

    The same last-token shift is broadcast over all prompt tokens, preserving
    the experiment's intervention.  Decoder-input extraction and inference
    require left-padded prompts so the last token is a real token.  Explicit
    cache positions detect both one-token prompts and cached continuations.
    A cache-free forward is treated as a fresh prompt.
    """
    if layer.steering_matrix is None or layer.strength == 0.0:
        return None
    if len(args) > 1:
        arguments = layer._steering_forward_signature.bind_partial(*args, **kwargs).arguments
    else:
        arguments = kwargs
    if not _is_prefill(layer, arguments):
        return None
    hidden = args[0] if args else kwargs["hidden_states"]
    execution_device = _execution_device(layer)
    if execution_device is not None:
        # Module pre-hooks run before Accelerate dispatches native forward.
        hidden = hidden.to(execution_device)
    matrix = layer.steering_matrix
    if matrix.device != hidden.device or matrix.dtype != hidden.dtype:
        # Handles mixed precision and device-map dispatch without changing h's
        # dtype or copying the entire (layers, d, d) tensor to a single GPU.
        matrix = matrix.to(device=hidden.device, dtype=hidden.dtype)
        layer.steering_matrix = matrix
    shift = layer.strength * (hidden[:, -1, :] @ matrix)
    hidden = hidden + shift.unsqueeze(1)
    if args:
        return (hidden, *args[1:]), kwargs
    return args, {**kwargs, "hidden_states": hidden}
