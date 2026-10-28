"""Text generation helpers which decode new tokens, never the input prompt."""

import torch


def _generate(
    model, texts, tokenizer, generation_config, system_prompt, max_new_tokens, warmup=False
):
    if not texts:
        return []
    messages = []
    for text in texts:
        conversation = [{"role": "user", "content": text}]
        if system_prompt:
            conversation.insert(0, {"role": "system", "content": system_prompt})
        messages.append(conversation)
    prompts = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    previous_side = tokenizer.padding_side
    try:
        tokenizer.padding_side = "left"
        inputs = tokenizer(prompts, return_tensors="pt", add_special_tokens=False, padding=True).to(
            model.device
        )
    finally:
        tokenizer.padding_side = previous_side
    kwargs = dict(
        generation_config=generation_config, pad_token_id=tokenizer.pad_token_id, do_sample=False
    )
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            if warmup:
                # Dynamic external models use this pass to calibrate their alpha state.
                if not hasattr(model, "reset_alpha"):
                    raise TypeError("Dynamic generation requires a model with reset_alpha()")
                model.reset_alpha()
                model.generate(**inputs, max_new_tokens=2, **kwargs)
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, **kwargs)
    finally:
        model.train(was_training)
    sequences = outputs.sequences if hasattr(outputs, "sequences") else outputs
    offset = (
        0 if getattr(model.config, "is_encoder_decoder", False) else inputs["input_ids"].shape[1]
    )
    return [
        answer.strip()
        for answer in tokenizer.batch_decode(sequences[:, offset:], skip_special_tokens=True)
    ]


def batch_generating(
    model, inputs, tokenizer, generation_config, if_support_sys_prompt, sys_prompt, model_sign=None
):
    """model_sign is a legacy compatibility argument; token slicing is model-independent."""
    return _generate(
        model,
        inputs,
        tokenizer,
        generation_config,
        sys_prompt if if_support_sys_prompt else None,
        256,
    )


def batch_generating_dynamic(
    model, inputs, tokenizer, generation_config, if_support_sys_prompt, sys_prompt, model_sign=None
):
    return _generate(
        model,
        inputs,
        tokenizer,
        generation_config,
        sys_prompt if if_support_sys_prompt else None,
        128,
        warmup=True,
    )
