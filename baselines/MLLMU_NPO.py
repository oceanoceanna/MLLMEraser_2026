"""LoRA unlearning with per-example Negative Preference Optimization."""

import argparse
import json
import math
import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch
from accelerate import Accelerator
from peft import LoraConfig, PeftModel, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    MllamaForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    get_scheduler,
)

from data_process.data_preprocess import LLAVA_multimodal_Dataset, collate_multimodal
from lib.npo import npo_loss, sequence_nll


def find_all_linear_names(model):
    """Preserve the existing vision/language LoRA targets, excluding embeddings/head."""
    excluded = ("embeddings", "embed_tokens", "patch_embed")
    return sorted(
        {
            ".".join(name.split(".")[-2:])
            for name, module in model.named_modules()
            if isinstance(module, torch.nn.Linear)
            and name.split(".")[-1] != "lm_head"
            and not any(keyword in name for keyword in excluded)
        }
    )


def _model_class(model_id):
    model_id = model_id.lower()
    if "llava" in model_id:
        return LlavaForConditionalGeneration
    if "qwen2.5" in model_id or "qwen2_5" in model_id:
        return Qwen2_5_VLForConditionalGeneration
    if "llama" in model_id:
        return MllamaForConditionalGeneration
    raise ValueError("Supported model families: LLaVA, Qwen2.5-VL, and Mllama")


def _load_model(model_id, path, device):
    # A single device per model avoids inference-only automatic CPU/disk offload
    # and gives Accelerator ownership of the trainable model under DDP.
    dtype = torch.float32
    if device.type == "cuda":
        dtype = (
            torch.bfloat16
            if "qwen" in model_id.lower() and torch.cuda.is_bf16_supported()
            else torch.float16
        )
    return _model_class(model_id).from_pretrained(
        path,
        torch_dtype=dtype,
        device_map={"": str(device)},
        low_cpu_mem_usage=True,
        local_files_only=True,
    )


def load_model_and_processor(args, device):
    model = _load_model(args.model_id, args.vanilla_dir, device)
    processor_source = (
        args.vanilla_dir
        if (Path(args.vanilla_dir) / "preprocessor_config.json").exists()
        else args.model_id
    )
    processor = AutoProcessor.from_pretrained(processor_source)
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token_id is None:
        if processor.tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer requires an existing pad or EOS token")
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    if len(processor.tokenizer) > model.get_input_embeddings().weight.shape[0]:
        raise ValueError(
            "Processor vocabulary exceeds model embeddings; use the processor saved with this checkpoint"
        )
    if "llava" in args.model_id.lower():
        # Older saved processors omit these fields needed for image-token expansion.
        processor.patch_size = model.config.vision_config.patch_size
        processor.vision_feature_select_strategy = model.config.vision_feature_select_strategy
        processor.num_additional_image_tokens = 1
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.rank,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=find_all_linear_names(model),
            init_lora_weights="gaussian",
        ),
    )
    model.config.use_cache = False
    model.print_trainable_parameters()
    return model, processor


def train_epoch(
    model, oracle_model, train_dataloader, optimizer, lr_scheduler, accelerator, beta, epoch=0
):
    """Train one epoch; Accelerator handles accumulation and the final partial batch."""
    model.train()
    oracle_model.eval()
    reference_device = next(oracle_model.parameters()).device
    total_loss, example_count, window_examples = 0.0, 0, 0
    optimizer.zero_grad()
    progress = tqdm(
        train_dataloader, desc=f"Epoch {epoch + 1}", disable=not accelerator.is_local_main_process
    )
    for batch in progress:
        with accelerator.accumulate(model):
            labels = batch["labels"]
            inputs = {key: value for key, value in batch.items() if key != "labels"}
            # Avoid retaining the frozen model's logits across the current forward.
            with torch.no_grad():
                reference_inputs = {
                    key: value.to(reference_device) for key, value in inputs.items()
                }
                reference_outputs = oracle_model(**reference_inputs)
                reference_nll = sequence_nll(reference_outputs.logits, labels)
                del reference_outputs, reference_inputs
            outputs = model(**inputs)
            current_nll = sequence_nll(outputs.logits, labels)
            loss = npo_loss(current_nll, reference_nll, beta)
            # Accumulate summed per-example gradients, then normalize the whole
            # effective batch. This also handles a shorter final microbatch.
            count = labels.shape[0]
            window_examples += count
            accelerator.backward(loss * count * accelerator.gradient_accumulation_steps)
            if accelerator.sync_gradients:
                total_examples = accelerator.reduce(
                    torch.tensor(window_examples, device=accelerator.device), reduction="sum"
                ).item()
                # DDP has averaged gradients across ranks, so undo that factor
                # before dividing by the global example count.
                scale = accelerator.num_processes / total_examples
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(scale)
                window_examples = 0
            optimizer.step()
            if accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped:
                lr_scheduler.step()
            optimizer.zero_grad()
        total_loss += loss.detach().item() * count
        example_count += count
        progress.set_postfix(loss=loss.detach().item())
    if not example_count:
        raise ValueError("Training loader is empty")
    return total_loss / example_count


def main(args):
    accumulation_steps = args.gradient_accumulation_steps or (
        4 if args.gradient_accumulation else 1
    )
    # LoRA adapter weights remain fp32; AMP handles the frozen fp16 backbone.
    mixed_precision = "no"
    if torch.cuda.is_available():
        mixed_precision = (
            "bf16" if "qwen" in args.model_id.lower() and torch.cuda.is_bf16_supported() else "fp16"
        )
    accelerator = Accelerator(
        gradient_accumulation_steps=accumulation_steps, mixed_precision=mixed_precision
    )
    forget_file = Path(args.data_split_dir) / f"forget_{args.forget_split_ratio}" / args.forget_file
    dataset = LLAVA_multimodal_Dataset(pd.read_parquet(forget_file))
    if not dataset:
        raise ValueError(f"No valid QA pairs in {forget_file}")
    model, processor = load_model_and_processor(args, accelerator.device)
    reference_device = accelerator.device
    if accelerator.num_processes == 1 and torch.cuda.device_count() > 1:
        reference_device = torch.device("cuda:1")
    oracle_model = _load_model(args.model_id, args.oracle_model_id, reference_device)
    oracle_model.requires_grad_(False)
    oracle_model.eval()
    oracle_model.config.use_cache = False
    if (
        model.get_input_embeddings().weight.shape
        != oracle_model.get_input_embeddings().weight.shape
    ):
        raise ValueError(
            "Current and reference models must share the same vocabulary and embedding shape"
        )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    train_dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=partial(
            collate_multimodal,
            processor=processor,
            answer_only=args.ans_only,
            max_length=args.max_length,
        ),
    )
    optimizer = AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad), lr=args.lr
    )
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=math.ceil(len(train_dataloader) / accumulation_steps) * args.num_epochs,
    )
    for epoch in range(args.num_epochs):
        loss = train_epoch(
            model,
            oracle_model,
            train_dataloader,
            optimizer,
            lr_scheduler,
            accelerator,
            args.beta,
            epoch,
        )
        accelerator.print(f"Epoch {epoch + 1} loss: {loss:.6f}")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        if isinstance(unwrapped_model, PeftModel):
            unwrapped_model = unwrapped_model.merge_and_unload()
        unwrapped_model.config.use_cache = True
        unwrapped_model.save_pretrained(args.save_dir)
        processor.save_pretrained(args.save_dir)
        with open(Path(args.save_dir) / "trainer_config.json", "w", encoding="utf-8") as handle:
            json.dump(vars(args), handle, indent=2)
        accelerator.print(f"Model and processor saved to {args.save_dir}")
    accelerator.wait_for_everyone()


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "1", "yes"):
        return True
    if value.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError("Expected true/false, yes/no, or 1/0")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_id", required=True, help="Model family or processor checkpoint")
    parser.add_argument("--vanilla_dir", required=True, help="Local initial model checkpoint")
    parser.add_argument(
        "--oracle_model_id", required=True, help="Local frozen reference checkpoint"
    )
    parser.add_argument("--save_dir", default="./saved_model", help="Merged model output directory")
    parser.add_argument(
        "--data_split_dir", required=True, help="Directory containing forget_<ratio> splits"
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--forget_split_ratio", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--beta", type=float, default=0.4, help="Positive NPO inverse temperature")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Optional maximum processed length; oversized sequences raise an error",
    )
    parser.add_argument(
        "--gradient_accumulation",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="Compatibility flag: accumulate 4 microbatches",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=None,
        help="Explicit accumulation count; overrides the compatibility flag",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Reduce activation memory independently of accumulation",
    )
    parser.add_argument("--forget_file", default="train-00000-of-00001.parquet")
    parser.add_argument(
        "--ans_only",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=True,
        help="Restrict targets to the assistant answer; pass false for full text",
    )
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank for every model family")
    return parser


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    for name in ("batch_size", "num_epochs", "rank", "lr", "beta"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name} must be finite and positive")
    if args.gradient_accumulation_steps is not None and args.gradient_accumulation_steps <= 0:
        parser.error("--gradient_accumulation_steps must be positive")
    if args.max_length is not None and args.max_length < 2:
        parser.error("--max_length must be at least 2")
    if not 0 < args.forget_split_ratio <= 100:
        parser.error("--forget_split_ratio must be in (0, 100]")
    return args


if __name__ == "__main__":
    main(parse_args())
