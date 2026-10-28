"""Shared image-textual MLLMU evaluation and command-line runners.

Classification and cloze report percentages. Generation reports the per-question
mean of ROUGE F1 and smoothed sentence BLEU-4 in [0, 1]. Pure-text questions are
outside these metrics and are skipped before generation.
"""

import argparse
import json
import logging
import random
import re
from io import BytesIO
from pathlib import Path

import pandas as pd
import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from PIL import Image
from rouge_score import rouge_scorer
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASSIFICATION_METRIC = "Classification image-textual Question Accuracy"
CLOZE_METRIC = "Cloze image_textual_accuracy"
GENERATION_METRICS = {
    "rouge1": "Average ROUGE-1 (Image_Textual)",
    "rouge2": "Average ROUGE-2 (Image_Textual)",
    "rougeL": "Average ROUGE-L (Image_Textual)",
    "bleu": "Average BLEU (Image_Textual)",
}
TASKS = ("classification", "generation", "cloze")
SPLITS = ("forget", "test", "retain", "real")
SPLIT_TITLES = {
    "forget": "Forget Set Results",
    "test": "Test Set Results",
    "retain": "Retain Set (shared dataset) Results",
    "real": "Retain Set (real person) Results",
}


def _report(message, logger=None):
    if logger is None:
        print(message)
    else:
        logger.info(message)


def load_and_combine_parquet_files(directory):
    """Read every shard in a stable order; fail clearly for an empty directory."""
    files = sorted(Path(directory).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files found in {directory}")
    return pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)


def load_evaluation_data(source):
    """Accept an already loaded frame, a Parquet file, or a shard directory."""
    if isinstance(source, pd.DataFrame):
        return source
    path = Path(source)
    if path.is_dir():
        return load_and_combine_parquet_files(path)
    return pd.read_parquet(path)


def _evaluation_samples(source, id_list_file=None, mode="default", forget_parquet_file=None):
    frame = load_evaluation_data(source)
    if id_list_file is not None:
        with open(id_list_file, encoding="utf-8") as file:
            ids = json.load(file)
        if not isinstance(ids, list):
            raise ValueError("The ID list file must contain a JSON list.")
    elif mode == "test" and forget_parquet_file is not None:
        ids = load_evaluation_data(forget_parquet_file)["ID"].unique()
    else:
        return frame
    return frame[frame["ID"].isin(ids)]


def _sample_image(row, mode, seed):
    # A row-specific seed keeps the chosen test image identical across tasks,
    # even if a row contains no questions for one of those tasks.
    if mode == "test" and "images" in row:
        images = row["images"]
        if images is None or len(images) == 0:
            raise ValueError(f"Sample {row['ID']} has no test images.")
        image_record = random.Random(f"{seed}:{row['ID']}").choice(images)
    else:
        image_record = row["image"]
    image_bytes = image_record.get("bytes")
    image_source = BytesIO(image_bytes) if image_bytes is not None else image_record.get("path")
    if image_source is None:
        raise ValueError(f"Sample {row['ID']} has neither image bytes nor an image path.")
    with Image.open(image_source) as image:
        return image.convert("RGB")


def compute_bleu(ground_truth, predicted_answer):
    """Preserve the benchmark's whitespace-tokenized smoothed BLEU-4."""
    return sentence_bleu(
        [ground_truth.split()],
        predicted_answer.split(),
        smoothing_function=SmoothingFunction().method1,
    )


def formulate_prompt_with_options(question, options):
    return question + "\n" + "\n".join(f"{key}: {value}" for key, value in options.items())


def parse_classification_answer(response, options):
    """Read a standalone option label, not the first letter of arbitrary prose."""
    match = re.match(
        r"^\s*(?:(?:the\s+)?(?:correct\s+)?answer\s*(?:is\s*)?[:\-]?\s*)?"
        r"[\[(]?([a-zA-Z0-9])(?=$|[\s\]).:,])",
        response,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    answer = match.group(1).upper()
    return answer if answer in {str(key).upper() for key in options} else None


def model_family(model_id):
    name = model_id.lower()
    for family in ("llava", "qwen", "llama"):
        if family in name:
            return family
    raise ValueError(
        f"Unsupported model ID: {model_id}. Expected LLaVA, Qwen2.5-VL, or Llama vision."
    )


def _generate_answer(question, image, processor, tokenizer, model, args, task):
    family = model_family(args.model_id)
    if family == "qwen":
        conversation = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}
        ]
        prompt = processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        ).strip()
    elif family == "llava":
        prompt = f"USER: <image>\n{question}\nASSISTANT:"
    elif task == "generation":
        # Retain the existing Mllama generation prompt for metric comparability.
        prompt = f"<|image|><|begin_of_text|>### Question:{question}\n### Answer:"
    else:
        prompt = f"USER: <|image|><|begin_of_text|>\n{question}\nASSISTANT:"
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    # Respect Accelerate's input device instead of assuming cuda:0.
    inputs = inputs.to(model.device)
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
    if hasattr(outputs, "sequences"):
        outputs = outputs.sequences
    continuation = outputs[0, inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(continuation, skip_special_tokens=True)


def evaluate_classification(
    parquet_file,
    processor,
    tokenizer,
    model,
    args,
    id_list_file=None,
    mode="default",
    forget_parquet_file=None,
    logger=None,
):
    samples = _evaluation_samples(parquet_file, id_list_file, mode, forget_parquet_file)
    correct = total = 0
    for _, row in tqdm(samples.iterrows(), total=len(samples)):
        questions = row["Classification_Task"].get("Image_Textual_Questions", [])
        if len(questions) == 0:
            continue
        image = _sample_image(row, mode, getattr(args, "seed", 42))
        for item in questions:
            question = formulate_prompt_with_options(item["Question"], item["Options"])
            question += "\nJust give ONE letter representing the answer directly."
            response = _generate_answer(
                question, image, processor, tokenizer, model, args, "classification"
            )
            predicted = parse_classification_answer(response, item["Options"])
            correct += predicted == str(item["Correct_Answer"]).strip().upper()
            total += 1
    accuracy = 100 * correct / total if total else 0.0
    _report(f"{CLASSIFICATION_METRIC}: {accuracy:.2f}%", logger)
    return {CLASSIFICATION_METRIC: accuracy}


def evaluate_fill_in_the_blank(
    parquet_file,
    processor,
    tokenizer,
    model,
    args,
    id_list_file=None,
    mode="default",
    forget_parquet_file=None,
    logger=None,
):
    samples = _evaluation_samples(parquet_file, id_list_file, mode, forget_parquet_file)
    correct = total = 0
    for _, row in tqdm(samples.iterrows(), total=len(samples)):
        questions = [item for item in row["Mask_Task"] if item["Type"] == "Image_Textual"]
        if not questions:
            continue
        image = _sample_image(row, mode, getattr(args, "seed", 42))
        for item in questions:
            question = item["Question"].replace("__", "[Blank]")
            question += (
                "\nPlease **ONLY** provide the correct answer that should replace the [Blank]."
            )
            response = _generate_answer(question, image, processor, tokenizer, model, args, "cloze")
            truth = item["Ground_Truth"].lower()
            # This is the benchmark's case-insensitive substring metric.
            correct += bool(truth) and truth in response.lower()
            total += 1
    accuracy = 100 * correct / total if total else 0.0
    _report(f"Cloze Image-Textual Question Accuracy: {accuracy:.2f}%", logger)
    return {CLOZE_METRIC: accuracy}


def evaluate_generation(
    parquet_file,
    processor,
    tokenizer,
    model,
    args,
    mode="default",
    forget_parquet_file=None,
    logger=None,
):
    samples = _evaluation_samples(parquet_file, mode=mode, forget_parquet_file=forget_parquet_file)
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    totals = {key: 0.0 for key in GENERATION_METRICS}
    count = 0
    for _, row in tqdm(samples.iterrows(), total=len(samples)):
        questions = [item for item in row["Generation_Task"] if item["Type"] == "Image_Textual"]
        if not questions:
            continue
        image = _sample_image(row, mode, getattr(args, "seed", 42))
        for item in questions:
            question = item["Question"]
            if model_family(args.model_id) != "llama":
                question += "\nAnswer the question based on your trained knowledge in one sentence in ENGLISH."
            response = _generate_answer(
                question, image, processor, tokenizer, model, args, "generation"
            )
            truth = item["Ground_Truth"]
            scores = scorer.score(truth, response)
            for key in ("rouge1", "rouge2", "rougeL"):
                totals[key] += scores[key].fmeasure
            totals["bleu"] += compute_bleu(truth, response)
            count += 1
    averages = {
        label: totals[key] / count if count else 0.0 for key, label in GENERATION_METRICS.items()
    }
    for label, value in averages.items():
        _report(f"{label}: {value}", logger)
    return averages


def parse_selection(value, allowed):
    selected = tuple(dict.fromkeys(re.split(r"[,_\s]+", value.strip())))
    invalid = set(selected) - set(allowed)
    if invalid:
        raise ValueError(f"Invalid selection {sorted(invalid)}; choose from {', '.join(allowed)}.")
    return selected


def parse_arguments(argv=None, steering=None):
    parser = argparse.ArgumentParser(description="Evaluate MLLMU image-textual tasks.")
    defaults = {"llava": "llava-hf/llava-1.5-7b-hf", "qwen": "Qwen/Qwen2.5-VL-7B-Instruct"}
    parser.add_argument("--model_id", required=steering is None, default=defaults.get(steering))
    default_checkpoint = PROJECT_ROOT / ("LLaVA_Vanilla" if steering == "llava" else "Qwen_Vanilla")
    parser.add_argument(
        "--cache_path",
        default=str(default_checkpoint) if steering else None,
        help="Model checkpoint path or Hugging Face ID (defaults to model_id for baseline).",
    )
    parser.add_argument("--data_split_folder", default=str(PROJECT_ROOT / "data" / "MLLMU-Bench"))
    parser.add_argument(
        "--test_data",
        help="Test Parquet file or directory; defaults to <data_split_folder>/Test_Set.",
    )
    parser.add_argument(
        "--celebrity_data",
        help="Celebrity Parquet file or directory; defaults to <data_split_folder>/Retain_Set.",
    )
    parser.add_argument(
        "--forget_ratio",
        type=int,
        default=10,
        help="Forget percentage used in split directory names.",
    )
    parser.add_argument(
        "--eval_list",
        default="_".join(TASKS if steering else SPLITS),
        help="Underscore/comma-separated tasks (steering) or dataset splits (baseline).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for choosing test images consistently across tasks.",
    )
    parser.add_argument(
        "--attn_implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        help="Optional Transformers attention backend; FlashAttention requires a separate installation.",
    )
    if steering:
        strengths = (-0.25, -0.35, -0.25) if steering == "llava" else (-0.30, -0.26, -0.27)
        parser.add_argument("--steering_matrix_path", required=True)
        for task, strength in zip(TASKS, strengths):
            parser.add_argument(f"--{task}_strength", type=float, default=strength)
    args = parser.parse_args(argv)
    if not 0 < args.forget_ratio < 100:
        parser.error("--forget_ratio must be between 1 and 99.")
    try:
        parse_selection(args.eval_list, TASKS if steering else SPLITS)
        family = model_family(args.model_id)
        if steering and family != steering:
            parser.error(f"This entrypoint requires a {steering} model.")
    except ValueError as error:
        parser.error(str(error))
    args.cache_path = args.cache_path or args.model_id
    args.test_data = args.test_data or str(Path(args.data_split_folder) / "Test_Set")
    args.celebrity_data = args.celebrity_data or str(Path(args.data_split_folder) / "Retain_Set")
    return args


def _load_model(args, steering=None):
    # Lazy, model-specific imports keep --help lightweight and avoid importing
    # an unrelated model implementation in each environment.
    from transformers import AutoProcessor

    family = model_family(args.model_id)
    if steering == "llava":
        from newcode.llava_steering import LlavaForSteering as model_class
    elif steering == "qwen":
        from newcode.qwen_steering import QwenForSteering as model_class
    elif family == "llava":
        from transformers import LlavaForConditionalGeneration as model_class
    elif family == "qwen":
        from transformers import Qwen2_5_VLForConditionalGeneration as model_class
    else:
        from transformers import MllamaForConditionalGeneration as model_class
    use_cuda = torch.cuda.is_available()
    dtype = torch.float32
    if use_cuda:
        dtype = (
            torch.bfloat16 if family == "qwen" and torch.cuda.is_bf16_supported() else torch.float16
        )
    kwargs = {
        "torch_dtype": dtype,
        "device_map": "auto" if use_cuda else "cpu",
        "low_cpu_mem_usage": True,
    }
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation
    try:
        processor = AutoProcessor.from_pretrained(args.cache_path)
    except OSError:
        if args.cache_path == args.model_id:
            raise
        # Legacy model-only checkpoints did not always save their processor.
        logging.getLogger(__name__).warning(
            "No processor could be loaded from %s; using %s.", args.cache_path, args.model_id
        )
        processor = AutoProcessor.from_pretrained(args.model_id)
    tokenizer = processor.tokenizer
    # Decoder steering reads the final prompt token; left padding keeps that
    # token valid if callers later batch examples of different lengths.
    tokenizer.padding_side = "left"
    model = model_class.from_pretrained(args.cache_path, **kwargs)
    if family == "llava":
        processor.patch_size = model.config.vision_config.patch_size
        processor.vision_feature_select_strategy = model.config.vision_feature_select_strategy
        # LLaVA 1.5's CLIP vision tower emits a CLS token before patch tokens.
        # Older processor files default this value to zero, causing an image
        # placeholder/features mismatch even after patch_size is populated.
        vision_type = getattr(model.config.vision_config, "model_type", "clip_vision_model")
        if vision_type == "clip_vision_model":
            processor.num_additional_image_tokens = 1
    model.eval()
    return processor, tokenizer, model


def _load_splits(args, selected):
    root = Path(args.data_split_folder)
    paths = {
        "forget": root / f"forget_{args.forget_ratio}",
        "retain": root / f"retain_{100 - args.forget_ratio}",
        "test": Path(args.test_data),
        "real": Path(args.celebrity_data),
    }
    required = set(selected)
    if "test" in selected:
        required.add("forget")
    return {name: load_evaluation_data(paths[name]) for name in SPLITS if name in required}


def _run_task(task, split, frames, processor, tokenizer, model, args, logger=None):
    evaluator = {
        "classification": evaluate_classification,
        "cloze": evaluate_fill_in_the_blank,
        "generation": evaluate_generation,
    }[task]
    return evaluator(
        frames[split],
        processor,
        tokenizer,
        model,
        args,
        mode=split,
        forget_parquet_file=frames.get("forget") if split == "test" else None,
        logger=logger,
    )


def _summary(results, logger=None, baseline=False):
    _report("************************IMAGE_TEXTUAL RESULT*****************:", logger)
    labels = {"forget": "Forget", "test": "Test", "retain": "Retain", "real": "Real"}
    for task, metric in (
        ("classification", CLASSIFICATION_METRIC),
        ("generation", GENERATION_METRICS["rougeL"]),
        ("cloze", CLOZE_METRIC),
    ):
        _report(f"#######################{task.upper()} RESULTS################:", logger)
        for split, by_task in results.items():
            if task in by_task:
                label = labels[split] + (" Set" if baseline else "")
                _report(f"{label} {metric}:{by_task[task][metric]}", logger)


def run_baseline(args):
    selected = parse_selection(args.eval_list, SPLITS)
    frames = _load_splits(args, selected)
    processor, tokenizer, model = _load_model(args)
    results = {}
    for split in selected:
        _report(f"### Evaluating {split.title()} Set ###")
        results[split] = {
            task: _run_task(task, split, frames, processor, tokenizer, model, args)
            for task in TASKS
        }
        _report(SPLIT_TITLES[split] + ":")
        for metrics in results[split].values():
            _report(metrics)
    _summary(results, baseline=True)
    return {
        SPLIT_TITLES[split]: {
            "fill_in_the_blank" if task == "cloze" else task: metrics
            for task, metrics in tasks.items()
        }
        for split, tasks in results.items()
    }


def _steering_logger(args, family, selected):
    log_dir = PROJECT_ROOT / f"log_{family}"
    log_dir.mkdir(parents=True, exist_ok=True)
    if set(selected) == set(TASKS):
        name = (
            f"all_cloze_{args.cloze_strength}_generation_{args.generation_strength}_"
            f"classification_{args.classification_strength}_{args.forget_ratio}"
        )
    else:
        stem = re.sub(r"[^\w.-]", "_", Path(args.steering_matrix_path).stem)
        settings = "_".join(f"{task}_{getattr(args, task + '_strength')}" for task in selected)
        name = f"{stem}_{settings}_{args.forget_ratio}"
    logger = logging.getLogger(f"mllmu.evaluation.{family}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    handler = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler())
    return logger


def run_steering(args, family):
    selected = parse_selection(args.eval_list, TASKS)
    # Load on CPU first; each decoder layer installs its own device/dtype copy.
    matrix = torch.load(args.steering_matrix_path, map_location="cpu", weights_only=True)
    if not isinstance(matrix, torch.Tensor) or matrix.ndim != 3:
        raise ValueError("The steering matrix must be a [layers, hidden_size, hidden_size] tensor.")
    frames = _load_splits(args, SPLITS)
    processor, tokenizer, model = _load_model(args, family)
    config = getattr(model.config, "text_config", model.config)
    expected = (config.num_hidden_layers, config.hidden_size, config.hidden_size)
    if tuple(matrix.shape) != expected:
        raise ValueError(
            f"Steering matrix shape {tuple(matrix.shape)} does not match model shape {expected}."
        )
    logger = _steering_logger(args, family, selected)
    results = {split: {} for split in SPLITS}
    for task in selected:
        # Preserve the original intervention on all decoder layers except layer 0.
        strength = [0.0] + [getattr(args, task + "_strength")] * (config.num_hidden_layers - 1)
        model.set_steering_parameters(steering_matrix=matrix, strength=strength)
        _report(f"### Evaluating {task} task ###", logger)
        for split in SPLITS:
            results[split][task] = _run_task(
                task, split, frames, processor, tokenizer, model, args, logger
            )
        metric = {
            "classification": CLASSIFICATION_METRIC,
            "generation": GENERATION_METRICS["rougeL"],
            "cloze": CLOZE_METRIC,
        }[task]
        result_name = "fill_in_the_blank" if task == "cloze" else task
        for split in SPLITS:
            _report(f"{split}_{result_name}_result:{results[split][task][metric]}", logger)
    _summary(results, logger)
    return results
