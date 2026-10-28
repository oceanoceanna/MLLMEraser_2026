"""Evaluate a steering matrix with a Qwen2.5-VL checkpoint."""

from lib.evaluation import (
    compute_bleu,
    evaluate_classification,
    evaluate_fill_in_the_blank,
    evaluate_generation,
    formulate_prompt_with_options,
    load_and_combine_parquet_files,
    run_steering,
)
from lib.evaluation import (
    parse_arguments as _parse_arguments,
)

__all__ = [
    "compute_bleu",
    "evaluate_classification",
    "evaluate_fill_in_the_blank",
    "evaluate_generation",
    "formulate_prompt_with_options",
    "load_and_combine_parquet_files",
    "parse_arguments",
    "main",
]


def parse_arguments(argv=None):
    return _parse_arguments(argv, steering="qwen")


def main():
    run_steering(parse_arguments(), "qwen")


if __name__ == "__main__":
    main()
