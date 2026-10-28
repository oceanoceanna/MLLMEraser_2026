"""Evaluate a baseline checkpoint on the MLLMU image-textual benchmarks."""

from lib.evaluation import (
    compute_bleu,
    evaluate_classification,
    evaluate_fill_in_the_blank,
    evaluate_generation,
    formulate_prompt_with_options,
    load_and_combine_parquet_files,
    parse_arguments,
    run_baseline,
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


def main():
    run_baseline(parse_arguments())


if __name__ == "__main__":
    main()
