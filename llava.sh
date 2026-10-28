#!/usr/bin/env bash
# Run the complete LLaVA experiment from this file.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

python_bin="${PYTHON_BIN:-python}"
model_path="${MODEL_PATH:-./LLaVA_Vanilla}"
data_dir="${DATA_DIR:-./data/MLLMU-Bench}"
training_dir="${TRAIN_DATA_DIR:-./baseline_train_split}"
forget_ratio="${FORGET_RATIO:-10}"
subspace_ratio="${SUBSPACE_RATIO:-0.3}"
ridge_lambda="${RIDGE_LAMBDA:-0.1}"
save_dir="${SAVE_DIR:-./NPO_llava}"
log_dir="${LOG_DIR:-./logs/llava}"
matrix_path="${MATRIX_PATH:-$data_dir/steering_matrix/llava_${subspace_ratio}_${ridge_lambda}_${forget_ratio}.pt}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [ "$#" -ne 0 ]; then
  printf 'Usage: bash llava.sh (configure paths and parameters with environment variables)\n' >&2
  exit 2
fi
command -v "$python_bin" >/dev/null || { printf 'Python executable not found: %s\n' "$python_bin" >&2; exit 1; }
for directory in "$model_path" "$data_dir" "$training_dir"; do
  if [ ! -d "$directory" ]; then
    printf 'Required directory not found: %s\nPrepare the model/data or set MODEL_PATH, DATA_DIR and TRAIN_DATA_DIR.\n' "$directory" >&2
    exit 1
  fi
done
mkdir -p "$log_dir"

# 1. Extract refusal directions and forget/retain embeddings.
"$python_bin" -m refusal_direction.extract_llava --model_path "$model_path"
for split in forget retain; do
  "$python_bin" -m newcode.embed_llava --model_path "$model_path" \
    --dataset "$split" --forget_ratio "$forget_ratio" --data_dir "$data_dir"
done

# 2. Fit and evaluate the steering matrices.
"$python_bin" -m newcode.all_calculate_steering_matrix --model llava \
  --this_ratio "$subspace_ratio" --lambda_reg_para "$ridge_lambda" \
  --forget_ratio "$forget_ratio" --data_dir "$data_dir" --output "$matrix_path"
"$python_bin" MLLMU_eval_steering_llava.py --cache_path "$model_path" \
  --forget_ratio "$forget_ratio" --data_split_folder "$data_dir" --steering_matrix_path "$matrix_path"

# 3. Evaluate the vanilla model, train NPO, and evaluate the merged NPO model.
"$python_bin" MLLMU_eval.py --model_id llava-hf/llava-1.5-7b-hf --cache_path "$model_path" \
  --forget_ratio "$forget_ratio" --data_split_folder "$data_dir" > "$log_dir/vanilla.log"
CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-$CUDA_VISIBLE_DEVICES}" \
  "$python_bin" -m baselines.MLLMU_NPO --model_id llava-hf/llava-1.5-7b-hf \
  --vanilla_dir "$model_path" --oracle_model_id "$model_path" \
  --save_dir "$save_dir" --data_split_dir "$training_dir" \
  --batch_size "${BATCH_SIZE:-4}" --beta "${BETA:-0.9}" \
  --lr "${LEARNING_RATE:-1e-5}" --forget_split_ratio "$forget_ratio" \
  --num_epochs "${NUM_EPOCHS:-2}"
"$python_bin" MLLMU_eval.py --model_id llava-hf/llava-1.5-7b-hf --cache_path "$save_dir" \
  --forget_ratio "$forget_ratio" --data_split_folder "$data_dir" > "$log_dir/npo.log"
