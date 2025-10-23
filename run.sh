#!/bin/bash
# KITE (Korean Instruction-following Task Evaluation) - Evaluation Script
# This script runs evaluation on Korean instruction-following tasks

# Configuration
SHOT_NUM=0  # Number of few-shot examples (0, 1, 3, or 5)
GPU_ID=0    # Starting GPU ID

# Dataset selection: 'general' for KITE General (translated IFEval), 'korean' for KITE Korean (culturally-aware)
DATASET_TYPE="korean"  # Options: 'general' or 'korean'

# For KITE Korean, select categories to evaluate
# Available categories: 'acrostic', 'honorifics', 'numbers', 'postposition', or 'all'
KOREAN_CATEGORIES=('acrostic' 'honorifics' 'numbers' 'postposition')

# Model configuration
# Supported model types: 'openai', 'hf' (HuggingFace), 'solar', 'clova'
MODEL_TYPE="hf"
MODELS=(
  "meta-llama/Meta-Llama-3-8B-Instruct"
  # "google/gemma-7b-it"
  # "yanolja/EEVE-Korean-Instruct-10.8B-v1.0"
)

# OpenAI models (uncomment to use)
# MODEL_TYPE="openai"
# MODELS=("gpt-3.5-turbo" "gpt-4o")

# Function to run evaluation
run_evaluation() {
  local instruction_file=$1
  local response_dir=$2
  local eval_dir=$3
  local model=$4

  echo "Running evaluation for model: $model"
  echo "Instruction file: $instruction_file"

  CUDA_VISIBLE_DEVICES=$GPU_ID \
  python korean_instruction_following_eval/main.py \
    --instruction_file "$instruction_file" \
    --response_output_dir "$response_dir" \
    --eval_output_dir "$eval_dir" \
    --shot_num $SHOT_NUM \
    --verbosity -1 \
    --model_type "$MODEL_TYPE" \
    --model "$model"
}

# Main execution
if [ "$DATASET_TYPE" == "korean" ]; then
  echo "Evaluating KITE Korean (Culturally-Aware Instructions)"

  for category in "${KOREAN_CATEGORIES[@]}"; do
    echo "Processing category: $category"

    INSTRUCTION_FILE="korean_instruction_following_eval/data/culturally_aware/instruction/${category}.jsonl"
    RESPONSE_DIR="korean_instruction_following_eval/data/culturally_aware/response/${category}/${SHOT_NUM}_shot"
    EVAL_DIR="korean_instruction_following_eval/data/eval_results/${category}/${SHOT_NUM}_shot"

    for model in "${MODELS[@]}"; do
      run_evaluation "$INSTRUCTION_FILE" "$RESPONSE_DIR" "$EVAL_DIR" "$model"

      # Increment GPU ID for HuggingFace models to parallelize
      if [ "$MODEL_TYPE" == "hf" ]; then
        ((GPU_ID++))
      fi
    done
  done

elif [ "$DATASET_TYPE" == "general" ]; then
  echo "Evaluating KITE General (Translated and Filtered IFEval)"

  INSTRUCTION_FILE="korean_instruction_following_eval/data/translated_and_filtered/instruction/relevant.jsonl"
  RESPONSE_DIR="korean_instruction_following_eval/data/translated_and_filtered/response/${SHOT_NUM}_shot"
  EVAL_DIR="korean_instruction_following_eval/data/eval_results/translated_and_filtered/${SHOT_NUM}_shot"

  for model in "${MODELS[@]}"; do
    run_evaluation "$INSTRUCTION_FILE" "$RESPONSE_DIR" "$EVAL_DIR" "$model"

    # Increment GPU ID for HuggingFace models to parallelize
    if [ "$MODEL_TYPE" == "hf" ]; then
      ((GPU_ID++))
    fi
  done

else
  echo "Error: Invalid DATASET_TYPE. Must be 'general' or 'korean'"
  exit 1
fi

echo "Evaluation complete!"
