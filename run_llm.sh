#!/bin/bash

# Define the parameters for the inference script
CASE_ID=5
MODEL="llama-1b"  # llama-1b, llama-3b, llama-8b, llama-70b, vicuna, medalpaca-7b, medalpaca-13b, mistral-nemo, mistral-7b, phi3
CACHE_DIR=".cache"
DATA_PATH="StanfordAIMI/srrg_findings_impression"
OUTPUT_FILE_MIMIC="output/case${CASE_ID}/${MODEL}/test_predictions_mimic.json"
OUTPUT_FILE_CHEXPERT="output/case${CASE_ID}/${MODEL}/test_predictions_chexpert.json"
MAX_LENGTH=2048
MAX_GEN_LENGTH=350
MIN_GEN_LENGTH=120

# Run the Python script with the defined parameters
python src/run_llm.py \
    --case_id "$CASE_ID" \
    --model "$MODEL" \
    --cache_dir "$CACHE_DIR" \
    --data_path "$DATA_PATH" \
    --output_file_mimic "$OUTPUT_FILE_MIMIC" \
    --output_file_chexpert "$OUTPUT_FILE_CHEXPERT" \
    --max_input_length "$MAX_LENGTH" \
    --max_gen_length "$MAX_GEN_LENGTH" \
    --min_gen_length "$MIN_GEN_LENGTH" 
