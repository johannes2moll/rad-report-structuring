#!/bin/bash

# Define the parameters for the inference script
MODEL="roberta-base" # Choose model from constants.py > MODELS
CACHE_DIR=".cache"
DATA_PATH="StanfordAIMI/srrg_findings_impression"
OUTPUT_FILE_MIMIC="output/case100/${MODEL}/test_predictions_mimic.json"
OUTPUT_FILE_CHEXPERT="output/case100/${MODEL}/test_predictions_chexpert.json"
MAX_LENGTH=512
BATCH_SIZE=32
MAX_GEN_LENGTH=350
MIN_GEN_LENGTH=120

# Run the Python script with the defined parameters
python src/run_model.py \
    --model "$MODEL" \
    --cache_dir "$CACHE_DIR" \
    --data_path "$DATA_PATH" \
    --output_file_mimic "$OUTPUT_FILE_MIMIC" \
    --output_file_chexpert "$OUTPUT_FILE_CHEXPERT" \
    --max_input_length "$MAX_LENGTH" \
    --batch_size "$BATCH_SIZE" \
    --max_gen_length "$MAX_GEN_LENGTH" \
    --min_gen_length "$MIN_GEN_LENGTH" \