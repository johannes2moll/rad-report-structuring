#!/bin/bash

# Define the parameters for the inference script
CASE_ID=1
MODEL_PATH="microsoft/phi-2"
MODEL_NAME=$(echo "$MODEL_PATH" | awk -F'/' '{print $2}')
CACHE_DIR=".cache"
DATA_PATH="StanfordAIMI/srrg_findings_impression"
OUTPUT_FILE="output/${MODEL_NAME}/test_results.json"
MAX_LENGTH=512
BATCH_SIZE=1
MAX_GEN_LENGTH=512
MIN_GEN_LENGTH=120

# Run the Python script with the defined parameters
python src/run_llm.py \
    --case_id "$CASE_ID" \
    --model_path "$MODEL_PATH" \
    --cache_dir "$CACHE_DIR" \
    --data_path "$DATA_PATH" \
    --output_file "$OUTPUT_FILE" \
    --max_input_length "$MAX_LENGTH" \
    --batch_size "$BATCH_SIZE" \
    --max_gen_length "$MAX_GEN_LENGTH" \
    --min_gen_length "$MIN_GEN_LENGTH" 
