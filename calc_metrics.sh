#!/bin/bash

MODEL_PATH="jomoll/biomed-roberta"
MODEL_NAME=$(echo "$MODEL_PATH" | awk -F'/' '{print $2}')
PRED_FILE="output/${MODEL_NAME}/test_results.json"
REF_FILE="data/test_structured.json"
OUTPUT_FILE="output/${MODEL_NAME}/metrics.json"

python src/calc_metrics.py \
    --pred_file "$PRED_FILE" \
    --ref_file "$REF_FILE" \
    --output_file "$OUTPUT_FILE" 