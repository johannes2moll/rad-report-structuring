#!/bin/bash

MODEL="llama-1b"  #llama-1b, llama-3b, llama-8b, llama-70b, vicuna, medalpaca-7b, medalpaca-13b, mistral-nemo, mistral-7b, phi3
CASE_ID=1
PRED_FILE="output/case${CASE_ID}/${MODEL}/test_predictions.json"
REF_DATA_PATH="StanfordAIMI/srrg_findings_impression"
OUTPUT_FILE="output/case${CASE_ID}/${MODEL}/metrics.json"

python src/calc_metrics.py \
    --pred_file "$PRED_FILE" \
    --ref_data_path "$REF_DATA_PATH" \
    --output_file "$OUTPUT_FILE" 