#!/bin/bash
module load cuda/12.6.1 

#TODO: Choose model from constants.py > MODELS
MODEL="roberta-base"
TRAIN_DATA="StanfordAIMI/srrg_findings_impression"
OUTPUT_DIR="roberta-base-1"

# Set the W&B project name and run name
export WANDB_PROJECT="roberta_training"
export WANDB_NAME="roberta-base"  # Custom W&B run name

export PYTHONPATH=.

python src/train_model.py \
    --model $MODEL \
    --data_path "$TRAIN_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --resume_from_checkpoint True \
    --fp16 False \
    --bf16 True \
    --num_train_epochs 10 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "epoch" \
    --logging_steps 50 \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --load_best_model_at_end True \
    --lr_scheduler_type "cosine" \
    --learning_rate 1e-4 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.05 \
    --model_max_length 370 \
    --generation_max_length 286 \
    --generation_min_length 120 \
    --report_to "wandb" \
    --group_by_length True \
    --gradient_checkpointing False \
    --lazy_preprocess True \
    --dataloader_num_workers 1 \

    #--attn_implementation "flash_attention_2" \
    
