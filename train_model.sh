#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_PROTO=simple
export FI_EFA_FORK_SAFE=1
export FI_LOG_LEVEL=1
export FI_EFA_USE_DEVICE_RDMA=1
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0
export OMPI_MCA_mtl_base_verbose=1
export FI_EFA_ENABLE_SupupHM_TRANSFER=0
export FI_PROVIDER=efa
export FI_EFA_TX_MIN_CREDITS=64
export NCCL_TREE_THRESHOLD=0
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
module load cuda/12.6.1 

GPUS_PER_NODE=$(nvidia-smi -L | wc -l)
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_ADDR=$(nslookup "$MASTER_ADDR" | grep -oP '(?<=Address: ).*') # Get IP for hostname

echo "GPUS_PER_NODE: ${GPUS_PER_NODE}"
echo "MASTER_ADDR: ${MASTER_ADDR}"
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
export MASTER_PORT
echo "MASTER_PORT: ${MASTER_PORT}"

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE
    --rdzv_backend static
"
#TODO: Choose model: "FacebookAI/roberta-base", "allenai/biomed_roberta_base", "UCSD-VA-health/RadBERT-RoBERTa-4m", "models/RoBERTa-base-PM-M3-Voc-distill-align"
MODEL="FacebookAI/roberta-base"
TRAIN_DATA="StanfordAIMI/srrg_findings_impression"
OUTPUT_DIR="roberta/roberta-base-4"

export HF_HOME="/home/users/jomoll/.cache"
export PYTHONPATH=.

torchrun $DISTRIBUTED_ARGS src/train_model.py \
    --model_name_or_path $MODEL \
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
    --save_total_limit 3 \
    --load_best_model_at_end True \
    --lr_scheduler_type "cosine" \
    --learning_rate 1e-4 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.05 \
    --model_max_length 370 \
    --report_to "wandb" \
    --group_by_length True \
    --gradient_checkpointing False \
    --lazy_preprocess True \
    --dataloader_num_workers 1 \

    #--attn_implementation "flash_attention_2" \
    
