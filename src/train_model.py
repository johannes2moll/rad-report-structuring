####################################################################################################
# This script is used to fine-tune a phi3 model on radiology report section generation. 
# The model is fine-tuned on a dataset of free text radiology reports and their corresponding structured impression sections.
# For training, execute the corresponding bash file on a machine with a GPU.
####################################################################################################

# Standard Library Imports
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, List
import numpy as np
# PyTorch Imports
import torch
import evaluate
from datasets import load_dataset

# Hugging Face Transformers Imports
import transformers
from transformers import (
    Seq2SeqTrainer,
    TrainerCallback,
)
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

from transformers.integrations import deepspeed
from transformers.trainer_pt_utils import LabelSmoother

# Accelerate and Distributed Training
from accelerate.utils import DistributedType

import constants
from utils import load_and_preprocess_dataset, get_data_collator

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="StanfordAIMI/RadLLaMA-7b")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    generation_max_length: Optional[int] = field(
        default=286, metadata={"help": "Maximum length of generated sequences"}
    )
    generation_min_length: Optional[int] = field(
        default=120, metadata={"help": "Minimum length of generated sequences"}
    )
    use_lora: bool = False
    
    # Adding the new fields as per your request
    generation_config: Optional[transformers.GenerationConfig] = field(
        default=None, metadata={"help": "Generation config used during sequence generation."}
    )
    generation_num_beams: int = field(
        default=5, metadata={"help": "Number of beams to use for beam search during generation."}
    )
    predict_with_generate: bool = field(
        default=True, metadata={"help": "Whether to use generate() for predictions during evaluation."}
    )
    save_on_cpu: bool = field(
        default=True, metadata={"help": "Whether to save the model on the CPU instead of the GPU."}
    )

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    # check if zero3 mode enabled
    if deepspeed.is_deepspeed_zero3_enabled():
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    else:
        state_dict = trainer.model.state_dict()
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer._save(constants.DIR_MODELS_TUNED+output_dir, state_dict=state_dict)

def make_supervised_data_module(tokenizer, data_args, max_len):
    """Prepare datasets and collator for supervised fine-tuning."""
    rank0_print("Loading and preprocessing data...")
    # Load and preprocess training and evaluation datasets
    train_dataset = load_and_preprocess_dataset(data_args.data_path, tokenizer, split="train", max_len=max_len)
    eval_dataset = load_and_preprocess_dataset(data_args.data_path, tokenizer, split="validate", max_len=max_len)
    # Create a data collator
    data_collator = get_data_collator(tokenizer)

    return {
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
    }

def infer_model_class_and_trainable_parameters(training_args, model_name):
    model_config = transformers.AutoConfig
    if "bert" in model_name:
        model_class = transformers.EncoderDecoderModel
    elif "t5" in model_name:
        model_class = transformers.AutoModelForSeq2SeqLM
    else: 
        model_class = transformers.AutoModelForCausalLM
    tokenizer_class = transformers.AutoTokenizer
    return model_config, model_class, tokenizer_class, training_args

class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience: int, threshold: float = 0.0):
        self.patience = patience
        self.threshold = threshold
        self.best_score = None
        self.patience_counter = 0

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        # Retrieve the ROUGE-L score from the metrics (computed by compute_metrics)
        current_score = metrics.get("eval_rougeL", None)
        print(f"Current ROUGE-L score: {current_score}")
        if current_score is None:
            print("EarlyStopping: No ROUGE-L score found in evaluation metrics.")
            return

        # Initialize best_score if this is the first evaluation
        if self.best_score is None:
            self.best_score = current_score

        # If the current score improves, reset patience counter and update best_score
        elif current_score >= self.best_score + self.threshold:
            print(f"EarlyStopping: ROUGE-L improved from {self.best_score} to {current_score}.")
            self.best_score = current_score
            self.patience_counter = 0

        # If no improvement, increase patience counter
        else:
            self.patience_counter += 1
            print(f"EarlyStopping: ROUGE-L did not improve. Patience counter increased to {self.patience_counter}.")

        # If patience counter exceeds the allowed patience, stop training
        if self.patience_counter >= self.patience:
            print("EarlyStopping: Stopping early due to no improvement in ROUGE-L.")
            control.should_training_stop = True


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    (
        model_args,
        data_args,
        training_args,
    ) = parser.parse_args_into_dataclasses()
   
    if getattr(training_args, 'deepspeed', None):
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    # Load model and tokenizer
    model_config, model_class, tokenizer_class, training_args = infer_model_class_and_trainable_parameters(
        training_args, model_args.model_name_or_path
    )

    config = model_config.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    config.use_cache = False    

    tokenizer = tokenizer_class.from_pretrained(
      model_args.model_name_or_path,
      cache_dir=training_args.cache_dir,
      model_max_length=training_args.model_max_length,
      padding_side="right",
      use_fast=False,
      trust_remote_code=True,
    )

    if model_class == transformers.EncoderDecoderModel:
        model = model_class.from_encoder_decoder_pretrained(
            model_args.model_name_or_path,
            model_args.model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
            torch_dtype=torch.bfloat16,
            use_flash_attention_2=False
        )
        model.config.decoder_start_token_id = tokenizer.cls_token_id
        model.config.pad_token_id = tokenizer.pad_token_id    
        model.config.bos_token_id = tokenizer.cls_token_id
        model.config.config.decoder_start_token_id = tokenizer.cls_token_id
        model.config.encoder.decoder_start_token_id = tokenizer.cls_token_id
        model.config.decoder.decoder_start_token_id = tokenizer.cls_token_id
    else:
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
            torch_dtype=torch.bfloat16,
            use_flash_attention_2=False
        )

    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions
        # If predictions are logits, convert them to token IDs using argmax
        if len(pred_ids.shape) > 2:
            pred_ids = np.argmax(pred_ids, axis=-1)

        
        # Remove None and invalid token IDs (e.g., negative values)
        pred_ids = [[token_id for token_id in pred if token_id is not None and token_id >= 0] for pred in pred_ids]

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = tokenizer.pad_token_id
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

        rouge = evaluate.load("rouge")
        rouge_results = rouge.compute(predictions=pred_str, references=label_str)
        return {
            "eval_rouge1": rouge_results["rouge1"],
            "eval_rouge2": rouge_results["rouge2"],
            "eval_rougeL": rouge_results["rougeL"],
        }

    # Load data
    data_module = make_supervised_data_module(
    tokenizer=tokenizer,
    data_args=data_args,
    max_len=training_args.model_max_length,
    )
    training_args.generate_config = transformers.GenerationConfig(decoder_start_token_id=model.config.decoder_start_token_id, max_new_tokens=training_args.generation_max_length, min_new_tokens=training_args.generation_min_length)
    trainer = Seq2SeqTrainer(
        model=model, tokenizer=tokenizer, args=training_args, 
        train_dataset=data_module["train_dataset"],
        eval_dataset=data_module["eval_dataset"],
        data_collator=data_module["data_collator"],
        compute_metrics=compute_metrics,  # Your compute_metrics should return ROUGE scores
        callbacks=[EarlyStoppingCallback(patience=3, threshold=0.01)]  # Set patience and threshold
    )
    trainer.train(resume_from_checkpoint=False)
    trainer.save_state()

    # Optional: print one training sample
    if local_rank == 0:
        sample_input = data_module["eval_dataset"][0]
        sample_target = sample_input["labels"]
        sample_input = {k: v.to(trainer.args.device) for k, v in sample_input.items()}
        sample_input = sample_input["input_ids"].unsqueeze(0)
        sample_output = model.generate(sample_input, max_new_tokens=286, min_new_tokens=120, num_beams=5, early_stopping=True, decoder_start_token_id=model.config.decoder_start_token_id)
        print("sample eval in: ", tokenizer.decode(sample_input[0], skip_special_tokens=False))
        print("out: ", tokenizer.decode(sample_output[0], skip_special_tokens=False))
        sample_target = torch.where(sample_target == IGNORE_TOKEN_ID, torch.tensor(1), sample_target)
        print("sample eval target: ", tokenizer.decode(sample_target, skip_special_tokens=True))
   
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    
    # Optional: push to hub
    model.push_to_hub("jomoll/roberta-base4", private=True)

if __name__ == "__main__":
    torch.cuda.empty_cache()
    train()
