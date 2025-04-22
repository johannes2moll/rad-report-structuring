####################################################################################################
# This script is used to fine-tune a phi3 model on radiology report section generation. 
# The model is fine-tuned on a dataset of free text radiology reports and their corresponding structured impression sections.
# For training, execute the corresponding bash file on a machine with a GPU.
####################################################################################################

from dataclasses import dataclass, field
from typing import Dict, Optional, List
import numpy as np
import torch
import evaluate
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
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

import constants
from utils import load_and_preprocess_dataset_llm, get_data_collator, load_llm_model

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class ModelArguments:
    model: Optional[str] = field(default="roberta-base")

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
    output_dir: str = field(default="roberta-base")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    generation_max_length: int = field(
        default=286, metadata={"help": "Maximum length of generated sequences"}
    )
    generation_min_length: int = field(
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

@dataclass
class LoraArguments:
    lora_r: int = 32
    lora_alpha: int = 32
    case_id: int = 0

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, tokenizer, output_dir: str):
    """Collects the state dict and dump to disk."""
    # check if zero3 mode enabled
    if deepspeed.is_deepspeed_zero3_enabled():
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    else:
        state_dict = trainer.model.state_dict()
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer._save(output_dir, state_dict=state_dict)
    # save tokenizer
    tokenizer.save_pretrained(output_dir)
    print("pushed model to hf hub: ", output_dir)

def make_supervised_data_module(tokenizer, data_args, max_len):
    """Prepare datasets and collator for supervised fine-tuning."""
    rank0_print("Loading and preprocessing data...")
    # Load and preprocess training and evaluation datasets
    train_dataset = load_and_preprocess_dataset_llm(data_args.data_path, tokenizer, split="train", max_len=max_len)
    eval_dataset = load_and_preprocess_dataset_llm(data_args.data_path, tokenizer, split="validate", max_len=max_len)
    # Create a data collator
    data_collator = get_data_collator(tokenizer)

    return {
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
    }

class FixedSeq2SeqTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Move all tensor inputs to the model's device
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        
        outputs = model(**inputs)
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss
def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()
    training_args.hf_name = training_args.output_dir
    training_args.output_dir = constants.DIR_MODELS_TUNED+training_args.output_dir
    
    # Load model and tokenizer
    model, tokenizer = load_llm_model(constants.LLMS[model_args.model], cache_dir='.', task="train")
    if lora_args.case_id == 0 or lora_args.case_id == 105:
        # load peft model
        if model_args.model != "gpt2" and model_args.model != "opt":
            if model_args.model == "phi3":
                target_modules = ["o_proj", "qkv_proj"]
            else:
                target_modules= ["q_proj", "k_proj", "v_proj", "o_proj"]
            lora_config = LoraConfig(
                    r=lora_args.lora_r,
                    lora_alpha=lora_args.lora_alpha,
                    target_modules=target_modules,
                    lora_dropout=0.1,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
        
            model = get_peft_model(model, lora_config)
            print(model)
            print("Chose target modules:",lora_config.target_modules)
            print("Chose r value:",lora_config.r, "alpha value:",lora_config.lora_alpha)
            model.print_trainable_parameters()
    
    # Load data
    data_module = make_supervised_data_module(
    tokenizer=tokenizer,
    data_args=data_args,
    max_len=training_args.model_max_length,
    )
    training_args.generate_config = transformers.GenerationConfig(decoder_start_token_id=model.config.decoder_start_token_id, max_new_tokens=training_args.generation_max_length, min_new_tokens=training_args.generation_min_length, max_length=None)
    trainer = FixedSeq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args, 
    train_dataset=data_module["train_dataset"],
    eval_dataset=data_module["eval_dataset"],
    data_collator=data_module["data_collator"],
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
   
    safe_save_model_for_hf_trainer(trainer=trainer, tokenizer=tokenizer, output_dir=training_args.output_dir)
    
    # Optional: push to hub
    model.push_to_hub("jomoll/"+training_args.hf_name, private=True)

if __name__ == "__main__":
    torch.cuda.empty_cache()
    train()
