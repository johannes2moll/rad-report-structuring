import json
import argparse
import torch
import transformers
import torch.utils.data
from datasets import Dataset
import numpy as np
from tqdm import tqdm
import re

import constants

def load_model(model_name: str, cache_dir: str, model_max_length: int):
    if "bert" in model_name:
        model = transformers.EncoderDecoderModel.from_pretrained(model_name, trust_remote_code=True)
    elif "t5" in model_name:
        model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True)
    else: 
        model = transformers.AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True)
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def load_data(test_data_path):
    """Preprocess the test data for inference."""
    
    with open(test_data_path+"test_unstructured.json", "r") as f_o , open(test_data_path+"test_structured.json", "r") as f_s:
        lines_original = json.load(f_o)
        lines_structured = json.load(f_s)
        assert len(lines_original) == len(lines_structured)

        list_original = []
        list_structured = []
        for idx, original in enumerate(lines_original):
                elem_original = original.strip()
                structured = lines_structured[idx]
                elem_structured = structured.rstrip()

                list_original.append(elem_original)
                list_structured .append(elem_structured)

        
        data_list = [
            {"idx": i, "inputs": list_original[i], "label": list_structured[i]}
            for i in range(len(list_original))
        ]
        dataset = Dataset.from_list(data_list)

        return dataset

def preprocess_function(examples, tokenizer, max_len: int):
    # tokenize examples['sentence'] (finding) and examples['text_label'] (summary)
    model_inputs = tokenizer(
        examples["inputs"],
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    labels = tokenizer(
        examples["label"],
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    labels = labels["input_ids"]
    labels[
        labels == tokenizer.pad_token_id
    ] = -100  # replace padding token id's by -100 so it's ignored by the loss
    model_inputs["labels"] = labels
    return model_inputs

def generate_predictions(model, tokenizer, test_loader, device, max_gen_length: int, min_gen_length: int,):
    model.eval()
    model.to(device)
    
    predictions = []
    
    # Initialize tqdm progress bar, setting the total number of batches
    progress_bar = tqdm(test_loader, desc="Generating predictions", unit="batch")

    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask=batch["attention_mask"].to(device)
        with torch.no_grad():
            generated_ids = model.generate(input_ids, attention_mask, max_new_tokens=max_gen_length, min_new_tokens= min_gen_length,decoder_start_token_id=model.config.decoder_start_token_id, num_beams=5, early_stopping=True)
        decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        predictions.extend(decoded_preds)
    return predictions



def parse_args_run():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--cache_dir", type=str, required=True, help="Cache directory for the tokenizer and model")
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to the test data in JSON format")
    parser.add_argument("--output_file", type=str, required=True, help="File to save the predictions")
    parser.add_argument("--max_input_length", type=int, default=8192, help="Maximum sequence length for the model")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--max_gen_length", type=int, default=256, help="Maximum generation length for the model")
    parser.add_argument("--min_gen_length", type=int, default=1, help="Minimum generation length for the model")
    

    args = parser.parse_args()
    return args

def parse_args_calc():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", type=str, required=True, help="Path to the prediction file")
    parser.add_argument("--ref_file", type=str, required=True, help="Path to the reference file")
    parser.add_argument("--output_file", type=str, required=True, help="File to save the results")
    args = parser.parse_args()
    return args

def extract_sections(report):
    """Extracts Findings and Impression sections from a radiology report."""
    # Use case-insensitive regex to locate sections
    findings_match = re.search(r'(?i)findings\s*:', report)
    impression_match = re.search(r'(?i)impression\s*:', report)
    
    # Ensure both sections exist
    if not findings_match or not impression_match:
        return None, None
    
    # Extract Findings section
    findings_start = findings_match.end()
    findings_end = impression_match.start()
    findings = report[findings_start:findings_end].strip()
    
    # Extract Impression section
    impression_start = impression_match.end()
    impression = report[impression_start:].strip()
    
    return findings, impression

def get_lists(prediction_file, reference_file):
    with open(prediction_file, "r") as f:
        predictions = json.load(f)
    with open(reference_file, "r") as f:
        references = json.load(f)
    assert len(predictions) == len(references)

    ref_findings_list = []
    ref_impression_list = []
    pred_findings_list = []
    pred_impression_list = []

    for idx, ref in enumerate(references):
        # Extract Findings and Impression sections from original report
        ref_findings, ref_impression = extract_sections(ref)
        # Extract Findings and Impression sections from generated report
        pred_findings, pred_impression = extract_sections(predictions[idx])
        
        # Append to lists
        ref_findings_list.append(ref_findings)
        ref_impression_list.append(ref_impression)
        pred_findings_list.append(pred_findings)
        pred_impression_list.append(pred_impression)
    
    return ref_findings_list, ref_impression_list, pred_findings_list, pred_impression_list
