import json
import argparse
import torch
import transformers
import torch.utils.data
from datasets import Dataset, load_dataset
import numpy as np
from tqdm import tqdm
import re

import constants
IGNORE_TOKEN_ID = -100

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
            try:
                generated_ids = model.generate(input_ids, attention_mask, max_new_tokens=max_gen_length, min_new_tokens= min_gen_length,decoder_start_token_id=model.config.decoder_start_token_id, num_beams=5, early_stopping=True)
            except:
                generated_ids = model.generate(input_ids, max_new_tokens=max_gen_length, min_new_tokens= min_gen_length, num_beams=5, early_stopping=True)
        decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        predictions.extend(decoded_preds)
    return predictions

def get_data_collator(tokenizer):
    """Return a data collator for seq2seq tasks."""
    return transformers.DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, label_pad_token_id=IGNORE_TOKEN_ID)


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

def load_data_old(test_data_path):
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

def load_and_preprocess_dataset(data_path, tokenizer, max_len, split="train", system_message="", batch_size=16):
    """Load and preprocess dataset in batches."""
    # Load hf dataset
    raw_data = load_dataset(data_path, split=split)

    # Preprocess data in batches
    processed_data = raw_data.map(
        lambda batch: preprocess_batch(batch, tokenizer, max_len, system_message),
        batched=True,
        batch_size=batch_size,
        desc="Running tokenizer on "+split+" dataset")

    return processed_data

def load_and_preprocess_dataset_llm(data_path, tokenizer, max_len, split="train", system_message="", batch_size=16, case_id=0):
    """Load and preprocess dataset in batches."""
    
    # Load hf dataset
    raw_data = load_dataset(data_path, split=split)

    # Preprocess data in batches
    processed_data = raw_data.map(
        lambda batch: preprocess_batch_llm(batch, tokenizer, max_len, system_message, case_id),
        batched=True,
        batch_size=batch_size,
        desc="Running tokenizer on "+split+" dataset")

    return processed_data    

def load_llm_model(model_name: str, cache_dir: str, model_max_length: int):
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

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

def parse_args_calc():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", type=str, required=True, help="Path to the prediction file")
    parser.add_argument("--ref_file", type=str, required=True, help="Path to the reference file")
    parser.add_argument("--output_file", type=str, required=True, help="File to save the results")
    args = parser.parse_args()
    return args

def parse_args_run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--cache_dir", type=str, required=True, help="Cache directory for the tokenizer and model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the test data in JSON format")
    parser.add_argument("--output_file", type=str, required=True, help="File to save the predictions")
    parser.add_argument("--max_input_length", type=int, default=8192, help="Maximum sequence length for the model")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--max_gen_length", type=int, default=256, help="Maximum generation length for the model")
    parser.add_argument("--min_gen_length", type=int, default=1, help="Minimum generation length for the model")
    args = parser.parse_args()
    return args

def parse_args_run_llm():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_id", type=int, required=True, help="Case number for the prompt")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--cache_dir", type=str, required=True, help="Cache directory for the tokenizer and model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the test data in JSON format")
    parser.add_argument("--output_file", type=str, required=True, help="File to save the predictions")
    parser.add_argument("--max_input_length", type=int, default=8192, help="Maximum sequence length for the model")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--max_gen_length", type=int, default=256, help="Maximum generation length for the model")
    parser.add_argument("--min_gen_length", type=int, default=1, help="Minimum generation length for the model")
    args = parser.parse_args()
    return args

def preprocess_batch(batch, tokenizer: transformers.PreTrainedTokenizer, max_len: int, system_message: str = ""):
    """Preprocess a batch of samples."""
    # Validate fields
    original = batch.get("original_report", [])
    structured = batch.get("structured_report", [])
    if not original or not structured:
        raise ValueError("Batch does not contain 'original_report' or 'structured_report' fields.")

    # Tokenize the input and target text
    input_text = [system_message + report for report in original]
    inputs = tokenizer(input_text, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
    target_text = structured
    labels = tokenizer(target_text, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")["input_ids"]
    labels[labels == tokenizer.pad_token_id] = IGNORE_TOKEN_ID
    inputs["labels"] = labels

    return inputs

def preprocess_batch_llm(batch, tokenizer: transformers.PreTrainedTokenizer, max_len: int, system_message: str = "", case_id: int = 0):
    """Preprocess a batch of samples."""
    START_PREFIX = "<|user|> Your task is to improve the formatting of a radiology report to a clear and concise radiology report with section headings. \n Sections to include (if applicable):\n 1. Exam Type: Provide the specific type of examination conducted. \n2. History: Provide a brief clinical history and state the clinical question or suspicion that prompted the imaging.\n3. Technique: Describe the examination technique and any specific protocols used.\n 4. Comparison: Note any prior imaging studies reviewed for comparison with the current exam.\n 5. Findings: \nDescribe all positive observations and any relevant negative observations for each organ or organ system under distinct headers.\nStart with the organ system name followed by a colon, then list observations. Here is the corresponding template:\n Organ 1:\n- Observation 1\n \n Organ 2:\n- Observation 1\n- Observation 2\n\n6. Impression: Summarize the key findings with a numbered list from the most to the least clinically relevant. Ensure all findings are numbered. \n\nOriginal report: . \n\n"
    ICL_PROMPT_1 = "<|user|> You will be provided with a sample input and corresponding answer. Your task is to generate a fitting answer to the new input. Sample input: 'EXAMINATION:  Chest radiograph\n \n INDICATION:  ___ year old woman with DKA, concern for infection.  // Evaluate\n for pneumonia\n \n TECHNIQUE:  Portable AP chest radiograph.\n \n COMPARISON:  Chest radiograph from ___\n \n FINDINGS: \n \n Bilateral diffuse fluffy opacities are increased from previous examination\n suggestive of pulmonary edema.  Loss of visualization of the bilateral\n hemidiaphragms suggests layering effusions.  Stable cardiomegaly.  An impacted\n fracture of the left humeral surgical neck with periosteal new bone formation\n and dislocation of the humerus from glenoid is chronic.\n \n IMPRESSION: \n \n Moderate pulmonary edema and layering pleural effusion.  In view of extensive\n pulmonary changes, this impossible to exclude super infection.'\nSample output: 'Exam Type: Chest radiograph\n\nHistory: Adult female with diabetic ketoacidosis (DKA), concern for infection. Evaluation for pneumonia.\n\nTechnique: Portable anteroposterior (AP) chest radiograph.\n\nComparison: Prior chest radiograph available for comparison.\n\nFindings:\nLungs and Airways:\n- Bilateral diffuse fluffy opacities, suggestive of pulmonary edema.\n\nPleura:\n- Loss of visualization of the bilateral hemidiaphragms, suggesting layering effusions.\n\nCardiovascular:\n- Stable cardiomegaly.\n\nMusculoskeletal and Chest Wall:\n- Chronic impacted fracture of the left humeral surgical neck with periosteal new bone formation.\n- Dislocation of the humerus from the glenoid.\n\nImpression:\n1. Moderate pulmonary edema.\n2. Layering pleural effusions.\n3. Chronic fracture and dislocation involving the left humeral surgical neck and glenoid.'\n \n New input: \n"
    ICL_PROMPT_2 = "<|user|> You will be provided with two sample inputs and corresponding answers. Your task is to generate a fitting answer to the new input. Sample input 1: 'EXAMINATION:  Chest radiograph\n \n INDICATION:  ___ year old woman with DKA, concern for infection.  // Evaluate\n for pneumonia\n \n TECHNIQUE:  Portable AP chest radiograph.\n \n COMPARISON:  Chest radiograph from ___\n \n FINDINGS: \n \n Bilateral diffuse fluffy opacities are increased from previous examination\n suggestive of pulmonary edema.  Loss of visualization of the bilateral\n hemidiaphragms suggests layering effusions.  Stable cardiomegaly.  An impacted\n fracture of the left humeral surgical neck with periosteal new bone formation\n and dislocation of the humerus from glenoid is chronic.\n \n IMPRESSION: \n \n Moderate pulmonary edema and layering pleural effusion.  In view of extensive\n pulmonary changes, this impossible to exclude super infection.'\nSample output 1: 'Exam Type: Chest radiograph\n\nHistory: Adult female with diabetic ketoacidosis (DKA), concern for infection. Evaluation for pneumonia.\n\nTechnique: Portable anteroposterior (AP) chest radiograph.\n\nComparison: Prior chest radiograph available for comparison.\n\nFindings:\nLungs and Airways:\n- Bilateral diffuse fluffy opacities, suggestive of pulmonary edema.\n\nPleura:\n- Loss of visualization of the bilateral hemidiaphragms, suggesting layering effusions.\n\nCardiovascular:\n- Stable cardiomegaly.\n\nMusculoskeletal and Chest Wall:\n- Chronic impacted fracture of the left humeral surgical neck with periosteal new bone formation.\n- Dislocation of the humerus from the glenoid.\n\nImpression:\n1. Moderate pulmonary edema.\n2. Layering pleural effusions.\n3. Chronic fracture and dislocation involving the left humeral surgical neck and glenoid.'\n \n Sample input 2: 'INDICATION:  History: ___M with fever  // Eval for pneumonia    COMPARISON:  ___.    FINDINGS:     PA and lateral chest radiographs. The patient is rotated to the right. There  is an inferior approach hemodialysis catheter terminating in the right atrium.  The lungs are clear. There is no pleural effusion or pneumothorax.  There may  be mild pulmonary vascular engorgement, but no interstitial edema. The  cardiomediastinal silhouette is stable.    IMPRESSION:     No acute cardiopulmonary process.',\n Sample output 2: 'Exam Type: PA and lateral chest radiographs.  History: Male patient with fever. Evaluation for pneumonia.  Technique: Posteroanterior (PA) and lateral views of the chest were obtained.  Findings: Lungs and Airways: - The lungs are clear.  Pleura: - No pleural effusion or pneumothorax identified.  Cardiovascular: - Mild pulmonary vascular engorgement noted, but no interstitial edema. - Cardiomediastinal silhouette is stable.  Tubes, Catheters, and Support Devices: - Inferior approach hemodialysis catheter terminating in the right atrium.  Musculoskeletal and Chest Wall: - Patient is rotated to the right.  Impression: 1. No evidence of acute cardiopulmonary disease. 2. Presence of hemodialysis catheter in the right atrium. 3. Mild pulmonary vascular engorgement without interstitial edema.'\n\nNew input: \n"
    cases = {
        # discrete prompting: null, prefix, and in-context examples (1,2,4)
        0: {},  # No prompt, control case
        1: {"prompt": f"\n{ICL_PROMPT_1}\n Input: "},
        2: {"prompt": f"\n{ICL_PROMPT_2}\n Input: "},
        5: {"start_prefix": START_PREFIX},  # Instruction-only prompting
        10:{"both": f"{START_PREFIX}\n{ICL_PROMPT_1}\n Input: "},
    }
    # Validate fields
    original = batch.get("original_report", [])
    structured = batch.get("structured_report", [])
    if not original or not structured:
        raise ValueError("Batch does not contain 'original_report' or 'structured_report' fields.")
    # Tokenize the input and target text
    if case_id in cases and 'start_prefix' in cases[case_id]:
        # Apply instruction-based prompt (zero-shot style)
        input_text = [cases[case_id]['start_prefix'] + f"{original}" + "\n Output: " for original in original]
        
    elif case_id in cases and 'prompt' in cases[case_id]:
        # Apply in-context learning prompt (few-shot style)
        input_text = [cases[case_id]['prompt'] + f"{original}" + "\n Output: " for original in original]
        
    elif case_id in cases and 'both' in cases[case_id]:
        # Apply both instruction and in-context learning prompts
        input_text = [cases[case_id]['both'] + f"{original}" + "\n Output: " for original in original]
    else:
        input_text = f"{original}" + "\n Output: "
            
    inputs = tokenizer(input_text, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
    target_text = structured
    labels = tokenizer(target_text, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")["input_ids"]
    labels[labels == tokenizer.pad_token_id] = IGNORE_TOKEN_ID
    inputs["labels"] = labels

    return inputs