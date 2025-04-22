import json
import argparse
import torch
import transformers
import torch.utils.data
from datasets import load_dataset
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
    predictions = []
    # Initialize tqdm progress bar, setting the total number of batches
    progress_bar = tqdm(test_loader, desc="Generating predictions", unit="batch")
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask=batch["attention_mask"].to(device)
        with torch.no_grad():
            try:
                generated_ids = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=max_gen_length, min_new_tokens= min_gen_length,decoder_start_token_id=model.config.decoder_start_token_id, num_beams=5, early_stopping=True, max_length=None)
            except:
                generated_ids = model.generate(input_ids, max_new_tokens=max_gen_length, min_new_tokens=min_gen_length, num_beams=5, early_stopping=True, max_length=None)
        decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        predictions.extend(decoded_preds)
        
        # only generate the first 100 samples
        if len(predictions) >= 100:
            print("Only generating the first 100 samples")
            break
    return predictions

def get_data_collator(tokenizer):
    """Return a data collator for seq2seq tasks."""
    return transformers.DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, label_pad_token_id=IGNORE_TOKEN_ID)

def get_lists(prediction_file, reference_path, dataset_name="mimic", split="test_reviewed"):
    with open(prediction_file, "r") as f:
        predictions = json.load(f)
    print(len(predictions), "predictions loaded")
    references = load_dataset(reference_path, split=split)
    selected_references = []
    for i in range (len(references)):
        if dataset_name in references["id"][i]:
            selected_references.append(references["structured_report"][i])
    print(len(selected_references), "references loaded")
    if len(predictions) < len(selected_references):
        print("Warning: Number of predictions is less than number of references.")
        print("Number of predictions:", len(predictions))
        print("Number of references:", len(selected_references))
        print("Truncating references to match number of predictions.")
        selected_references = selected_references[:len(predictions)]

    # for t5 models, reformat the predictions by adding newline characters
    try:
        if predictions[0].count("\n") == 0:
            predictions = reformat_radiology_output(predictions)
            print("Reformatted predictions for T5 models.")
            #print("Sample prediction:", predictions[0])
    except Exception as e:
        print("Error:", e)

    ref_findings_list = []
    ref_impression_list = []
    pred_findings_list = []
    pred_impression_list = []

    for idx, ref in enumerate(selected_references):
        # Extract Findings and Impression sections from original report
        ref_findings, ref_impression = extract_sections(ref)
        # Extract Findings and Impression sections from generated report
        pred_findings, pred_impression = extract_sections(predictions[idx])
        # remove <pad> tokens from predictions
        try: pred_findings = pred_findings.replace("<pad>", "").strip()
        except: continue
        try: pred_impression = pred_impression.replace("<pad>", "").strip()
        except: continue
        # Append to lists
        ref_findings_list.append(ref_findings)
        ref_impression_list.append(ref_impression)
        pred_findings_list.append(pred_findings)
        pred_impression_list.append(pred_impression)
    
    return ref_findings_list, ref_impression_list, pred_findings_list, pred_impression_list

def load_and_preprocess_dataset(data_path, tokenizer, max_len, split="train", system_message="<|user|>", batch_size=16):
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

def load_and_preprocess_dataset_llm(data_path, tokenizer, max_len, split="train", system_message="", batch_size=1, case_id=0):
    """Load and preprocess dataset in batches."""
    # For open-end generation, padding causes issues which is why we use batch size of 1
    # Load hf dataset
    raw_data = load_dataset(data_path, split=split)

    # Preprocess data in batches
    processed_data = raw_data.map(
        lambda batch: preprocess_batch_llm(batch, tokenizer, max_len, system_message, case_id, task=split),
        batched=True,
        batch_size=batch_size,
        desc="Running tokenizer on "+split+" dataset")

    return processed_data    

def load_llm_model(model_name: str, cache_dir: str, task: str = "train"):
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
    #if task != "train":
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def load_model(model_name: str, task: str = "train"):
    
    config = transformers.AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if re.search(r'(?i)bert', model_name):
        if task == "train": 
            model = transformers.EncoderDecoderModel.from_encoder_decoder_pretrained(model_name, model_name, config=config, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="right", use_fast=False)
        else: 
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = transformers.EncoderDecoderModel.from_pretrained(model_name, trust_remote_code=True).to(device)
            if re.search(r'(?i)PM', model_name): # roberta-PM needs local tokenizer, if tokenizer not available on hf, use locally saved model
                tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="right", use_fast=False)
            else: tokenizer = transformers.AutoTokenizer.from_pretrained("FacebookAI/roberta-base", trust_remote_code=True, padding_side="right", use_fast=False)
        model.config.decoder_start_token_id = tokenizer.cls_token_id
        model.config.bos_token_id = tokenizer.cls_token_id
        print("Loaded EncoderDecoderModel")
    elif "t5" in model_name.lower() or "scifive" in model_name.lower():
        model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True)
        if task == "train": 
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="right", use_fast=False)
        else:
            tokenizer = transformers.AutoTokenizer.from_pretrained("t5-base", trust_remote_code=True, padding_side="right", use_fast=False)
        print("Loaded AutoModelForSeq2SeqLM")
    else: 
        model = transformers.AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True)
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="right", use_fast=False)
        print("Loaded AutoModelForCausalLM")
    
    #if task != "train":
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id   
    
    return model, tokenizer

def parse_args_calc():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file_mimic", type=str, required=True, help="Path to the prediction file")
    parser.add_argument("--pred_file_chexpert", type=str, required=True, help="Path to the reference dataset")
    parser.add_argument("--ref_data_path", type=str, required=True, help="Path to the reference dataset")
    parser.add_argument("--output_file", type=str, required=True, help="File to save the results")
    args = parser.parse_args()
    return args

def parse_args_run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--cache_dir", type=str, required=True, help="Cache directory for the tokenizer and model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the test data in JSON format")
    parser.add_argument("--output_file_mimic", type=str, required=True, help="File to save the predictions")
    parser.add_argument("--output_file_chexpert", type=str, required=True, help="File to save the predictions")
    parser.add_argument("--max_input_length", type=int, default=8192, help="Maximum sequence length for the model")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--max_gen_length", type=int, default=256, help="Maximum generation length for the model")
    parser.add_argument("--min_gen_length", type=int, default=1, help="Minimum generation length for the model")
    parser.add_argument("--load_from_hf", type=bool, help="Load model from Hugging Face model hub")
    args = parser.parse_args()
    return args

def parse_args_run_llm():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_id", type=int, required=True, help="Case number for the prompt")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--cache_dir", type=str, required=True, help="Cache directory for the tokenizer and model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the test data in JSON format")
    parser.add_argument("--output_file_mimic", type=str, required=True, help="File to save the predictions")
    parser.add_argument("--output_file_chexpert", type=str, required=True, help="File to save the predictions")
    parser.add_argument("--max_input_length", type=int, default=8192, help="Maximum sequence length for the model")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--max_gen_length", type=int, default=256, help="Maximum generation length for the model")
    parser.add_argument("--min_gen_length", type=int, default=1, help="Minimum generation length for the model")
    parser.add_argument("--lora_r", type=int, default=8, help="LORA r parameter")
    parser.add_argument("--lora_alpha", type=int, default=8, help="LORA alpha parameter")
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
    inputs["attention_mask"] = inputs["input_ids"].ne(tokenizer.pad_token_id)  # Add attention mask

    target_text = ["<|assistant|>" + s for s in structured]
    labels = tokenizer(target_text, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")["input_ids"]
    labels[labels == tokenizer.pad_token_id] = IGNORE_TOKEN_ID  # Mask out padding for loss computation

    inputs["labels"] = labels

    return inputs

def preprocess_batch_llm(batch, tokenizer: transformers.PreTrainedTokenizer, max_len: int, system_message: str = "", case_id: int = 0, task: str = "test"):
    """Preprocess a batch of samples."""
    # Predefined prompts for ICL examples
    START_PREFIX = "<|system|> You are a radiology expert.<|end|> <|user|> Your task is to improve the formatting of a radiology report to a clear and concise radiology report with section headings.\n\nGuidelines:\n1. Section Headers: Each section should start with the section header followed by a colon. Provide the relevant information as specified for each section.\n2. Identifiers: Remove sentences where identifiers have been replaced with consecutive underscores ('___'). Also, remove sentences containing the following identifiers: dates, surnames, first names, healthcare providers, vendors, institutions. Important: keep sex and age information if present.\n Findings and Impression Sections: Focus solely on the current examination results. Do not reference previous studies or historical data.\n4. Content Restrictions: Strictly include only the content that is relevant to the structured sections provided. Do not add or extrapolate information beyond what is found in the original report.\n\nSections to include (if applicable):\n1. Exam Type: Provide the specific type of examination conducted.\n2. History: Provide a brief clinical history and state the clinical question or suspicion that prompted the imaging.\n3. Technique: Describe the examination technique and any specific protocols used.\n4. Comparison: Note any prior imaging studies reviewed for comparison with the current exam.\n5. Findings:\nDescribe all positive observations and any relevant negative observations for each organ system, organizing them under specific headers. Follow this template for listing your observations:\nHeader 1:\n- Observation 1\n- ...\nHeader 2:\n- Observation 1\n- Observation 2\n- ...\nUse only the following headers for organ systems:\n- Lungs and Airways\n- Pleura\n- Cardiovascular \n- Hila and Mediastinum \n- Tubes, Catheters, and Support Devices \n- Musculoskeletal and Chest Wall\n- Abdominal\n- Other\nImportant: Do not use any headers other than those listed above. Only use the specified headers that correspond to the organ systems mentioned in the original radiology report.\n6. Impression: Summarize the key findings with a numbered list from the most to the least clinically relevant. Ensure all findings are numbered.\n\nThe radiology report to improve is the following:\n"
    ICL_PROMPT_1 = "<|system|> You are a radiology expert.<|end|> <|user|> You will be provided with a sample input and corresponding answer. Your task is to generate a fitting answer to the new input. Sample input: 'EXAMINATION:  Chest radiograph\n \n INDICATION:  ___ year old woman with DKA, concern for infection.  // Evaluate\n for pneumonia\n \n TECHNIQUE:  Portable AP chest radiograph.\n \n COMPARISON:  Chest radiograph from ___\n \n FINDINGS: \n \n Bilateral diffuse fluffy opacities are increased from previous examination\n suggestive of pulmonary edema.  Loss of visualization of the bilateral\n hemidiaphragms suggests layering effusions.  Stable cardiomegaly.  An impacted\n fracture of the left humeral surgical neck with periosteal new bone formation\n and dislocation of the humerus from glenoid is chronic.\n \n IMPRESSION: \n \n Moderate pulmonary edema and layering pleural effusion.  In view of extensive\n pulmonary changes, this impossible to exclude super infection.'\nSample output: 'Exam Type: Chest radiograph\n\nHistory: Adult female with diabetic ketoacidosis (DKA), concern for infection. Evaluation for pneumonia.\n\nTechnique: Portable anteroposterior (AP) chest radiograph.\n\nComparison: Prior chest radiograph available for comparison.\n\nFindings:\nLungs and Airways:\n- Bilateral diffuse fluffy opacities, suggestive of pulmonary edema.\n\nPleura:\n- Loss of visualization of the bilateral hemidiaphragms, suggesting layering effusions.\n\nCardiovascular:\n- Stable cardiomegaly.\n\nMusculoskeletal and Chest Wall:\n- Chronic impacted fracture of the left humeral surgical neck with periosteal new bone formation.\n- Dislocation of the humerus from the glenoid.\n\nImpression:\n1. Moderate pulmonary edema.\n2. Layering pleural effusions.\n3. Chronic fracture and dislocation involving the left humeral surgical neck and glenoid.'\n \n New input: \n"
    ICL_PROMPT_2_BOTH = "<|system|> You are a radiology expert.<|end|> <|user|> You will be provided with two sample inputs and corresponding answers. Your task is to generate a fitting answer to the new input. Sample input 1: 'EXAMINATION:  Chest radiograph\n \n INDICATION:  ___ year old woman with DKA, concern for infection.  // Evaluate\n for pneumonia\n \n TECHNIQUE:  Portable AP chest radiograph.\n \n COMPARISON:  Chest radiograph from ___\n \n FINDINGS: \n \n Bilateral diffuse fluffy opacities are increased from previous examination\n suggestive of pulmonary edema.  Loss of visualization of the bilateral\n hemidiaphragms suggests layering effusions.  Stable cardiomegaly.  An impacted\n fracture of the left humeral surgical neck with periosteal new bone formation\n and dislocation of the humerus from glenoid is chronic.\n \n IMPRESSION: \n \n Moderate pulmonary edema and layering pleural effusion.  In view of extensive\n pulmonary changes, this impossible to exclude super infection.'\nSample output 1: 'Exam Type: Chest radiograph\n\nHistory: Adult female with diabetic ketoacidosis (DKA), concern for infection. Evaluation for pneumonia.\n\nTechnique: Portable anteroposterior (AP) chest radiograph.\n\nComparison: Prior chest radiograph available for comparison.\n\nFindings:\nLungs and Airways:\n- Bilateral diffuse fluffy opacities, suggestive of pulmonary edema.\n\nPleura:\n- Loss of visualization of the bilateral hemidiaphragms, suggesting layering effusions.\n\nCardiovascular:\n- Stable cardiomegaly.\n\nMusculoskeletal and Chest Wall:\n- Chronic impacted fracture of the left humeral surgical neck with periosteal new bone formation.\n- Dislocation of the humerus from the glenoid.\n\nImpression:\n1. Moderate pulmonary edema.\n2. Layering pleural effusions.\n3. Chronic fracture and dislocation involving the left humeral surgical neck and glenoid.'\n \n Sample input 2: 'NARRATIVE: RADIOGRAPHIC EXAMINATION OF THE CHEST: 6/3/17   CLINICAL HISTORY: 61 years of age, Female, R o infiltarate..AML  (acute myeloblastic leukemia)   COMPARISON: 06/2017   PROCEDURE COMMENTS: Two views of the chest.    FINDINGS:   Unchanged position of the left upper extremity PICC line. Again seen  are surgical clips projecting over the right hemithorax. The  cardiomediastinal silhouette is stable in appearance. Increased  stranding opacities are noted in the left retrocardiac region. Subtle  stranding opacities in the right upper lung zone are unchanged..  There are no pleural or significant bony abnormalities. Absence of  the right breast shadow compatible with prior mastectomy.   IMPRESSION:   1.  Interval development of a band of increased linear stranding  opacities in the left retrocardiac region. Although this may  represent subsegmental atelectasis, an early or developing  consolidation could have similar appearance. Recommend clinical  correlation.       ACCESSION NUMBER: RUFLZXH This report has been anonymized. All dates are offset from the actual dates by a fixed interval associated with the patient.'\n\n Sample Ouput 2:'Exam Type: Chest Radiographic Examination  History: 61-year-old female with a history of acute myeloblastic leukemia (AML) presenting with suspicion of an infiltrate.  Technique: Two-view radiographic examination of the chest.  Findings: Tubes, Catheters, and Support Devices: - Unchanged position of the left upper extremity peripherally inserted central catheter (PICC) line.  Cardiovascular: - Stable appearance of the cardiomediastinal silhouette.  Lungs and Airways: - Increased stranding opacities in the left retrocardiac region. - Unchanged subtle stranding opacities in the right upper lung zone.  Musculoskeletal and Chest Wall: - Surgical clips over the right hemithorax. - Absence of the right breast shadow, compatible with prior mastectomy.  Other: - No pleural or significant bony abnormalities noted.  Impression: 1. Interval development of increased linear stranding opacities in the left retrocardiac region, which may represent subsegmental atelectasis or an early or developing consolidation. Clinical correlation is recommended.'\n\nNew input: \n"
    ICL_PROMPT_2_MIMIC = "<|system|> You are a radiology expert.<|end|> <|user|> You will be provided with a sample input and corresponding answer. Your task is to generate a fitting answer to the new input. Sample input: 'EXAMINATION:  Chest radiograph\n \n INDICATION:  ___ year old woman with DKA, concern for infection.  // Evaluate\n for pneumonia\n \n TECHNIQUE:  Portable AP chest radiograph.\n \n COMPARISON:  Chest radiograph from ___\n \n FINDINGS: \n \n Bilateral diffuse fluffy opacities are increased from previous examination\n suggestive of pulmonary edema.  Loss of visualization of the bilateral\n hemidiaphragms suggests layering effusions.  Stable cardiomegaly.  An impacted\n fracture of the left humeral surgical neck with periosteal new bone formation\n and dislocation of the humerus from glenoid is chronic.\n \n IMPRESSION: \n \n Moderate pulmonary edema and layering pleural effusion.  In view of extensive\n pulmonary changes, this impossible to exclude super infection.'\nSample output: 'Exam Type: Chest radiograph\n\nHistory: Adult female with diabetic ketoacidosis (DKA), concern for infection. Evaluation for pneumonia.\n\nTechnique: Portable anteroposterior (AP) chest radiograph.\n\nComparison: Prior chest radiograph available for comparison.\n\nFindings:\nLungs and Airways:\n- Bilateral diffuse fluffy opacities, suggestive of pulmonary edema.\n\nPleura:\n- Loss of visualization of the bilateral hemidiaphragms, suggesting layering effusions.\n\nCardiovascular:\n- Stable cardiomegaly.\n\nMusculoskeletal and Chest Wall:\n- Chronic impacted fracture of the left humeral surgical neck with periosteal new bone formation.\n- Dislocation of the humerus from the glenoid.\n\nImpression:\n1. Moderate pulmonary edema.\n2. Layering pleural effusions.\n3. Chronic fracture and dislocation involving the left humeral surgical neck and glenoid.'\n \n Sample input 2: 'INDICATION:  History: ___M with fever  // Eval for pneumonia    COMPARISON:  ___.    FINDINGS:     PA and lateral chest radiographs. The patient is rotated to the right. There  is an inferior approach hemodialysis catheter terminating in the right atrium.  The lungs are clear. There is no pleural effusion or pneumothorax.  There may  be mild pulmonary vascular engorgement, but no interstitial edema. The  cardiomediastinal silhouette is stable.    IMPRESSION:     No acute cardiopulmonary process.' \n \n Sample Output 2: 'Exam Type: PA and lateral chest radiographs.  History: Male patient with fever. Evaluation for pneumonia.  Technique: Posteroanterior (PA) and lateral views of the chest were obtained.  Findings: Lungs and Airways: - The lungs are clear.  Pleura: - No pleural effusion or pneumothorax identified.  Cardiovascular: - Mild pulmonary vascular engorgement noted, but no interstitial edema. - Cardiomediastinal silhouette is stable.  Tubes, Catheters, and Support Devices: - Inferior approach hemodialysis catheter terminating in the right atrium.  Musculoskeletal and Chest Wall: - Patient is rotated to the right.  Impression: 1. No evidence of acute cardiopulmonary disease. 2. Presence of hemodialysis catheter in the right atrium. 3. Mild pulmonary vascular engorgement without interstitial edema.' \n \n New Input: \n"
    ICL_PROMPT_2_CHEXPERT = "<|system|> You are a radiology expert.<|end|> <|user|> You will be provided with a sample input and corresponding answer. Your task is to generate a fitting answer to the new input. Sample input: 'NARRATIVE: RADIOGRAPHIC EXAMINATION OF THE CHEST: 6/3/17   CLINICAL HISTORY: 61 years of age, Female, R o infiltarate..AML  (acute myeloblastic leukemia)   COMPARISON: 06/2017   PROCEDURE COMMENTS: Two views of the chest.    FINDINGS:   Unchanged position of the left upper extremity PICC line. Again seen  are surgical clips projecting over the right hemithorax. The  cardiomediastinal silhouette is stable in appearance. Increased  stranding opacities are noted in the left retrocardiac region. Subtle  stranding opacities in the right upper lung zone are unchanged..  There are no pleural or significant bony abnormalities. Absence of  the right breast shadow compatible with prior mastectomy.   IMPRESSION:   1.  Interval development of a band of increased linear stranding  opacities in the left retrocardiac region. Although this may  represent subsegmental atelectasis, an early or developing  consolidation could have similar appearance. Recommend clinical  correlation.       ACCESSION NUMBER: RUFLZXH This report has been anonymized. All dates are offset from the actual dates by a fixed interval associated with the patient.' \n\n Sample Output: 'Exam Type: Chest Radiographic Examination  History: 61-year-old female with a history of acute myeloblastic leukemia (AML) presenting with suspicion of an infiltrate.  Technique: Two-view radiographic examination of the chest.  Findings: Tubes, Catheters, and Support Devices: - Unchanged position of the left upper extremity peripherally inserted central catheter (PICC) line.  Cardiovascular: - Stable appearance of the cardiomediastinal silhouette.  Lungs and Airways: - Increased stranding opacities in the left retrocardiac region. - Unchanged subtle stranding opacities in the right upper lung zone.  Musculoskeletal and Chest Wall: - Surgical clips over the right hemithorax. - Absence of the right breast shadow, compatible with prior mastectomy.  Other: - No pleural or significant bony abnormalities noted.  Impression: 1. Interval development of increased linear stranding opacities in the left retrocardiac region, which may represent subsegmental atelectasis or an early or developing consolidation. Clinical correlation is recommended.' \n\n Sample Input 2: 'NARRATIVE: SINGLE VIEW PORTABLE CHEST: 1-19-2015 CLINICAL HISTORY: 61-year-old man with history of heart transplant. COMPARISON: 1/19/2015 FINDINGS: There is redemonstration of right internal jugular central venous line, right internal jugular sheath, two mediastinal drains, sternotomy wires, and mediastinal surgical clips. Lung volumes have increased compared with the prior examination. There is decreased pulmonary edema and decreased bibasilar atelectasis. Small bilateral effusions persist. There is also decreased soft tissue edema. IMPRESSION: IMPROVED LUNG VOLUMES WITH DECREASED BIBASILAR ATELECTASIS, AS WELL AS DECREASED PULMONARY EDEMA. END OF IMPRESSION: SUMMARY:4-POSSIBLE SIGNIFICANT FINDINGS, MAY NEED ACTION I have personally reviewed the images for this examination and agree with the report transcribed above. By: scroger, ashlyn  on: 1/19/2015   ACCESSION NUMBER: YcxarrDeV This report has been anonymized. All dates are offset from the actual dates by a fixed interval associated with the patient.' \n\n Sample Output 2: 'Exam Type: Single view portable chest radiograph.  History: 61-year-old man with a history of heart transplant.  Technique: Portable anteroposterior chest radiograph.  Findings: Tubes, Catheters, and Support Devices: - Presence of right internal jugular central venous line. - Right internal jugular sheath observed. - Two mediastinal drains in situ. - Sternotomy wires and mediastinal surgical clips are noted.  Lungs and Airways: - Increased lung volumes compared to the prior examination. - Decreased pulmonary edema. - Decreased bibasilar atelectasis.  Pleura: - Small bilateral pleural effusions persist.  Other: - Decreased soft tissue edema.  Impression: 1. Improved lung volumes with decreased bibasilar atelectasis. 2. Decreased pulmonary edema. 3. Persistent small bilateral pleural effusions. 4. Decreased soft tissue edema.' \n\n New Input: \n"
    PROMPT_AND_ICL1 = "<|system|> You are a radiology expert.<|end|> <|user|> Your task is to improve the formatting of a radiology report to a clear and concise radiology report with section headings.\n\nGuidelines:\n1. Section Headers: Each section should start with the section header followed by a colon. Provide the relevant information as specified for each section.\n2. Identifiers: Remove sentences where identifiers have been replaced with consecutive underscores ('___'). Also, remove sentences containing the following identifiers: dates, surnames, first names, healthcare providers, vendors, institutions. Important: keep sex and age information if present.\n Findings and Impression Sections: Focus solely on the current examination results. Do not reference previous studies or historical data.\n4. Content Restrictions: Strictly include only the content that is relevant to the structured sections provided. Do not add or extrapolate information beyond what is found in the original report.\n\nSections to include (if applicable):\n1. Exam Type: Provide the specific type of examination conducted.\n2. History: Provide a brief clinical history and state the clinical question or suspicion that prompted the imaging.\n3. Technique: Describe the examination technique and any specific protocols used.\n4. Comparison: Note any prior imaging studies reviewed for comparison with the current exam.\n5. Findings:\nDescribe all positive observations and any relevant negative observations for each organ system, organizing them under specific headers. Follow this template for listing your observations:\nHeader 1:\n- Observation 1\n- ...\nHeader 2:\n- Observation 1\n- Observation 2\n- ...\nUse only the following headers for organ systems:\n- Lungs and Airways\n- Pleura\n- Cardiovascular \n- Hila and Mediastinum \n- Tubes, Catheters, and Support Devices \n- Musculoskeletal and Chest Wall\n- Abdominal\n- Other\nImportant: Do not use any headers other than those listed above. Only use the specified headers that correspond to the organ systems mentioned in the original radiology report.\n6. Impression: Summarize the key findings with a numbered list from the most to the least clinically relevant. Ensure all findings are numbered.\n\n You will be provided with a sample input and corresponding answer. Sample input: 'EXAMINATION:  Chest radiograph\n \n INDICATION:  ___ year old woman with DKA, concern for infection.  // Evaluate\n for pneumonia\n \n TECHNIQUE:  Portable AP chest radiograph.\n \n COMPARISON:  Chest radiograph from ___\n \n FINDINGS: \n \n Bilateral diffuse fluffy opacities are increased from previous examination\n suggestive of pulmonary edema.  Loss of visualization of the bilateral\n hemidiaphragms suggests layering effusions.  Stable cardiomegaly.  An impacted\n fracture of the left humeral surgical neck with periosteal new bone formation\n and dislocation of the humerus from glenoid is chronic.\n \n IMPRESSION: \n \n Moderate pulmonary edema and layering pleural effusion.  In view of extensive\n pulmonary changes, this impossible to exclude super infection.'\nSample output: 'Exam Type: Chest radiograph\n\nHistory: Adult female with diabetic ketoacidosis (DKA), concern for infection. Evaluation for pneumonia.\n\nTechnique: Portable anteroposterior (AP) chest radiograph.\n\nComparison: Prior chest radiograph available for comparison.\n\nFindings:\nLungs and Airways:\n- Bilateral diffuse fluffy opacities, suggestive of pulmonary edema.\n\nPleura:\n- Loss of visualization of the bilateral hemidiaphragms, suggesting layering effusions.\n\nCardiovascular:\n- Stable cardiomegaly.\n\nMusculoskeletal and Chest Wall:\n- Chronic impacted fracture of the left humeral surgical neck with periosteal new bone formation.\n- Dislocation of the humerus from the glenoid.\n\nImpression:\n1. Moderate pulmonary edema.\n2. Layering pleural effusions.\n3. Chronic fracture and dislocation involving the left humeral surgical neck and glenoid.'\n \n New input: \n"
   
    

    cases = {
        # discrete prompting: null, prefix, and in-context examples (1,2,4)
        0: {},  # No prompt, control case
        1: {"prompt": f"\n{ICL_PROMPT_1}\n Input: "},
        2: {"prompt": f"\n{ICL_PROMPT_2_BOTH}\n Input: "},
        5: {"start_prefix": START_PREFIX},  # Instruction-only prompting
        10:{"both": f"{START_PREFIX}\n{ICL_PROMPT_1}\n Input: "},
        105: {"start_prefix": START_PREFIX},  # lora with prefix
    }
    # Validate fields
    original = batch.get("original_report", [])
    structured = batch.get("structured_report", [])
    if not original or not structured:
        raise ValueError("Batch does not contain 'original_report' or 'structured_report' fields.")
    
    if task == "test" or task == "test_reviewed":
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
            input_text = "<|system|> You are a radiology expert.<|end|> <|user|>"+f"{original}" +"<|end|> \n<|assistant|> Output: "
        inputs = tokenizer(input_text, truncation=True, max_length=max_len, return_tensors="pt")    
        # Set labels to zero for LLM as we are not training the model
        inputs["labels"] = [0]

    else: 
        system_message = "<|system|><|end|> <|user|>"
        # Combine input (original) and output (structured) for training
        if case_id in cases and 'start_prefix' in cases[case_id]:
            # Apply instruction-based prompt (zero-shot style)
            input_text = [cases[case_id]['start_prefix'] + f"{original}" + "\n Output: " + f"{structured}" + "<|end|>" for original, structured in zip(original, structured)]
            
        elif case_id in cases and 'prompt' in cases[case_id]:
            # Apply in-context learning prompt (few-shot style)
            input_text = [cases[case_id]['prompt'] + f"{original}" + "\n Output: " + f"{structured}" + "<|end|>" for original, structured in zip(original, structured)]
            
        elif case_id in cases and 'both' in cases[case_id]:
            # Apply both instruction and in-context learning prompts
            input_text = [cases[case_id]['both'] + f"{original}" + "\n Output: " + f"{structured}" + "<|end|>" for original, structured in zip(original, structured)]
        else:
            input_text = ["<|system|> You are a radiology expert.<|end|> <|user|>"+f"{original}" +"<|end|> \n<|assistant|> Output: " + f"{structured}" + "<|end|>"
            for original, structured in zip(original, structured)]
        
        # Tokenize input text
        inputs = tokenizer(input_text, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")

        # Clone input_ids for labels
        inputs["labels"] = inputs["input_ids"].clone()

        # Mask prompt tokens with -100 in the labels
        for i, original_text in enumerate(original):
            prompt_text = system_message + f"{original_text}<|end|> \n<|assistant|> Output: "
            prompt_length = len(tokenizer(prompt_text, truncation=True, max_length=max_len)["input_ids"])
            inputs["labels"][i, :prompt_length] = -100  # Mask prompt tokens
    return inputs

def reformat_radiology_output(output_list):
    formatted_outputs = []
    for sample in output_list:
        # Remove extra <pad> tokens
        sample = sample.replace("<pad>", "").strip()

        # Split into sections based on known headers or patterns
        sections = ["History:", "Technique:", "Comparison:", "Findings:", "Impression:"]
        organs = ['Lungs and Airways:', 'Musculoskeletal and Chest Wall:','Cardiovascular:','Tubes, Catheters, and Support Devices:','Abdominal:','Pleura:','Other:','Hila and Mediastinum:']
        for section in sections:
            sample = sample.replace(section, f"\n{section}")
        for organ in organs:
            try:
                sample = sample.replace(organ, f"\n{organ}")
            except:
                continue
        # Ensure newlines after colons and before bullet points
        sample = sample.replace("- ", "\n- ")
        # Ensure newlines before numbers 
        try: sample = sample.replace("1.", "\n1.")
        except: continue
        try: sample = sample.replace("2.", "\n2.")
        except: continue
        try: sample = sample.replace("3.", "\n3.")
        except: continue
        try: sample = sample.replace("4.", "\n4.")
        except: continue
        try: sample = sample.replace("5.", "\n5.")
        except: continue
        try: sample = sample.replace("6.", "\n6.")
        except: continue
        try: sample = sample.replace("7.", "\n7.")
        except: continue
        # Remove any leading or trailing whitespace
        sample = sample.strip()

        formatted_outputs.append(sample)

    return formatted_outputs