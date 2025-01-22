import json
import torch
import torch.utils.data
import numpy as np
from tqdm import tqdm
import os
from utils import parse_args_run_llm, load_llm_model, load_and_preprocess_dataset_llm, generate_predictions, get_data_collator
import constants
from peft import PeftModel, PeftConfig

def main():
    
    args = parse_args_run_llm()
    # Load model and tokenizer
    if args.case_id == 0: # model is finetuned with lora
        print("Loading model with PEFT...")
        peft_path = args.model
        config = PeftConfig.from_pretrained(peft_path)
        model, tokenizer = load_llm_model(config.base_model_name_or_path, args.cache_dir, task="run")
        model = PeftModel.from_pretrained(model, peft_path)
    else: # model is not finetuned and just prompted
        model, tokenizer = load_llm_model(constants.LLMS[args.model], args.cache_dir, task="run")
    model.eval()
    # Create DataLoader for batch processing
    test_dataset = load_and_preprocess_dataset_llm(args.data_path, tokenizer, split="test_reviewed", max_len=args.max_input_length, case_id=args.case_id)
    # split by 'id' containing 'mimic' or 'chexpert'
    test_dataset_mimic = test_dataset.filter(lambda example: 'mimic' in example['id'])
    test_dataset_chexpert = test_dataset.filter(lambda example: 'chexpert' in example['id'])
    test_dataset_mimic=test_dataset_mimic.remove_columns(['original_report', 'structured_report', 'findings_section', 'impression_section', 'history_section', 'technique_section', 'comparison_section', 'exam_type_section', 'image_paths', 'id', 'labels'])
    test_dataset_chexpert=test_dataset_chexpert.remove_columns(['original_report', 'structured_report', 'findings_section', 'impression_section', 'history_section', 'technique_section', 'comparison_section', 'exam_type_section', 'image_paths', 'id', 'labels'])
    
    test_loader_mimic = torch.utils.data.DataLoader(test_dataset_mimic, batch_size=args.batch_size, collate_fn=get_data_collator(tokenizer))
    test_loader_chexpert = torch.utils.data.DataLoader(test_dataset_chexpert, batch_size=args.batch_size, collate_fn=get_data_collator(tokenizer))
        
    # Generate predictions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions_mimic = generate_predictions(model, tokenizer, test_loader_mimic, device, args.max_gen_length, args.min_gen_length)
    predictions_chexpert = generate_predictions(model, tokenizer, test_loader_chexpert, device, args.max_gen_length, args.min_gen_length)


    # For auto-regressive models, remove the input text from the predictions
    predictions_mimic =[prediction.rsplit("Output:", 1)[-1].strip() for prediction in predictions_mimic]
    predictions_chexpert =[prediction.rsplit("Output:", 1)[-1].strip() for prediction in predictions_chexpert]

    print("Sample prediction:", predictions_mimic[0])

    # Write predictions to file
    # create folder if it doesn't exist
    try:
        with open(args.output_file_mimic, "w") as f:
            json.dump(predictions_mimic, f)
    except FileNotFoundError:
        os.makedirs(os.path.dirname(args.output_file_mimic))
        with open(args.output_file, "w") as f:
            json.dump(predictions_mimic, f)   
    print(f"Predictions on MIMIC saved to {args.output_file_mimic}")

    try:
        with open(args.output_file_chexpert, "w") as f:
            json.dump(predictions_chexpert, f)
    except FileNotFoundError:
        os.makedirs(os.path.dirname(args.output_file_chexpert))
        with open(args.output_file, "w") as f:
            json.dump(predictions_chexpert, f)
    print(f"Predictions on CheXpert saved to {args.output_file_chexpert}")
    
    
if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()