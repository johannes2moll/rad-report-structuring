import json
import torch
import torch.utils.data
import numpy as np
from tqdm import tqdm
import os


from utils import load_model, parse_args_run, load_and_preprocess_dataset, get_data_collator, generate_predictions
import constants
LOAD_FROM_HF = False

def main():

    args = parse_args_run()

    # Load model and tokenizer
    if LOAD_FROM_HF:
        model, tokenizer = load_model(args.model, task="run")
    else:
        model, tokenizer = load_model(constants.DIR_MODELS_TUNED+args.model+'/', task="run")
    
    # Create DataLoader for batch processing
    test_dataset = load_and_preprocess_dataset(args.data_path, tokenizer, split="test_reviewed", max_len=args.max_input_length)
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

    # Write predictions to file
    # create folder if it doesn't exist
    try:
        with open(args.output_file_mimic, "w") as f:
            json.dump(predictions_mimic, f)
    except FileNotFoundError:
        os.makedirs(os.path.dirname(args.output_file_mimic))
        with open(args.output_file_mimic, "w") as f:
            json.dump(predictions_mimic, f)   
    print(f"Predictions on MIMIC saved to {args.output_file_mimic}")

    try:
        with open(args.output_file_chexpert, "w") as f:
            json.dump(predictions_chexpert, f)
    except FileNotFoundError:
        os.makedirs(os.path.dirname(args.output_file_chexpert))
        with open(args.output_file_chexpert, "w") as f:
            json.dump(predictions_chexpert, f)
    print(f"Predictions on CheXpert saved to {args.output_file_chexpert}")
    
if __name__ == "__main__":
    main()
