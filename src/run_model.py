import json
import torch
import torch.utils.data
import numpy as np
from tqdm import tqdm
import os


from utils import load_model, parse_args_run, load_and_preprocess_dataset, get_data_collator, generate_predictions
import constants

def main():

    args = parse_args_run()

    # Load model and tokenizer
    model, tokenizer = load_model(constants.DIR_MODELS_TUNED+args.model+'/', task="run")
    
    # Create DataLoader for batch processing
    test_dataset = load_and_preprocess_dataset(args.data_path, tokenizer, split="test", max_len=args.max_input_length)
    test_dataset=test_dataset.remove_columns(['original_report', 'structured_report', 'findings_section', 'impression_section', 'history_section', 'technique_section', 'comparison_section', 'exam_type_section', 'image_paths', 'id', 'labels'])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=get_data_collator(tokenizer))
    
    # Generate predictions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions = generate_predictions(model, tokenizer, test_loader, device, args.max_gen_length, args.min_gen_length)

    # Write predictions to file
    # create folder if it doesn't exist
    try:
        with open(args.output_file, "w") as f:
            json.dump(predictions, f)
    except FileNotFoundError:
        os.makedirs(os.path.dirname(args.output_file))
        with open(args.output_file, "w") as f:
            json.dump(predictions, f)   
    print(f"Predictions saved to {args.output_file}")
    
if __name__ == "__main__":
    main()
