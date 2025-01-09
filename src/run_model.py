import json
import argparse
import torch
import torch.utils.data
import numpy as np
from tqdm import tqdm
from transformers import default_data_collator


from utils import load_model, load_data, parse_args, preprocess_function

def generate_predictions(model, tokenizer, test_loader, device, max_gen_length: int, min_gen_length: int):
    model.eval()
    model.to(device)
    
    predictions = []
    # Initialize tqdm progress bar, setting the total number of batches
    progress_bar = tqdm(test_loader, desc="Generating predictions", unit="batch")

    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        with torch.no_grad():
            generated_ids = model.generate(input_ids, max_new_tokens=max_gen_length, min_new_tokens= min_gen_length, decoder_start_token_id=model.config.decoder_start_token_id, num_beams=5, early_stopping=True)
        decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        predictions.extend(decoded_preds)
    return predictions


def main():

    args = parse_args()

    # Load model and tokenizer
    model, tokenizer = load_model(args.model_path, args.cache_dir, args.max_input_length)
    
    # Create DataLoader for batch processing
    dataset = load_data(args.test_data_path)
    # Preprocess test data
    test_inputs = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, max_len=args.max_input_length),
        batched=True,
        num_proc=1,
        load_from_cache_file=False,
        remove_columns=["inputs", "label"],
        desc="Running tokenizer on dataset")
    
    test_loader = torch.utils.data.DataLoader(test_inputs, batch_size=args.batch_size, collate_fn=default_data_collator)

    # Generate predictions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions = generate_predictions(model, tokenizer, test_loader, device, args.max_gen_length, args.min_gen_length)

    try:
        # Save predictions to file
        with open(args.output_file, "w") as f:
            json.dump(predictions, f)
            print(f"Predictions saved to {args.output_file}")

    except:
        with open("test_predictions.json", "w") as f:
            json.dump(predictions, f)
            print(f"Predictions saved to test_predictions.json")

            
if __name__ == "__main__":
    main()
