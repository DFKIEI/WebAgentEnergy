import argparse
import json
import os
from tqdm import tqdm
import torch
from transformers import AutoTokenizer

def count_tokens_in_dataset(tokenizer, data_path: str) -> dict:
    """Count tokens in the Synatra Mind2Web dataset."""
    token_counts = {
        "total": 0,
        "prompts": 0,
        "system_messages": 0,
        "user_messages": 0,
        "responses": 0
    }
    
    print(f"\nCounting tokens in: {data_path}")
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    total_samples = len(data)
    
    for entry in tqdm(data, desc="Processing samples"):
        # Count prompt tokens
        prompt_tokens = len(tokenizer.encode(entry["prompt"]))
        token_counts["prompts"] += prompt_tokens
        
        # Count system message tokens
        system_msg = "You are a helpful assistant"
        system_tokens = len(tokenizer.encode(system_msg))
        token_counts["system_messages"] += system_tokens
        
        # Count user message tokens
        user_msg = entry["prompt"]
        user_tokens = len(tokenizer.encode(user_msg))
        token_counts["user_messages"] += user_tokens
        
        # Count response tokens (labels)
        if "label" in entry:
            response_tokens = len(tokenizer.encode(entry["label"]))
            token_counts["responses"] += response_tokens
    
    # Calculate total
    token_counts["total"] = sum(v for k, v in token_counts.items() if k != "total")
    
    # Print summary
    print("\nToken Count Summary:")
    print(f"{'='*50}")
    print(f"Total samples processed: {total_samples:,}")
    print(f"Total tokens: {token_counts['total']:,}")
    print(f"Prompt tokens: {token_counts['prompts']:,}")
    print(f"System message tokens: {token_counts['system_messages']:,}")
    print(f"User message tokens: {token_counts['user_messages']:,}")
    print(f"Response tokens: {token_counts['responses']:,}")
    print(f"Average tokens per sample: {token_counts['total']/total_samples:,.1f}")
    
    return token_counts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_name", type=str, help="JSON file to analyze (e.g., website_test.json)")
    parser.add_argument("model_name", type=str, help="Model name or path for tokenizer")
    
    args = parser.parse_args()
    
    print(f"Loading tokenizer from {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    data_path = os.path.join("../data", args.file_name)
    counts = count_tokens_in_dataset(tokenizer, data_path)
    
    # Save results
    output_dir = "token_counts"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"token_counts_{os.path.splitext(args.file_name)[0]}.json")
    
    with open(output_file, 'w') as f:
        json.dump(counts, f, indent=2)
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()