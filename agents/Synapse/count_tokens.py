import os
import json
import argparse
import pickle
from tqdm import tqdm
import torch
from transformers import AutoTokenizer
from synapse.envs.mind2web.env_utils import load_json

def count_tokens_in_dataset(tokenizer, samples) -> dict:
    """Count tokens in the Mind2Web dataset."""
    token_counts = {
        "total": 0,
        "prompts": 0,
        "tasks": 0,
        "actions": 0,
        "candidates": 0
    }
    
    total_samples = 0
    total_actions = 0
    
    for sample in tqdm(samples, desc="Processing samples"):
        total_samples += 1
        
        # Count task tokens
        task = sample.get('confirmed_task', '')
        task_tokens = len(tokenizer.encode(task))
        token_counts["tasks"] += task_tokens
        
        # Count action tokens
        for action in sample.get('actions', []):
            total_actions += 1
            
            # Count tokens in positive candidates
            for candidate in action.get('pos_candidates', []):
                candidate_text = f"{candidate.get('text', '')} {candidate.get('tag_name', '')} {candidate.get('type', '')}"
                candidate_tokens = len(tokenizer.encode(candidate_text))
                token_counts["candidates"] += candidate_tokens
            
            # Count tokens in negative candidates
            for candidate in action.get('neg_candidates', []):
                candidate_text = f"{candidate.get('text', '')} {candidate.get('tag_name', '')} {candidate.get('type', '')}"
                candidate_tokens = len(tokenizer.encode(candidate_text))
                token_counts["candidates"] += candidate_tokens
            
            # Count action representation tokens
            if 'action_reprs' in sample:
                action_text = " ".join(sample['action_reprs'])
                action_tokens = len(tokenizer.encode(action_text))
                token_counts["actions"] += action_tokens
        
        # Generate and count prompt tokens (similar to eval_sample_llama)
        prompt = f"You are assisting with web automation. Complete the task: {task}\n"
        prompt_tokens = len(tokenizer.encode(prompt))
        token_counts["prompts"] += prompt_tokens
    
    # Calculate total
    token_counts["total"] = sum(v for k, v in token_counts.items() if k != "total")
    
    # Print summary
    print("\nToken Count Summary:")
    print(f"{'='*50}")
    print(f"Total samples processed: {total_samples:,}")
    print(f"Total actions: {total_actions:,}")
    print(f"Total tokens: {token_counts['total']:,}")
    print(f"Prompt tokens: {token_counts['prompts']:,}")
    print(f"Task tokens: {token_counts['tasks']:,}")
    print(f"Action tokens: {token_counts['actions']:,}")
    print(f"Candidate tokens: {token_counts['candidates']:,}")
    print(f"Average tokens per sample: {token_counts['total']/total_samples:,.1f}")
    print(f"Average tokens per action: {token_counts['total']/total_actions:,.1f}")
    
    return token_counts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument(
        "--benchmark", type=str, 
        choices=["test_task", "test_website", "test_domain"],
        required=True
    )
    parser.add_argument(
        "--model_name", 
        type=str,
        default="codellama/CodeLlama-7b-Instruct-hf"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None
    )
    
    args = parser.parse_args()
    
    print(f"Loading tokenizer from {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    
    print(f"\nLoading {args.benchmark} data")
    samples = load_json(args.data_dir, args.benchmark)
    
    # Add prediction scores and ranks to candidates
    with open(os.path.join(args.data_dir, "scores_all_data.pkl"), "rb") as f:
        candidate_results = pickle.load(f)
    
    counts = count_tokens_in_dataset(tokenizer, samples)
    
    # Save results
    output_dir = "token_counts"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"token_counts_{args.benchmark}.json")
    
    with open(output_file, 'w') as f:
        json.dump(counts, f, indent=2)
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()