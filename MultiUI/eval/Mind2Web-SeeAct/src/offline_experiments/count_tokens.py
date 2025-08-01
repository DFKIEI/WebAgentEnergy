import os
import json
import argparse
import yaml
from tqdm import tqdm
from transformers import AutoTokenizer
from src.data_utils.prompts import generate_prompt

def count_tokens_in_dataset(tokenizer, source_data_path: str, task_type: str) -> dict:
    """Count tokens in the Mind2Web-SeeAct dataset for specific task type."""
    token_counts = {
        "total": 0,
        "system_prompts": 0,
        "user_prompts": 0,
        "tasks": 0,
        "previous_actions": 0
    }
    
    print(f"\nCounting tokens in: {task_type}")
    cur_source_data_path = os.path.join(source_data_path, task_type)
    
    # Get all valid files
    all_file_lst = os.listdir(cur_source_data_path)
    file_lst = []
    for action_file in all_file_lst:
        if os.path.exists(os.path.join(cur_source_data_path, action_file, "queries.jsonl")):
            file_lst.append(action_file)
    
    total_queries = 0
    for action_file in tqdm(file_lst, desc="Processing files"):
        jsonl_path = os.path.join(cur_source_data_path, action_file, "queries.jsonl")
        
        try:
            with open(jsonl_path, "r", encoding="utf-8") as reader:
                for line in reader:
                    query = json.loads(line)
                    total_queries += 1
                    
                    # Count tokens in task
                    task_text = query.get('confirmed_task', '')
                    task_tokens = len(tokenizer.encode(task_text))
                    token_counts["tasks"] += task_tokens
                    
                    # Count tokens in previous actions
                    prev_actions = query.get('previous_actions', [])
                    prev_actions_text = "; ".join(prev_actions)
                    prev_tokens = len(tokenizer.encode(prev_actions_text))
                    token_counts["previous_actions"] += prev_tokens
                    
                    # Generate and count prompt tokens
                    prompt_list = generate_prompt('bbox_generate', 
                                               task=task_text,
                                               previous=prev_actions)
                    
                    if len(prompt_list) >= 2:
                        system_tokens = len(tokenizer.encode(prompt_list[0]))
                        user_tokens = len(tokenizer.encode(prompt_list[1]))
                        token_counts["system_prompts"] += system_tokens
                        token_counts["user_prompts"] += user_tokens
                    else:
                        system_tokens = len(tokenizer.encode(prompt_list[0]))
                        token_counts["system_prompts"] += system_tokens
                    
        except Exception as e:
            print(f"Error processing {jsonl_path}: {e}")
            continue
    
    # Calculate total tokens
    token_counts["total"] = sum(token_counts.values()) - token_counts["total"]
    
    # Print summary
    print("\nToken Count Summary:")
    print(f"{'='*50}")
    print(f"Split: {task_type}")
    print(f"Total queries processed: {total_queries:,}")
    print(f"Total tokens: {token_counts['total']:,}")
    print(f"System prompt tokens: {token_counts['system_prompts']:,}")
    print(f"User prompt tokens: {token_counts['user_prompts']:,}")
    print(f"Task tokens: {token_counts['tasks']:,}")
    print(f"Previous actions tokens: {token_counts['previous_actions']:,}")
    print(f"Average tokens per query: {token_counts['total']/total_queries:,.1f}")
    
    return token_counts

def main():
    parser = argparse.ArgumentParser(description='Count tokens in Mind2Web-SeeAct dataset')
    parser.add_argument('--task_types', type=str, default='test_website',
                      help='Task types to analyze (comma-separated)')
    parser.add_argument('--model_name', type=str,
                      default="neulab/UIX-Qwen2-Mind2Web",
                      help='Model name for tokenizer')
    parser.add_argument('--data_name', type=str,
                      default='bbox_generate_gt_crop_offline_data_-1choices',
                      help='Data name')
    
    args = parser.parse_args()
    
    # Setup paths
    source_data_path = f"/path/to/MultiUI/screenshot_generation/data/Mind2Web_bbox_eval/{args.data_name}"
    
    print(f"Loading tokenizer from {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    
    # Process each task type
    for task_type in args.task_types.split(','):
        counts = count_tokens_in_dataset(tokenizer, source_data_path, task_type)
        
        # Save results
        output_dir = "token_counts"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"token_counts_{task_type}.json")
        
        with open(output_file, 'w') as f:
            json.dump(counts, f, indent=2)
        print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()