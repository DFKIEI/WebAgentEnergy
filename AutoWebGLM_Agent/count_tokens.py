import os
import json
import argparse
from transformers import AutoTokenizer
from tqdm import tqdm

def count_tokens_in_dataset(tokenizer, data_root: str, task_type: str) -> dict:
    """Count tokens in the Mind2Web dataset for a specific split."""
    token_counts = {
        "total": 0,
        "prompts": 0,
        "tasks": 0,
        "references": 0,
        "num_tasks": 0,
        "num_files": 0
    }
    
    # Map task_type to actual file paths
    path_mapping = {
        "test_domain": "domain/test.json",
        "test_task": "task/test.json", 
        "test_website": "website/test.json"
    }
    
    if task_type not in path_mapping:
        # If it doesn't match predefined mapping, use as direct path
        file_path = os.path.join(data_root, f"{task_type}.json")
    else:
        file_path = os.path.join(data_root, path_mapping[task_type])
    
    print(f"\nCounting tokens in: {file_path}")
    
    # Check if it's a single file or directory
    if os.path.isfile(file_path):
        files_to_process = [file_path]
    elif os.path.isdir(os.path.dirname(file_path)):
        # Look for JSON files in the directory
        folder_path = os.path.dirname(file_path)
        all_files = [
            os.path.join(folder_path, f) 
            for f in os.listdir(folder_path) 
            if f.endswith('.json')
        ]
        files_to_process = all_files
    else:
        raise FileNotFoundError(f"Path not found: {file_path}")
    
    for file_path in tqdm(files_to_process, desc="Processing files"):
        token_counts["num_files"] += 1
        
        with open(file_path, "r") as f:
            tasks = json.load(f)

        for task in tasks:
            token_counts["num_tasks"] += 1
            
            # Count tokens in task instruction
            instruction = task.get("confirmed_task", "").strip()
            task_tokens = len(tokenizer.encode(instruction))
            token_counts["tasks"] += task_tokens

            # Count tokens in reference actions
            reference = " -> ".join(task.get("action_reprs", []))
            ref_tokens = len(tokenizer.encode(reference))
            token_counts["references"] += ref_tokens

            # Create and count tokens in full prompt
            prompt = f"""You are a web automation agent. Your goal is to generate a precise sequence of actions to complete a web task using the following format:
[element] label -> ACTION: value
Task: {instruction}"""
            prompt_tokens = len(tokenizer.encode(prompt))
            token_counts["prompts"] += prompt_tokens

    # Calculate total
    token_counts["total"] = token_counts["prompts"] + token_counts["references"]

    # Print summary
    print("\nToken Count Summary:")
    print(f"{'='*50}")
    print(f"Split: {task_type}")
    print(f"Files processed: {token_counts['num_files']}")
    print(f"Tasks processed: {token_counts['num_tasks']}")
    print(f"Total tokens: {token_counts['total']:,}")
    print(f"Prompt tokens: {token_counts['prompts']:,}")
    print(f"Task instruction tokens: {token_counts['tasks']:,}")
    print(f"Reference tokens: {token_counts['references']:,}")
    
    if token_counts['num_tasks'] > 0:
        print(f"Average tokens per task: {token_counts['total']/token_counts['num_tasks']:,.1f}")
        print(f"Average prompt tokens per task: {token_counts['prompts']/token_counts['num_tasks']:,.1f}")
        print(f"Average reference tokens per task: {token_counts['references']/token_counts['num_tasks']:,.1f}")
    
    return token_counts

def main():
    parser = argparse.ArgumentParser(description='Count tokens in Mind2Web dataset')
    parser.add_argument('--task_type', type=str, required=True,
                      help='Which split to analyze (test_domain, test_task, test_website, or direct path)')
    parser.add_argument('--data_root', type=str, 
                      default="/path/to/your/mind2web",
                      help='Path to Mind2Web data directory')
    parser.add_argument('--model_name', type=str,
                      default="THUDM/chatglm3-6b",
                      help='Model name for tokenizer')
    
    args = parser.parse_args()
    
    print(f"Loading tokenizer from {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    
    try:
        counts = count_tokens_in_dataset(tokenizer, args.data_root, args.task_type)
        
        # Save results
        output_dir = "token_counts"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"token_counts_{args.task_type.replace('/', '_')}.json")
        
        with open(output_file, 'w') as f:
            json.dump(counts, f, indent=2)
        print(f"\nResults saved to {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please check your file paths and try again.")

if __name__ == "__main__":
    main()