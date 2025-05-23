import os
import json
from transformers import AutoTokenizer, AutoModel
import torch

#################################### Import carbontracker and apply the fix ---- ONLY NEEDED IF RUNNING ON SLURM CLUSTER
try:
    from carbontracker.tracker import CarbonTracker, CarbonTrackerThread
    
    # Save original method
    original_log_components_info = CarbonTrackerThread._log_components_info
    
    # Create fixed method
    def fixed_log_components_info(self):
        log = ["The following components were found:"]
        for comp in self.components:
            name = comp.name.upper()
            # Fix here: decode byte strings in device names
            devices = [d.decode('utf-8') if isinstance(d, bytes) else d for d in comp.devices()]
            devices = ", ".join(devices)
            log.append(f"{name} with device(s) {devices}.")
        log_str = " ".join(log)
        print(log_str)
    
    # Apply the patch
    CarbonTrackerThread._log_components_info = fixed_log_components_info
    print("Successfully patched carbontracker device handling")
    
except (ImportError, AttributeError) as e:
    print(f"Failed to set up carbontracker: {e}")
##########################################################################################

# Path where your split folders are located
data_root = "/netscratch/banwari/Mind2Web/data"
split_folders = ["test_domain", "test_task", "test_website"]
output_dir = "/netscratch/banwari/llm_energy/AutoWebGLM/results"
os.makedirs(output_dir, exist_ok=True)

# Load model
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True).half().cuda()
model.eval()

# Initialize the carbon tracker
tracker = CarbonTracker(epochs=1)

print("Starting carbon tracking")
tracker.epoch_start()

# Inference loop
for folder in split_folders:
    folder_path = os.path.join(data_root, folder)
    combined_results = []

    print(f"\n--- Processing: {folder_path} ---")
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".json"):
            input_path = os.path.join(folder_path, filename)
            output_path = os.path.join(output_dir, f"results_{filename}")

            with open(input_path, "r") as f:
                tasks = json.load(f)

            local_results = []
            for task in tasks:
                task_id = task.get("annotation_id", "")
                instruction = task.get("confirmed_task", "").strip()
                reference = " -> ".join(task.get("action_reprs", []))

                if not instruction:
                    print(f"⚠ Skipping task {task_id} due to empty instruction")
                    continue

                prompt = f"""You are a web automation agent. Your goal is to generate a precise sequence of actions to complete a web task using the following format:

                            [element] label -> ACTION: value (for TYPE/SELECT), or just ACTION (for CLICK)

                            Use only these operation types:
                            - CLICK
                            - TYPE: some text
                            - SELECT: dropdown value

                            ⚠ Do not explain or narrate — only list the actions.

                            Here is an example:

                            Task: Find the cheapest 3-star hotel with a guest rating of 4 stars for booking two rooms in Pune, India, for 4 adults and one 4-year-old child, near the airport with a check-in date of March 26 and checkout date of March 28.

                            Example Output:
                            [textbox] Destination or property name -> TYPE: PUNE  
                            [strong] Pune -> CLICK  
                            [listitem] 26 -> CLICK  
                            [listitem] 28 -> CLICK  
                            [i]   -> CLICK  
                            [i]   -> CLICK  
                            [i]   -> CLICK  
                            [i]   -> CLICK  
                            [select] <1 -> SELECT: 4  
                            [span] Done -> CLICK  
                            [em] 3 -> CLICK  
                            [span] Search -> CLICK  
                            [li] 4+ -> CLICK  
                            [i]   -> CLICK  
                            [span] Lowest Price (After Tax) -> CLICK

                            ---

                            Now complete the following task in the same format:

                            Task: {instruction}"""
                try:
                    response, _ = model.chat(
                        tokenizer,
                        prompt,
                        history=[],
                        max_new_tokens=512  # Reasonable cap; adjust as needed
                    )
                    response = response[:5000]  # Truncate overly verbose responses
                except Exception as e:
                    print(f"[ERROR] Failed task {task_id}: {e}")
                    response = f"[ERROR] {str(e)}"


                result = {
                    "task_id": task_id,
                    "prediction": response,
                    "reference": reference
                }

                local_results.append(result)
                combined_results.append(result)

            # Save per-file output
            with open(output_path, "w") as f:
                json.dump(local_results, f, indent=2)
            print(f"✔ Saved individual results to {output_path}")

    # Save combined output for this split category
    combined_path = os.path.join(output_dir, f"combined_{folder}.json")
    with open(combined_path, "w") as f:
        json.dump(combined_results, f, indent=2)
    print(f"Combined results saved to {combined_path}")

# End tracking even if an exception occurs
print("Ending carbon tracking")
tracker.epoch_end()
tracker.stop()