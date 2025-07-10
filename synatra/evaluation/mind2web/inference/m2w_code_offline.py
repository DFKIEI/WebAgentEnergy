import argparse
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

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


# Load model and tokenizer globally to avoid reloading in every subprocess
MODEL = None
TOKENIZER = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_and_tokenizer(model_name):
    global MODEL, TOKENIZER
    TOKENIZER = AutoTokenizer.from_pretrained(model_name)
    MODEL = AutoModelForCausalLM.from_pretrained(model_name)
    MODEL.to(DEVICE)
    MODEL.eval()


def generate_locally(messages, temperature, max_tokens, top_p):
    prompt = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in messages])
    inputs = TOKENIZER(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = MODEL.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=TOKENIZER.eos_token_id
        )

    response = TOKENIZER.decode(outputs[0], skip_special_tokens=True)
    return response


def get_result_data(file_name, model_name):
    with open(f"../data/{file_name}", "r") as f:
        data = json.load(f)
        result_data = []
        for entry in data:
            result_data.append(
                {
                    "prompt": entry["prompt"],
                    "label": entry["response"].split("\n")[-1],
                }
            )
        return result_data


def query_model(entry, file_name, model_name):
    err_cnt_dict = defaultdict(int)
    start_time = time.time()
    current = entry["prompt"]

    messages = [{"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": current}]

    tot = 0
    for m in messages:
        tot += len(TOKENIZER.encode(m["role"]))
        tot += len(TOKENIZER.encode(m["content"]))
        tot += 10
    if tot > 3800:
        print(tot)
        err_cnt_dict["prompt>3800"] += 1
        return

    response = generate_locally(
        messages=messages,
        temperature=0.1,
        max_tokens=4096 - tot,
        top_p=0.1,
    )

    action = response.split("\n")[-2].strip() if "\n" in response else response.strip()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Run time: {execution_time} seconds.")
    print(err_cnt_dict)
    model_id = os.path.basename(model_name.rstrip("/"))
    slurm_job_name = os.environ.get("SLURM_JOB_NAME", "defaultjob")

    output_file = file_name.replace(
        ".json", f"_{model_id}_{file_name}_{slurm_job_name}_prediction_codeprompt.jsonl"
    )

    dir_path = os.path.dirname(output_file)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    with open(output_file, "a") as f_of:
        json_line = json.dumps({"predict": action, "label": entry["label"]})
        f_of.write(f"{json_line}\n")


def main():

    # Initialize the carbon tracker
    tracker = CarbonTracker(epochs=1)

    print("Starting carbon tracking")
    tracker.epoch_start()

    parser = argparse.ArgumentParser()
    parser.add_argument("file_name", type=str)
    parser.add_argument("model_name", type=str)  # Should be path or HF model like 'oottyy/Synatra-CodeLlama'
    args = parser.parse_args()

    print("Load model")
    load_model_and_tokenizer(args.model_name)
    print("Load model done!")

    print("Get result data")
    result_data = get_result_data(args.file_name, args.model_name)
    print("Get result data done!")

    print("Query model")
    #with ProcessPoolExecutor() as executor:
    #    results = list(
    #        executor.map(
    #            query_model,
    #            result_data,
    #            repeat(args.file_name),
    #            repeat(args.model_name),
    #        )
    #    )
    for entry in result_data:
        query_model(entry, args.file_name, args.model_name)
    print("Query model done!")

    # End tracking even if an exception occurs
    print("Ending carbon tracking")
    tracker.epoch_end()
    tracker.stop()


if __name__ == "__main__":
    main()
