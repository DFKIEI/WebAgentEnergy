import pickle
import logging
import argparse
import os
import sys
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from synapse.envs.mind2web.env_utils import load_json
from synapse.agents.mind2web import eval_sample_llama

logger = logging.getLogger("synapse")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)

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
        self.logger.info(log_str)
        self.logger.output(log_str, verbose_level=1)
    
    # Apply the patch
    CarbonTrackerThread._log_components_info = fixed_log_components_info
    logger.info("Successfully patched carbontracker device handling")
    
except (ImportError, AttributeError) as e:
    logger.warning(f"Failed to set up carbontracker: {e}")
##########################################################################################

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument(
        "--benchmark", type=str, choices=["test_task", "test_website", "test_domain"]
    )
    parser.add_argument("--previous_top_k_elements", type=int, default=3)
    parser.add_argument("--top_k_elements", type=int, default=5)
    parser.add_argument("--retrieve_top_k", type=int, default=3)
    parser.add_argument("--base_model", type=str)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--lora_dir", type=str, default=None)
    parser.add_argument("--no_memory", action="store_true", default=False)
    parser.add_argument("--no_trajectory", action="store_true", default=False)
    parser.add_argument("--multi_choice", action="store_true", default=False)

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()
    current_path = os.getcwd()
    args.memory_path = os.path.join(current_path, "synapse/memory/mind2web")
    args.log_dir = os.path.join(current_path, "results/mind2web")

    # Initialize the carbon tracker
    tracker = CarbonTracker(epochs=1)
    
    logger.info("Starting carbon tracking")
    tracker.epoch_start()

    print("Load data")
    # Evaluate test set
    assert args.benchmark in ["test_task", "test_website", "test_domain"]
    samples = load_json(args.data_dir, args.benchmark)

    # add prediction scores and ranks to candidates
    with open(os.path.join(args.data_dir, "scores_all_data.pkl"), "rb") as f:
        candidate_results = pickle.load(f)
    candidate_scores = candidate_results["scores"]
    candidate_ranks = candidate_results["ranks"]
    for sample in samples:
        for s in sample["actions"]:
            sample_id = f"{sample['annotation_id']}_{s['action_uid']}"
            for candidates in [s["pos_candidates"], s["neg_candidates"]]:
                for candidate in candidates:
                    candidate_id = candidate["backend_node_id"]
                    candidate["score"] = candidate_scores[sample_id][candidate_id]
                    candidate["rank"] = candidate_ranks[sample_id][candidate_id]

    # load model
    print("Loading model")
    # Before loading the model, add:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    # And modify the model loading to add:
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=args.cache_dir,
        use_flash_attention_2=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, cache_dir=args.cache_dir)

    if args.lora_dir is not None:
        model = PeftModel.from_pretrained(
            model,
            args.lora_dir,
            torch_dtype=torch.bfloat16,
        )

    model.half()

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    print("Starting eval")
    i = 0
    with torch.no_grad():
        for sample in tqdm(samples):
            eval_sample_llama(i, args, sample, model, tokenizer)
            i += 1

    # End tracking even if an exception occurs
    logger.info("Ending carbon tracking")
    tracker.epoch_end()
    tracker.stop()


if __name__ == "__main__":
    main()
