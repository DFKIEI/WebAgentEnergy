import json
import logging
import pickle
import sys
import pathlib

import hydra
from dataloader import MultiChoiceDataset, get_data_split
from omegaconf import DictConfig
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    logger.info(f"Counting tokens for model {cfg.model.model_name_or_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load candidate results if available
    candidate_results = None
    if cfg.data.score_file is not None:
        with open(cfg.data.score_file, "rb") as f:
            candidate_results = pickle.load(f)
    
    # Process all test datasets
    total_stats = {}
    
    for test_key, test_split_file in cfg.data.test_split_files.items():
        logger.info(f"Processing {test_key}...")
        
        # Load test data
        test_data = get_data_split(
            cfg.data.data_path,
            test_split_file,
            candidate_results=candidate_results,
        )
        
        # Create dataset
        test_dataset = MultiChoiceDataset(
            test_data,
            tokenizer,
            neg_ratio=cfg.train.neg_ratio,
            num_candidates=cfg.train.num_candidates,
            max_context_len=cfg.train.max_context_len,
            mode=cfg.model.mode,
        )
        
        logger.info(f"Dataset {test_key} has {len(test_dataset)} samples")
        
        # Count tokens for a sample of the dataset
        sample_size = min(500, len(test_dataset))
        total_tokens = 0
        total_input_tokens = 0
        total_label_tokens = 0
        samples_processed = 0
        
        for i in range(0, sample_size, 10):  # Sample every 10th item to get variety
            try:
                sample = test_dataset[i]
                input_tokens = len(sample["input_ids"])
                label_tokens = len(sample["labels"])
                
                total_input_tokens += input_tokens
                total_label_tokens += label_tokens
                total_tokens += input_tokens + label_tokens
                samples_processed += 1
                
                if i % 100 == 0:
                    logger.info(f"Processed {i} samples for {test_key}...")
                    
            except Exception as e:
                logger.warning(f"Error processing sample {i} for {test_key}: {e}")
                continue
        
        if samples_processed > 0:
            avg_input_tokens = total_input_tokens / samples_processed
            avg_label_tokens = total_label_tokens / samples_processed
            avg_total_tokens = total_tokens / samples_processed
            
            # Estimate for full dataset
            estimated_total_input = avg_input_tokens * len(test_dataset)
            estimated_total_labels = avg_label_tokens * len(test_dataset)
            estimated_total = avg_total_tokens * len(test_dataset)
            
            stats = {
                "dataset_size": len(test_dataset),
                "samples_analyzed": samples_processed,
                "avg_input_tokens": avg_input_tokens,
                "avg_label_tokens": avg_label_tokens,
                "avg_total_tokens": avg_total_tokens,
                "estimated_total_input_tokens": estimated_total_input,
                "estimated_total_label_tokens": estimated_total_labels,
                "estimated_total_tokens": estimated_total
            }
            
            total_stats[test_key] = stats
            
            logger.info(f"Stats for {test_key}:")
            logger.info(f"  Dataset size: {len(test_dataset):,}")
            logger.info(f"  Avg input tokens: {avg_input_tokens:.2f}")
            logger.info(f"  Avg label tokens: {avg_label_tokens:.2f}")
            logger.info(f"  Avg total tokens: {avg_total_tokens:.2f}")
            logger.info(f"  Estimated total tokens: {estimated_total:,.0f}")
        else:
            logger.error(f"No samples could be processed for {test_key}")
    
    # Save results
    output_file = "token_count_results.json"
    with open(output_file, "w") as f:
        json.dump(total_stats, f, indent=2)
    
    logger.info(f"Token counting complete. Results saved to {output_file}")
    
    # Print summary
    grand_total = sum(stats["estimated_total_tokens"] for stats in total_stats.values())
    logger.info(f"\nSUMMARY:")
    logger.info(f"Total estimated tokens across all datasets: {grand_total:,.0f}")
    
    return total_stats

if __name__ == "__main__":
    main()