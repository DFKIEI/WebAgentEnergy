#!/bin/bash

srun -K \
--output=token_count_synapse_domain_%j.out \
--error=token_count_synapse_domain_%j.err \
--job-name="token_count_synapse_domain" \
--ntasks=1 \
--gpus-per-task=0 \
--cpus-per-task=2 \
-p RTX3090 \
--mem=32GB \
--container-mounts="/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,$(pwd):$(pwd),$(pwd)/..:$(pwd)/.." \
--container-image=/enroot/nvcr.io_nvidia_pytorch_23.12-py3.sqsh \
--container-workdir="$(pwd)" \
bash -c "./install.sh && python count_tokens.py \
    --data_dir /netscratch/banwari/Mind2Web/data \
    --benchmark test_domain \
    --model_name codellama/CodeLlama-7b-Instruct-hf \
    --cache_dir /netscratch/banwari/llm_energy/Synapse/models"