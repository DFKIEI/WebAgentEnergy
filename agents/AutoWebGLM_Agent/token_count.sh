#!/bin/bash

srun -K \
--output=token_count_glm_domain_%j.out \
--error=token_count_glm_domain_%j.err \
--job-name="token_count_glm_domain" \
--ntasks=1 \
--gpus-per-task=0 \
--cpus-per-task=2 \
--mem=32GB \
--container-mounts="/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,$(pwd):$(pwd),$(pwd)/..:$(pwd)/.." \
--container-image=/enroot/nvcr.io_nvidia_pytorch_23.12-py3.sqsh \
--container-workdir="$(pwd)" \
bash -c "chmod +x install.sh && ./install.sh && python count_tokens.py --task_type test_domain"