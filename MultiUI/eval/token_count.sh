#!/bin/bash

srun -K \
--output=token_count_multiui_website_%j.out \
--error=token_count_multiui_website_%j.err \
--job-name="token_count_multiui_website_" \
--ntasks=1 \
--gpus-per-task=1 \
--cpus-per-task=2 \
-p RTX3090 \
--mem=32GB \
--container-mounts="/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,$(pwd):$(pwd)" \
--container-image=/enroot/nvcr.io_nvidia_pytorch_23.12-py3.sqsh \
--container-workdir="$(pwd)" \
bash -c "./install.sh && PYTHONPATH=Mind2Web-SeeAct:VisualWebBench \
         python Mind2Web-SeeAct/src/offline_experiments/count_tokens.py \
         --task_types test_website,test_task,test_website \
         --model_name neulab/UIX-Qwen2-Mind2Web"