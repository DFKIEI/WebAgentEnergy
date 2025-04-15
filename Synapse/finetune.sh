

srun -K \
--output=finetune%j.out \
--error=finetune%j.err \
--job-name="finetune" \
--ntasks=1 \
--gpus-per-task=1 \
--cpus-per-task=2 \
-p A100-80GB \
--mem=40GB \
--container-mounts="/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,$(pwd):$(pwd),$(pwd)/..:$(pwd)/.." \
--container-image=/enroot/nvcr.io_nvidia_pytorch_23.12-py3.sqsh \
--container-workdir="$(pwd)" \
--task-prolog="/home/banwari/llm_energy/Synapse/install.sh" \
python finetune_mind2web.py --data_dir /netscratch/banwari/Mind2Web/data --base_model codellama/CodeLlama-7b-Instruct-hf --cache_dir /netscratch/banwari/Mind2Web/Synapse/cache --lora_dir /netscratch/banwari/Mind2Web/Synapse/lora --no_trajectory --top_k_elements 20