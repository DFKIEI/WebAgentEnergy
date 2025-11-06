

srun -K \
--output=evaluate_mind2web_test_domain_v100%j.out \
--error=evaluate_mind2web_test_domain_v100%j.err \
--job-name="evaluate_mind2web_test_domain_v100" \
--ntasks=1 \
--gpus-per-task=1 \
--cpus-per-task=4 \
-p V100-32GB \
--mem=100GB \
--container-mounts="/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,$(pwd):$(pwd),$(pwd)/..:$(pwd)/.." \
--container-image=/enroot/nvcr.io_nvidia_pytorch_23.12-py3.sqsh \
--container-workdir="$(pwd)" \
--task-prolog="./install.sh" \
--time=3-00:00 \
python evaluate_mind2web.py --data_dir /path/to/Mind2Web/data --no_memory --no_trajectory --benchmark test_domain --base_model codellama/CodeLlama-7b-Instruct-hf --cache_dir /path/to/Mind2Web/Synapse/cache --lora_dir /path/to/Mind2Web/Synapse/lora-naive-2025-04-13-14-16 --top_k_elements 20
