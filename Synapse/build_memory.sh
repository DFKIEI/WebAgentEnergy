

srun -K \
--output=buildmemory%j.out \
--error=buildmemory%j.err \
--job-name="buildmemory" \
--ntasks=1 \
--gpus-per-task=1 \
--cpus-per-task=2 \
-p RTXA6000 \
--mem=40GB \
--container-mounts="/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,$(pwd):$(pwd),$(pwd)/..:$(pwd)/.." \
--container-image=/enroot/nvcr.io_nvidia_pytorch_23.12-py3.sqsh \
--container-workdir="$(pwd)" \
--task-prolog="/home/banwari/llm_energy/Synapse/install.sh" \
python build_memory.py --env mind2web --mind2web_data_dir /netscratch/banwari/Mind2Web/data --mind2web_top_k_elements 3