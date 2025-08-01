srun -K \
--output=count_tokens_website_xl_%j.out \
--error=count_tokens_website_xl_%j.err \
--job-name="count_tokens_website_xl" \
--ntasks=1 \
--gpus-per-task=0 \
--cpus-per-task=1 \
--mem=35GB \
--container-mounts="/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,$(pwd):$(pwd),$(pwd)/..:$(pwd)/.." \
--container-image=/enroot/nvcr.io_nvidia_pytorch_24.02-py3.sqsh \
--container-workdir="$(pwd)" \
--time=0-02:00:00 \
bash -c "./install.sh && python src/action_prediction/count_tokens.py model=flan-t5-xl"