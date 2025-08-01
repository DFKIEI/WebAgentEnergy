

srun -K \
--output=inference_m2wrtx3090%j.out \
--error=inference_m2wrtx3090%j.err \
--job-name="inference_m2wrtx3090" \
--ntasks=1 \
--gpus-per-task=1 \
--cpus-per-task=2 \
-p RTX3090 \
--mem=100GB \
--container-mounts="/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,$(pwd):$(pwd),$(pwd)/..:$(pwd)/.." \
--container-image=/enroot/nvcr.io_nvidia_pytorch_23.12-py3.sqsh \
--container-workdir="$(pwd)" \
--task-prolog="./install.sh" \
python inference_m2w.py