

srun -K \
--output=eval_syn_domaina100%j.out \
--error=eval_syn_domaina100%j.err \
--job-name="eval_syn_domaina100" \
--ntasks=1 \
--gpus-per-task=1 \
--cpus-per-task=1 \
-p RTXA6000 \
--mem=30GB \
--container-mounts="/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,$(pwd):$(pwd),$(pwd)/..:$(pwd)/.." \
--container-image=/enroot/nvcr.io_nvidia_pytorch_23.12-py3.sqsh \
--container-workdir="$(pwd)" \
python count_m2w.py