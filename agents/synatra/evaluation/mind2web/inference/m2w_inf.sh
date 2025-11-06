

srun -K \
--output=inf_syn_webh200%j.out \
--error=inf_syn_webh200%j.err \
--job-name="inf_syn_webh200" \
--ntasks=1 \
--gpus-per-task=1 \
--cpus-per-task=2 \
-p H200 \
--mem=100GB \
--container-mounts="/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,$(pwd):$(pwd),$(pwd)/..:$(pwd)/.." \
--container-image=/enroot/nvcr.io_nvidia_pytorch_23.12-py3.sqsh \
--container-workdir="$(pwd)" \
--task-prolog="./install.sh" \
python m2w_code_offline.py website_test.json /path/to/Synatra-Models/Synatra-CodeLlama