

srun -K \
--output=builddataset1%j.out \
--error=builddataset1%j.err \
--job-name="builddataset1" \
--ntasks=1 \
--gpus-per-task=1 \
--cpus-per-task=2 \
-p RTXA6000 \
--mem=40GB \
--container-mounts="/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,$(pwd):$(pwd),$(pwd)/..:$(pwd)/.." \
--container-image=/enroot/nvcr.io_nvidia_pytorch_23.12-py3.sqsh \
--container-workdir="$(pwd)" \
--task-prolog="./install.sh" \
python build_dataset.py --data_dir /path/to/Mind2Web/data --no_trajectory --top_k_elements 20 --benchmark train