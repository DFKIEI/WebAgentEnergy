srun -K \
--output=metric_m2w%j.out \
--error=metric_m2w%j.err \
--job-name="metric_m2w" \
--ntasks=1 \
--gpus-per-task=0 \
--cpus-per-task=1 \
--mem=40GB \
--container-mounts="/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,$(pwd):$(pwd)" \
--container-image=/enroot/nvcr.io_nvidia_pytorch_23.12-py3.sqsh \
--container-workdir="$(pwd)" \
bash -c "./install.sh && PYTHONPATH=Mind2Web-SeeAct \
         python Mind2Web-SeeAct/src/offline_experiments/action_generation/metric.py"
