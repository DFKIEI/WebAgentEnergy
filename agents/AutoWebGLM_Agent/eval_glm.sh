

srun -K \
--output=eval_glm_weba100%j.out \
--error=eval_glm_weba100%j.err \
--job-name="eval_glm_weba100" \
--ntasks=1 \
--gpus-per-task=1 \
--cpus-per-task=2 \
-p A100-40GB \
--mem=40GB \
--container-mounts="/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,$(pwd):$(pwd),$(pwd)/..:$(pwd)/.." \
--container-image=/enroot/nvcr.io_nvidia_pytorch_23.12-py3.sqsh \
--container-workdir="$(pwd)" \
--task-prolog="./install.sh" \
python eval.py /path/to/AutoWebGLM/results/combined_test_web.json