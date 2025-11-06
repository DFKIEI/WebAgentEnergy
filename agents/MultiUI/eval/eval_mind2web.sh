srun -K \
--output=eval_m2w_mutiui%j.out \
--error=eval_m2w_mutiui%j.err \
--job-name="eval_m2w_mutiui" \
--ntasks=1 \
--gpus-per-task=1 \
--cpus-per-task=1 \
-p RTXA6000 \
--mem=40GB \
--container-mounts="/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,$(pwd):$(pwd)" \
--container-image=/enroot/nvcr.io_nvidia_pytorch_23.12-py3.sqsh \
--container-workdir="$(pwd)" \
bash -c "./install.sh && PYTHONPATH=Mind2Web-SeeAct\
         python Mind2Web-SeeAct/src/offline_experiments/eval_m2w.py \
         --model_name qwen2_mind2web \
         --model_path neulab/UIX-Qwen2-Mind2Web \
         --task_types test_website"
