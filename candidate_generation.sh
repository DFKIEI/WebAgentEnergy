srun -K \
--output=canditate_generation_test_task%j.out \
--error=canditate_generation_test_task%j.err \
--job-name="canditate_generation_test_task" \
--ntasks=1 \
--gpus-per-task=1 \
--cpus-per-gpu=10 \
-p RTXA6000 \
--mem=40GB \
--container-mounts="/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,$(pwd):$(pwd),$(pwd)/..:$(pwd)/.." \
--container-image=/enroot/nvcr.io_nvidia_pytorch_23.12-py3.sqsh \
--container-workdir="$(pwd)" \
--task-prolog="/home/banwari/llm_energy/Mind2Web/install.sh" \
--time=3-00:00:00 \
python src/candidate_generation/evaluate.py \
    --model_path osunlp/MindAct_CandidateGeneration_deberta-v3-base \
    --data_path ./ \
    --split_file "./data/test_task/*.json" \
    --output_dir ./results/