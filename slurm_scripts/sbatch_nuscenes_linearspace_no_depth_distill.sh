#!/usr/bin/bash
#SBATCH --nodes=1
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --gres=gpu:a100:8
#SBATCH --cpus-per-task=10
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --dependency=singleton
#SBATCH --mail-user=YOUR_EMAIL
#SBATCH --mail-type=BEGIN,END,FAIL

run_name=$1
largest_checkpoint=$2
MODEL_CONFIG_FILE=$3
wandb_id=$4

echo $run_name
echo $largest_checkpoint
echo $MODEL_CONFIG_FILE
echo $wandb_id

module avail apptainer  # Check if Apptainer is available
module load apptainer   # Load the Apptainer module

export MASTER_ADDR=$(scontrol show hostname "${SLURM_JOB_NODELIST}" | head -n1)
export MASTER_PORT="$((${SLURM_JOB_ID} % 10000 + 10000))"
export WORLD_SIZE="${SLURM_NTASKS}"

if [ "$largest_checkpoint" == "" ]; then
    srun --gres=gpu:8 --ntasks=8 --ntasks-per-node=8 --nice=100 apptainer exec --nv /home/letianw/projects/rrg-swasland/letianw/distillnerf5.sif \
    bash -c \
    "cd /home/letianw/projects/rrg-swasland/letianw/distillnerf/DistillNeRF; \
    wandb online; \
    PYTHONPATH="./":$PYTHONPATH \
    python -u ./tools/train.py ./projects/DistillNeRF/configs/model_wrapper/${MODEL_CONFIG_FILE}.py --seed 0 \
    --work-dir=../work_dir_$run_name --launcher="slurm" \
    --cfg-options log_config.hooks[1].init_kwargs.name=$run_name log_config.hooks[1].init_kwargs.id=$wandb_id
    "

else
    srun --gres=gpu:8 --ntasks=8 --ntasks-per-node=8 --nice=100 apptainer exec --nv /home/letianw/projects/rrg-swasland/letianw/distillnerf5.sif \
    bash -c \
    "cd /home/letianw/projects/rrg-swasland/letianw/distillnerf/DistillNeRF; \
    wandb online; \
    PYTHONPATH="./":$PYTHONPATH \
    python -u ./tools/train.py ./projects/DistillNeRF/configs/model_wrapper/${MODEL_CONFIG_FILE}.py --seed 0 \
    --work-dir=../work_dir_$run_name --launcher="slurm" \
    --resume-from ${largest_checkpoint} --cfg-options log_config.hooks[1].init_kwargs.name=$run_name \
    log_config.hooks[1].init_kwargs.id=$wandb_id log_config.hooks[1].init_kwargs.resume="must"
    "
fi
