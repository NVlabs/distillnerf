
MODEL_CONFIG_FILE="model_wrapper_linearspace_no_depth_distilll"
STBATCH_FILE="sbatch_nuscenes_linearspace.sh"

date=$(date +'%Y-%m-%d_%H-%M-%S') 
run_name=${MODEL_CONFIG_FILE}_${date}
wandb_id=${MODEL_CONFIG_FILE}_${date}
checkpoint_dir=../work_dir_${MODEL_CONFIG_FILE}_${date}/latest.pth

echo ${checkpoint_dir}
# Run the first job, without checkpoint
sbatch --account YOUR_ACCOUNT DISTILLNERF_PATH/slurm_scripts/sequence_sbatch_v2.sh $run_name "" $MODEL_CONFIG_FILE $wandb_id
sleep 600
# Run the second job, with checkpoint
sbatch --account YOUR_ACCOUNT DISTILLNERF_PATH/slurm_scripts/sequence_sbatch_v2.sh $run_name $checkpoint_dir $MODEL_CONFIG_FILE $wandb_id

# Loop 15 times
for (( i=1; i<=100; i++ ))
do
    sleep 14100
    echo "Loop iteration: $i"
    sbatch --account YOUR_ACCOUNT DISTILLNERF_PATH/slurm_scripts/sequence_sbatch_v2.sh $run_name $checkpoint_dir $MODEL_CONFIG_FILE $wandb_id
done

