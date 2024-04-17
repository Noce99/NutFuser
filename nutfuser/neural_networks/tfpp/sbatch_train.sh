#!/bin/bash
#SBATCH --job-name=first_training_nut_dataset
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --output=/leonardo_work/IscrC_SSNeRF/job_log/output.out
#SBATCH --error=/leonardo_work/IscrC_SSNeRF/job_log/error.err
#SBATCH --partition=boost_usr_prod

# print info about current job
scontrol show job $SLURM_JOB_ID

echo "STARTED PRINTING NVIDIA INFO"
nvidia-smi --loop=1 >> /leonardo_work/IscrC_SSNeRF/job_log/nvidia_log.log &
echo q | htop | aha --black --line-fix > /leonardo_work/IscrC_SSNeRF/job_log/htop.html
ACTAUL_PATH=$(pwd)
echo "ACTUAL POSTION = $ACTAUL_PATH"
source /leonardo_work/IscrC_SSNeRF/NutFuser/bin/activate

cd /leonardo_work/IscrC_SSNeRF/NutFuser/src/neural_networks/tfpp

export OMP_NUM_THREADS=20  # Limits pytorch to spawn at most num cpus cores threads
export OPENBLAS_NUM_THREADS=1  # Shuts off numpy multithreading, to avoid threads spawning other threads.
TRAINING_DATASET="/leonardo_work/IscrC_SSNeRF/nut_dataset_200GB"
VALIDATION_DATASET="/leonardo_work/IscrC_SSNeRF/nut_dataset_24GB"
LOG_FOLDER="/leonardo_work/IscrC_SSNeRF/trains_log"
torchrun --nnodes=1 --nproc_per_node=4 --max_restarts=0 --rdzv_id=42353467 --rdzv_backend=c10d train.py --id test_my_data --batch_size 50 --setting 02_05_withheld --root_dir $TRAINING_DATASET --val_dir $VALIDATION_DATASET --logdir $LOG_FOLDER --use_controller_input_prediction 0 --use_wp_gru 0 --use_discrete_command 1 --use_tp 1 --continue_epoch 0 --cpu_cores 32 --num_repetitions 3 --lr 7.5e-5
# --nproc_per_node=8
# --cpu_cores 20
