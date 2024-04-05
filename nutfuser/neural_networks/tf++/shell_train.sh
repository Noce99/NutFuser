#!/bin/bash

VENV_TO_SURCE_PATH=$1
TRAIN_SCRIPT_PATH=$2
TRAINING_DATASET=$3
VALIDATION_DATASET=$4
LOG_FOLDER=$5
BATCH_SIZE=$6
TRAIN_FLOW=$7
NUM_OF_GPU=$8

source $VENV_TO_SURCE_PATH
export OMP_NUM_THREADS=20  # Limits pytorch to spawn at most num cpus cores threads
export OPENBLAS_NUM_THREADS=1  # Shuts off numpy multithreading, to avoid threads spawning other threads.
torchrun --nnodes=1 --nproc_per_node=$NUM_OF_GPU --max_restarts=0 --rdzv_id=42353467 --rdzv_backend=c10d $TRAIN_SCRIPT_PATH --id test_my_data --batch_size 10 --setting 02_05_withheld --root_dir $TRAINING_DATASET --val_dir $VALIDATION_DATASET --logdir $LOG_FOLDER --use_controller_input_prediction 0 --use_wp_gru 0 --use_discrete_command 1 --use_tp 1 --continue_epoch 0 --cpu_cores 20 --num_repetitions 3 --lr 7.5e-5 --use_flow $TRAIN_FLOW
# --nproc_per_node=8
# --cpu_cores 20
