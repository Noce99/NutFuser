#!/bin/bash

export OMP_NUM_THREADS=20  # Limits pytorch to spawn at most num cpus cores threads
export OPENBLAS_NUM_THREADS=1  # Shuts off numpy multithreading, to avoid threads spawning other threads.
TRAINING_DATASET="/leonardo_work/IscrC_SSNeRF/nut_dataset_200GB" # "/home/enrico/Downloads/nut_dataset_old"
VALIDATION_DATASET="/leonardo_work/IscrC_SSNeRF/nut_dataset_24GB" # "/home/enrico/Downloads/nut_dataset_old_val"
LOG_FOLDER="/leonardo_work/IscrC_SSNeRF/trains_log" # "/home/enrico/Projects/Carla/NutFuser/src/neural_networks/train_logs"
torchrun --nnodes=1 --nproc_per_node=4 --max_restarts=0 --rdzv_id=42353467 --rdzv_backend=c10d train.py --id test_my_data --batch_size 10 --setting 02_05_withheld --root_dir $TRAINING_DATASET --val_dir $VALIDATION_DATASET --logdir $LOG_FOLDER --use_controller_input_prediction 0 --use_wp_gru 0 --use_discrete_command 1 --use_tp 1 --continue_epoch 0 --cpu_cores 20 --num_repetitions 3 --lr 7.5e-5
# --nproc_per_node=8
# --cpu_cores 20
