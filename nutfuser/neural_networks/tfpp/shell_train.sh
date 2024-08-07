#!/bin/bash

VENV_TO_SURCE_PATH=$1
TRAIN_SCRIPT_PATH=$2
TRAINING_DATASET=$3
VALIDATION_DATASET=$4
LOG_FOLDER=$5
BATCH_SIZE=$6
TRAIN_FLOW=$7
NUM_OF_GPU=$8
WEIGHT_PATH=$9
STARTING_EPOCH=${10}
TRAIN_CONTROLL_NETWORK=${11}
USE_ABSTRACT_BEV_SEMANTIC=${12}
USE_SEMANTIC=${13}
USE_DEPTH=${14}
USE_BEV_SEMANTIC=${15}
USE_BOUNDING_BOXES=${16}
PREDICT_SPEED=${17}
PREDICT_ACCELERATION=${18}
EXPERIMENT_NAME=${19}

echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo $VENV_TO_SURCE_PATH
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
source $VENV_TO_SURCE_PATH
export OMP_NUM_THREADS=20  # Limits pytorch to spawn at most num cpus cores threads
export OPENBLAS_NUM_THREADS=1  # Shuts off numpy multithreading, to avoid threads spawning other threads.
torchrun --nnodes=1 --nproc_per_node=$NUM_OF_GPU --max_restarts=0 --rdzv_id=42353467 --rdzv_backend=c10d $TRAIN_SCRIPT_PATH --id $EXPERIMENT_NAME --batch_size $BATCH_SIZE --setting 02_05_withheld --root_dir $TRAINING_DATASET --val_dir $VALIDATION_DATASET --logdir $LOG_FOLDER --use_controller_input_prediction $TRAIN_CONTROLL_NETWORK --use_wp_gru 0 --use_discrete_command 0 --use_tp 1 --continue_epoch $STARTING_EPOCH --cpu_cores 20 --num_repetitions 3 --lr 7.5e-5 --use_flow $TRAIN_FLOW --load_file $WEIGHT_PATH --use_abstract_bev_semantic $USE_ABSTRACT_BEV_SEMANTIC --use_semantic $USE_SEMANTIC --use_depth $USE_DEPTH --use_bev_semantic $USE_BEV_SEMANTIC --detect_boxes $USE_BOUNDING_BOXES --predict_speed $PREDICT_SPEED --predict_acceleration $PREDICT_ACCELERATION
# --nproc_per_node=8
# --cpu_cores 20
