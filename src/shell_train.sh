#!/bin/bash

export OMP_NUM_THREADS=12 # Limits pytorch to spawn at most num cpus cores threads
export OPENBLAS_NUM_THREADS=1 # Shuts off numpy multithreading, to avoid threads spawning other threads.
torchrun --nnodes=1 --nproc_per_node=2 train.py
torchrun --nnodes=1 --nproc_per_node=1 train.py
