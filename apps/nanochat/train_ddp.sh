#!/bin/bash
# DDP training script for nanochat
#
# Usage:
#   Single GPU:      bash train_ddp.sh
#   2 GPUs:          bash train_ddp.sh 2
#   4 GPUs:          bash train_ddp.sh 4

NUM_GPUS=${1:-1}

if [ $NUM_GPUS -eq 1 ]; then
    echo "Training on single GPU..."
    python train.py \
        --batch-size 4 \
        --block-size 512 \
        --learning-rate 3e-4 \
        --accumulation-steps 8 \
        --eval-interval 500 \
        --save-interval 100
else
    echo "Training on $NUM_GPUS GPUs with DDP..."
    torchrun --nproc_per_node=$NUM_GPUS train.py \
        --ddp \
        --batch-size 1 \
        --block-size 2048 \
        --learning-rate 3e-4 \
        --accumulation-steps 8 \
        --eval-interval 500 \
        --save-interval 100
fi
