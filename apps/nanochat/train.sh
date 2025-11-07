#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python train.py \
    --data_path data/train.jsonl \
    --dataset_type text \
    --output_dir checkpoints \
    --vocab_size 32000 \
    --block_size 512 \
    --n_layer 6 \
    --n_head 6 \
    --n_embd 384 \
    --dropout 0.1 \
    --batch_size 4 \
    --accumulation_steps 8 \
    --learning_rate 3e-4 \
    --weight_decay 0.01 \
    --max_steps 10000 \
    --warmup_steps 1000 \
    --eval_interval 500 \
    --save_interval 1000 \
    --log_interval 10 \
    --device cuda:0 \
    --seed 42
