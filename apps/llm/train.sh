#!/bin/bash

export CUDA_VISIBLE_DEVICES=7
torchrun --nproc_per_node=1 --master-port=21323 train_sft.py
