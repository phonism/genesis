#!/bin/bash

torchrun --nproc_per_node=8 --master-port=31323 train.py
