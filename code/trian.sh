#!/usr/bin/env bash

python main.py \
--mode train \
--data_dir ../data/image/ \
--epochs 400 \
--batch_size 32 \
--num_gpus 1 \
--exp_name captcha_segmentation \
--run_name light \
--train_iter 0

