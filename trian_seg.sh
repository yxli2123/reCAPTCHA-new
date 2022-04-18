#!/usr/bin/env bash

python main.py \
--task segmentation_box \
--mode train \
--run_name box_bs16 \
--data_dir data/captcha_click \
--epochs 50 \
--batch_size 2 \
--train_iter 0 \
--valid_interval 5000 \
--save_interval 5000 \
--print_interval 1000

