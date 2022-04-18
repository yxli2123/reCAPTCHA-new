#!/usr/bin/env bash

python main.py \
--task recognition_bg \
--mode train \
--run_name bg_bs32 \
--data_dir data/captcha_click \
--epochs 50 \
--batch_size 32 \
--train_iter 0 \
--valid_interval 5000 \
--save_interval 5000 \
--print_interval 1000

