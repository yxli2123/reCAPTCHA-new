#!/usr/bin/env bash

python main.py \
--task recognition_bg \
--mode train \
--run_name bg_bs32 \
--data_dir data/captcha_click \
--epochs 50 \
--batch_size 32 \
--train_iter 45000 \
--ckpt exp_log/recognition_bg/bg_bs32/45000.pth \
--valid_interval 5000 \
--save_interval 5000 \
--print_interval 1000

