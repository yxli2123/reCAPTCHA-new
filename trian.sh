#!/usr/bin/env bash

python main.py \
--task segmentation_box \
--mode train \
--exp_name segmentation \
--run_name box_bs32 \
--data_dir ../data/captcha_click \
--epochs 100 \
--batch_size 32 \
--train_iter 0 \
--valid_interval 5000 \
--save_interval 5000 \
--print_interval 1000

python main.py \
--task recognition_bg \
--mode train \
--exp_name segmentation \
--run_name box_bs32 \
--data_dir ../data/captcha_click \
--epochs 100 \
--batch_size 32 \
--train_iter 0 \
--valid_interval 5000 \
--save_interval 5000 \
--print_interval 1000
