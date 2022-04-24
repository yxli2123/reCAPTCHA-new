#!/usr/bin/env bash

python main.py \
--task recognition_clean \
--mode train \
--run_name bs64_sim \
--data_dir data/captcha_click \
--epochs 80 \
--char_sim True \
--batch_size 64 \
--valid_interval 5000 \
--save_interval 5000 \
--print_interval 500

