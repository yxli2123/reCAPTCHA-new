#!/usr/bin/env bash

python main.py \
--mode test \
--task recognition_bg \
--ckpt exp_log/recognitino_bg/bg_bs32/45000.pth \
--data_dir data/captcha_click \
--num_char 1
