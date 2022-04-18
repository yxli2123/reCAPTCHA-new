#!/usr/bin/env bash

python main.py \
--mode test \
--task segmentation_box \
--ckpt exp_log/recognitino_bg/bs32/45000.pth \
--data_dir data/captcha_click \
--num_char 1
