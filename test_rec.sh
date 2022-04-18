#!/usr/bin/env bash

python main.py \
--mode test \
--task recognition_bg \
--ckpt exp_log/recognition_bg/bg_bs32/70000.pth \
--data_dir data/captcha_click \
