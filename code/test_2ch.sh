#!/usr/bin/env bash

python main.py \
--mode test \
--ckpt ../exp_log/captcha_2ch/20/3000.pth \
--data_dir ../data/image/captcha_2ch_00 \
--num_char 2

