#!/usr/bin/env bash

python main.py \
--mode train \
--ckpt ../exp_log/1ch_captcha/80/56000.pth \
--data_dir ../data/image/captcha_1ch_80 \
--epochs 400 \
--exp_name 1ch_captcha \
--run_name 80 \
--num_char 1 \
--train_iter 56000 
