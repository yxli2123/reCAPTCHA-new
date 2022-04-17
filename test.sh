#!/usr/bin/env bash

python main.py \
--mode test \
--ckpt ../exp_log/captcha_1ch_00/resnet50_bs64/20000.pth \
--data_dir ../data/image/captcha_1ch_00 \
--num_char 1

python main.py \
--mode test \
--ckpt ../exp_log/captcha_1ch_00/resnet50_bs64/20000.pth \
--data_dir ../data/image/captcha_1ch_10 \
--num_char 1

python main.py \
--mode test \
--ckpt ../exp_log/captcha_1ch_00/resnet50_bs64/20000.pth \
--data_dir ../data/image/captcha_1ch_20 \
--num_char 1

python main.py \
--mode test \
--ckpt ../exp_log/captcha_1ch_00/resnet50_bs64/20000.pth \
--data_dir ../data/image/captcha_1ch_30 \
--num_char 1

python main.py \
--mode test \
--ckpt ../exp_log/captcha_1ch_00/resnet50_bs64/20000.pth \
--data_dir ../data/image/captcha_1ch_40 \
--num_char 1

python main.py \
--mode test \
--ckpt ../exp_log/captcha_1ch_00/resnet50_bs64/20000.pth \
--data_dir ../data/image/captcha_1ch_50 \
--num_char 1 

python main.py \
--mode test \
--ckpt ../exp_log/captcha_1ch_00/resnet50_bs64/20000.pth \
--data_dir ../data/image/captcha_1ch_60 \
--num_char 1

python main.py \
--mode test \
--ckpt ../exp_log/captcha_1ch_00/resnet50_bs64/20000.pth \
--data_dir ../data/image/captcha_1ch_70 \
--num_char 1

python main.py \
--mode test \
--ckpt ../exp_log/captcha_1ch_00/resnet50_bs64/20000.pth \
--data_dir ../data/image/captcha_1ch_80 \
--num_char 1
