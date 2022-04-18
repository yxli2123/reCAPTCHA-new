#!/usr/bin/env bash

python main.py \
--mode test \
--task segmentation_box \
--ckpt exp_log/segmentation_box/box_bs2/45000.pth \
--data_dir data/captcha_click

