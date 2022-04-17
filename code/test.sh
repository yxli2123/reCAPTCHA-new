#!/usr/bin/env bash

python main.py \
--mode test \
--ckpt ../exp_log/captcha_segmentation/light_newdata/5000.pth \
--batch_size 1 \
--data_dir ../data/image/
