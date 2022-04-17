#!/usr/bin/env bash

python main.py \
--mode train \
--data_dir /data/liyx/reCAPTCHA-main/data/image/captcha_click/ \
--epochs 400 \
--batch_size 8 \
--num_gpus 1 \
--exp_name captcha_segmentation \
--run_name light_box \
--train_iter 0
