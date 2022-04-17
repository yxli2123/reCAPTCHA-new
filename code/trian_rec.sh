#!/usr/bin/env bash

python main_rec.py \
--mode train \
--data_dir /data1/liyx/reCAPTCHA/data/captcha_click/ \
--num_cls 3000 \
--epochs 200 \
--valid_interval 100 \
--save_interval 5000 \
--print_interval 100 \
--lr 1e-3 \
--batch_size 32 \
--num_gpus 1 \
--exp_name captcha_recognition \
--run_name bg_on_clean \
--train_iter 90000 \
--ckpt ../exp_log/captcha_recognition/resnet_bs32/90000.pth
