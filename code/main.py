import argparse
import json
import os
import random

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision


#from model import ResNet
from unet import UNet
from utils import *
from dataloader import LineCAPTCHA


def main():
    parser = argparse.ArgumentParser(description='reCAPTCHA')

    # Experiment Parameters
    parser.add_argument('--mode',           type=str,   default='train', choices=['train', 'test'])
    parser.add_argument('--exp_dir',        type=str,   default='../exp_log')
    parser.add_argument('--exp_name',       type=str,   default='captcha_2ch')
    parser.add_argument('--run_name',       type=str,   default='80')

    # Dataset Parameters
    parser.add_argument('--data_dir',       type=str,   default='../data/image/captcha_2ch_80/')
    parser.add_argument('--info_file',      type=str,   default='')
    parser.add_argument('--num_char',       type=int,   default=2)
    parser.add_argument('--topk',           type=int,   default=5)

    # Training Parameters
    parser.add_argument('--train_iter',     type=int,   default=0)
    parser.add_argument('--epochs',         type=int,   default=200)
    parser.add_argument('--batch_size',     type=int,   default=64)
    parser.add_argument('--lr',             type=float, default=1e-4)
    parser.add_argument('--threads',        type=int,   default=4)
    parser.add_argument('--valid_interval', type=int,   default=500)
    parser.add_argument('--print_interval', type=int,   default=100)
    parser.add_argument('--save_interval',  type=int,   default=1000)

    # Model Parameters
    parser.add_argument('--VM_name',        type=str,   default='ResNet101')
    parser.add_argument('--num_cls',        type=int,   default=2)
    parser.add_argument('--LM_name',        type=str,   default='bert-base-chinese')
    parser.add_argument('--ckpt',           type=str,   default=None)

    # Mics
    parser.add_argument('--random_seed',    type=int,   default=769)
    parser.add_argument('--num_gpus',       type=int,   default=1)

    args = parser.parse_args()

    # Set random seed
    setup_seed(args.random_seed)

    # Multi-GPU
    device_ids = range(args.num_gpus)
    device = torch.device(device_ids[0]) if args.num_gpus != 0 else torch.device('cpu')

    if args.mode == 'train':
        # Train params
        n_epochs = args.epochs
        batch_size = args.batch_size
        lr = args.lr

        ############################
        #            DATA          #
        ############################
        train_set = LineCAPTCHA(args.data_dir, split='train')
        valid_set = LineCAPTCHA(args.data_dir, split='valid')
        train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=args.threads, shuffle=True,
                                  drop_last=True)
        valid_loader = DataLoader(valid_set, batch_size=batch_size, num_workers=args.threads, shuffle=True,
                                  drop_last=True)

        ############################
        #           MODEL          #
        ############################
        # Build Models
        model = UNet(n_channels=3, n_classes=args.num_cls)  # ResNet50

        # Checkpoint
        if args.ckpt:
            checkpoint = torch.load(args.ckpt, map_location=device)
            model.load_state_dict(checkpoint)
        model = model.to(device)

        ############################
        #         OPTIMIZER        #
        ############################
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr)
        optimizer.zero_grad()

        ############################
        #           TRAIN          #
        ############################
        # Train log
        train_iter = args.train_iter
        log_dir = os.path.join(args.exp_dir, args.exp_name, args.run_name)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir)

        for epoch in range(n_epochs):
            for t, batch in enumerate(train_loader):
                train_iter += 1
                model.train()
                # Load batch to device
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

                # Predict
                y_pr = model(batch['image'])        # (B, num_cls, H, W)
                y_pr = y_pr.permute(0, 2, 3, 1)     # (B, H, W, num_cls)
                y_gt = batch['mask']                # (B, H, W)

                # Compute Loss and Backward Pass
                B, H, W, num_cls = y_pr.shape
                y_pr_flatten = y_pr.reshape(B * H * W, num_cls)
                y_gt_flatten = y_gt.flatten()
                loss = criterion(y_pr_flatten, y_gt_flatten)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if train_iter % args.print_interval == 0:
                    print(loss.item())
                    writer.add_scalar('train/loss', loss.item(), train_iter)
                    writer.add_images('train/images', batch['image'], train_iter)
                    writer.add_images('train/mask', batch['mask'].unsqueeze(1), train_iter)

                if train_iter % args.valid_interval == 0 or t == train_loader.__len__() - 1:
                    metric = test(valid_loader, model, device, criterion, args)
                    writer.add_scalar('valid/loss', metric['loss'], train_iter)
                    writer.add_scalar('valid/accuracy', metric['accuracy'], train_iter)
                    writer.add_scalar('valid/f1_score', metric['f1_score'], train_iter)

                if train_iter % args.save_interval == 0:
                    torch.save(model.state_dict(), os.path.join(log_dir, f"{train_iter}.pth"))

    elif args.mode == 'test':
        test_set = LineCAPTCHA(args.data_dir, split='test')
        test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.threads, shuffle=False,
                                 drop_last=False)

        # Build Models
        model = UNet(n_channels=3, n_classes=args.num_cls)
        criterion = nn.CrossEntropyLoss()
        # Checkpoint
        if args.ckpt:
            checkpoint = torch.load(args.ckpt, map_location=device)
            model.load_state_dict(checkpoint)

        metric = test(test_loader, model, device, criterion, args)
        
        # Save image and results file
        if not os.path.exists('./results'):
            os.makedirs('./results/mask')
            os.makedirs('./results/image')
            os.makedirs('./results/segmentation')

        for i in range(len(metric['y_pr'])):
            image_mask = metric['y_pr'][i] / 1.0
            image_orig = metric['x_gt'][i] / 1.0
            image_segm = image_mask * image_orig

            torchvision.utils.save_image(image_mask, f'results/mask/{i:04}.png')
            torchvision.utils.save_image(image_orig, f'results/image/{i:04}.png')
            torchvision.utils.save_image(image_segm, f'results/segmentation/{i:04}.png')
        
        metric.pop('y_pr')
        metric.pop('y_gt')
        print(metric)


@torch.no_grad()
def test(dataloader, model, device, criterion, args):
    y_pr_list = []
    y_gt_list = []
    x_gt_list = []
    loss_list = []
    model = model.to(device)
    for t, batch in enumerate(dataloader):
        model.eval()
        # Load batch to device
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

        # Predict
        x_gt_list.append(batch['image'])
        y_pr = model(batch['image'])       # (B, num_cls, H, W)
        y_gt = batch['mask']               # (B, H, W)

        # Loss
        B, num_cls, H, W = y_pr.shape
        y_pr = y_pr.permute(0, 2, 3, 1)
        y_pr_flatten = y_pr.reshape(B * H * W, num_cls)
        y_gt_flatten = y_gt.flatten()
        loss = criterion(y_pr_flatten, y_gt_flatten)
        loss_list.append(loss.item())

        # Classify
        y_pr = torch.argmax(y_pr, dim=3)  # (B, H, W)

        y_pr_list.append(y_pr)
        y_gt_list.append(y_gt)
    
    x_gt = torch.cat(x_gt_list, dim=0)  # (N, 3, H, W)
    y_pr = torch.cat(y_pr_list, dim=0)  # (N, H, W)
    y_gt = torch.cat(y_gt_list, dim=0)  # (N, H, W)
    loss = torch.tensor(loss_list).mean().item()

    metric = evaluate_segmentation(y_gt, y_pr)
    metric['loss'] = loss
    metric['x_gt'] = x_gt
    metric['y_pr'] = y_pr
    metric['y_gt'] = y_gt

    return metric


if __name__ == '__main__':
    main()

