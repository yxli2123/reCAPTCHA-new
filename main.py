import argparse
import os

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision


from model.resnet import resnet50 as ResNet
from model.unet import UNet
from utils import *
from dataloader import CAPTCHA


def main():
    parser = argparse.ArgumentParser(description='reCAPTCHA')

    # Experiment Parameters
    parser.add_argument('--mode',           type=str,   default='train',            choices=['train', 'test'])
    parser.add_argument('--task',           type=str,   default='segmentation_box', choices=['segmentation_box',
                                                                                             'segmentation_mask',
                                                                                             'recognition_bg',
                                                                                             'recognition_clean'])
    parser.add_argument('--exp_dir',        type=str,   default='exp_log')
    parser.add_argument('--exp_name',       type=str,   default='segmentation_box')
    parser.add_argument('--run_name',       type=str,   default='bs_32')

    # Dataset Parameters
    parser.add_argument('--data_dir',       type=str,   default='../data/image/captcha_2ch_80/')
    parser.add_argument('--info_file',      type=str,   default='')
    parser.add_argument('--decoder_file',   type=str,   default='../data/text/decoder_3000.json')
    parser.add_argument('--max_char',       type=int,   default=5)
    parser.add_argument('--H',              type=int,   default=200)
    parser.add_argument('--W',              type=int,   default=320)
    parser.add_argument('--char_size',      type=int,   default=50)

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
    parser.add_argument('--num_cls',        type=int,   default=0)
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
        train_set = CAPTCHA(args.data_dir, split='train', task=args.task, args=args)
        valid_set = CAPTCHA(args.data_dir, split='valid', task=args.task, args=args)
        train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=args.threads, shuffle=True,
                                  drop_last=True)
        valid_loader = DataLoader(valid_set, batch_size=1, num_workers=args.threads, shuffle=True,
                                  drop_last=True)

        ############################
        #           MODEL          #
        ############################
        # Build Models
        if 'segmentation' in args.task:
            num_cls = args.num_cls if args.num_cls else 2
            model = UNet(n_channels=3, n_classes=num_cls)
        elif args.task == 'recognition':
            num_cls = args.num_cls if args.num_cls else 3000
            model = ResNet(pretrained=False, num_classes=num_cls)
        else:
            raise ValueError("Invalid task, only support segmentation and recognition")

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
        exp_name = args.exp_name if args.exp_name else args.task
        log_dir = os.path.join(args.exp_dir, exp_name, args.run_name)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir)

        for epoch in range(n_epochs):
            for t, batch in enumerate(train_loader):
                train_iter += 1
                model.train()

                # Load batch to device
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

                # Forward the input by the model
                y_pr, y_gt = forward(batch, model, args.task)

                # Loss
                loss = criterion(y_pr, y_gt)

                # Optimizer
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if train_iter % args.print_interval == 0:
                    print(loss.item())
                    writer.add_scalar('train/loss', loss.item(), train_iter)
                    writer.add_images('train/image', batch['image'], train_iter)

                if train_iter % args.valid_interval == 0 or t == train_loader.__len__() - 1:
                    results = evaluate(valid_loader, model, device, criterion, args)
                    writer.add_scalar('valid/loss', results['metric']['loss'], train_iter)
                    writer.add_scalar('valid/accuracy', results['metric']['accuracy'], train_iter)

                if train_iter % args.save_interval == 0:
                    torch.save(model.state_dict(), os.path.join(log_dir, f"{train_iter}.pth"))

    elif args.mode == 'test':
        test_set = CAPTCHA(args.data_dir, split='test', task=args.task, args=args)
        test_loader = DataLoader(test_set, batch_size=1, num_workers=args.threads, shuffle=False, drop_last=False)

        # Build Models
        if 'segmentation' in args.task:
            num_cls = args.num_cls if args.num_cls else 2
            model = UNet(n_channels=3, n_classes=num_cls)
        elif args.task == 'recognition':
            num_cls = args.num_cls if args.num_cls else 3000
            model = ResNet(pretrained=False, num_classes=num_cls)
        else:
            raise ValueError("Invalid task, only support segmentation and recognition")

        criterion = nn.CrossEntropyLoss()

        # Checkpoint
        if args.ckpt:
            checkpoint = torch.load(args.ckpt, map_location=device)
            model.load_state_dict(checkpoint)

        results = evaluate(test_loader, model, device, criterion, args)

        print(results['metric']['loss'])
        print(results['metric']['accuracy'])
        
        # Save image and results file
        if not os.path.exists('./results'):
            os.makedirs('./results/mask')
            os.makedirs('./results/image')
            os.makedirs('./results/segmentation')
        fp = open('./results/results.txt', 'r')

        if 'segmentation' in args.task:
            for i, image, mask in enumerate(zip(results['x'], results['y_pr'])):
                segmentation = image * mask
                torchvision.utils.save_image(image, f'results/image/{i:04}.png')
                torchvision.utils.save_image(mask, f'results/mask/{i:04}.png')
                torchvision.utils.save_image(segmentation, f'results/segmentation/{i:04}.png')
        if 'recognition' in args.task:
            for i, label, prediction in enumerate(zip(results['y_gt'], results['y_pr'])):
                fp.write(f'index: {i:04}, label: {label}, prediction: {prediction}\n')

        for k, v in results['metric'].items():
            fp.write(f"{k}: {v}")


if __name__ == '__main__':
    main()
