import argparse
import os
from char_sim import similarity

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision
import torch.nn.functional as F

from model.resnet import resnet50 as ResNet
from model.unet import UNet
from utils import *
from dataloader import CAPTCHA
import math

class LabelWeightedCrossEntropyLoss(nn.Module):
    
    def __init__(self, alpha=math.e):
        self.alpha = alpha

    def forward(self, input, target):
        alpha_target = self.alpha ** target - 1
        p_gt = alpha_target / torch.sum(alpha_target, dim=1)
        return torch.sum(-p_gt * F.log_softmax(input), dim=1) 


def main():
    parser = argparse.ArgumentParser(description='reCAPTCHA')

    # Experiment Parameters
    parser.add_argument('--mode',           type=str,   default='train',            choices=['train', 'test'])
    parser.add_argument('--task',           type=str,   default='segmentation_box', choices=['segmentation_box',
                                                                                             'segmentation_mask',
                                                                                             'recognition_bg',
                                                                                             'recognition_clean'])
    parser.add_argument('--exp_dir',        type=str,   default='exp_log')
    parser.add_argument('--exp_name',       type=str,   default=None)
    parser.add_argument('--run_name',       type=str,   default='bs_32')

    # Dataset Parameters
    parser.add_argument('--data_dir',       type=str,   default='data/captcha_click')
    parser.add_argument('--info_file',      type=str,   default='')
    parser.add_argument('--decoder_file',   type=str,   default='data/text/decoder_3000.json')
    parser.add_argument('--max_char',       type=int,   default=5)
    parser.add_argument('--H',              type=int,   default=200)
    parser.add_argument('--W',              type=int,   default=320)
    parser.add_argument('--char_size',      type=int,   default=50)
    parser.add_argument('--char_sim',       type=bool,  default=False,              action='store_true')
    parser.add_argument('--char_sim_file',   type=str,   default='similarity_3000.pt')



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

    if args.char_sim:
        similarity_mat = torch.load(args.char_sim_file).to(device)
    else:
        similarity_mat = None

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
        elif 'recognition' in args.task:
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
        criterion = LabelWeightedCrossEntropyLoss() if args.char_sim else nn.CrossEntropyLoss()
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
                loss = criterion(y_pr, similarity_mat[y_gt]) if args.char_sim else criterion(y_pr, y_gt)

                # Optimizer
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if train_iter % args.print_interval == 0:
                    print(loss.item())
                    writer.add_scalar('train/loss', loss.item(), train_iter)
                    writer.add_images('train/image', batch['image'] if 'segmentation' in args.task else batch['image'][0], train_iter)

                if train_iter % args.valid_interval == 0 or t == train_loader.__len__() - 1:
                    results = evaluate(valid_loader, model, device, criterion, args, similarity_mat=similarity_mat)
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
        elif 'recognition' in args.task:
            num_cls = args.num_cls if args.num_cls else 3000
            model = ResNet(pretrained=False, num_classes=num_cls)
        else:
            raise ValueError("Invalid task, only support segmentation and recognition")

        criterion = LabelWeightedCrossEntropyLoss() if args.char_sim else nn.CrossEntropyLoss()

        # Checkpoint
        if args.ckpt:
            checkpoint = torch.load(args.ckpt, map_location=device)
            model.load_state_dict(checkpoint)

        results = evaluate(test_loader, model, device, criterion, args)

        print("    loss: ", results['metric']['loss'])
        print("accuracy: ", results['metric']['accuracy'])

        # Save image and results file
        if not os.path.exists('./results'):
            os.makedirs('./results')
        fp = open('./results/results.txt', 'w')

        if 'segmentation' in args.task:
            os.makedirs('./results/mask')
            os.makedirs('./results/image')
            os.makedirs('./results/segmentation')
            for i, (image, mask) in enumerate(zip(results['x'], results['y_pr'])):
                mask = mask.float()
                segmentation = image * mask
                torchvision.utils.save_image(image, f'results/image/{i:04}.png')
                torchvision.utils.save_image(mask, f'results/mask/{i:04}.png')
                torchvision.utils.save_image(segmentation, f'results/segmentation/{i:04}.png')
        if 'recognition' in args.task:
            for i, (label, prediction) in enumerate(zip(results['y_gt'], results['y_pr'])):
                fp.write(f'id: {i:04}, label: {label}, prediction: {prediction}\n')

        for k, v in results['metric'].items():
            fp.write(f"{k}: {v}\n")


if __name__ == '__main__':
    main()
