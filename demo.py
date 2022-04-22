import torch
import torch.nn as nn
from torchvision.utils import draw_bounding_boxes, save_image
import argparse
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from model.unet import UNet
from model.resnet import resnet18 as ResNet
import os
import json
from tqdm import tqdm
from crop import crop_stroke as crop
import random


def import_args():
    parser = argparse.ArgumentParser(description='reCAPTCHA')

    parser.add_argument('--captcha_dir',   type=str, default='data/captcha_click')
    parser.add_argument('--captcha_file',  type=str, default='')
    parser.add_argument('--captcha_label', type=str, default='')
    parser.add_argument('--H',             type=int, default=200)
    parser.add_argument('--W',             type=int, default=320)
    parser.add_argument('--char_size',     type=int, default=50)

    parser.add_argument('--encoder_file',  type=str, default='data/text/encoder_3000.json')
    parser.add_argument('--decoder_file',  type=str, default='data/text/decoder_3000.json')

    parser.add_argument('--ckpt_seg',      type=str, default='exp_log/segmentation_mask/mask_bs2_newbg/20000.pth')
    parser.add_argument('--ckpt_rec',      type=str, default='exp_log/recognition_clean/cleanTrain_cleanTeat_bs32/70000.pth')

    parser.add_argument('--num_gpus',      type=int, default=1)

    args = parser.parse_args()

    return args


def segment(image: torch.Tensor,
            model: nn.Module,
            device):
    image = image.to(device)
    x = image.to(device)                        # (3, H, W)
    x = x.unsqueeze(0)                          # (1, 3, H, W)
    logits = model(x)                           # (1, 2, H, W)
    mask = torch.argmax(logits, dim=1).float()  # (1, H, W)
    y = image * mask                            # (3, H, W)

    return y


def recognize(image_char: torch.Tensor,
              label: torch.Tensor,
              model: nn.Module,
              device):

    x = image_char.to(device)         # (N, 3, H, W)
    label = label.cpu().tolist()
    y_pr = model(x)                   # (N, 3000)
    
    y_candidate = y_pr[:, label]                 # (N, num_cls) --> (N, N)
    y_candidate = F.softmax(y_candidate, dim=1)  # (N, N)
    
    H, W = y_candidate.shape
    y = torch.zeros(H, dtype=torch.long)
    for i in range(H):
        loc = torch.argmax(y_candidate).item()   # index of the flattened NxN matrix 
        row, col = loc // H, loc % W
        y[row] = label[col]                      # sign label to the char

        # set the col and row to 0 for next iteration
        y_candidate[row, :] = 0
        y_candidate[:, col] = 0

    return torch.tensor(label).to(device)


def crop_segmentation(captcha: torch.Tensor, captcha_mask: torch.Tensor, args):
    # Boxes
    pad = transforms.Pad(2)
    captcha_mask_ = pad(captcha_mask)
    captcha_mask_[captcha_mask_ > 0] = 1
    captcha_mask_ = 255 * captcha_mask_.cpu().numpy()[0]  # (H, W), numpy array
    position_list = crop(captcha_mask_.astype('uint8'))   # (N) tuple of (upper left x, upper left y, width, height)
    boxes = torch.tensor(position_list)                   # (N, 4)
    boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]               # (N, 4) (xmin, ymin, xmax, ymax)
    
    for n in range(len(boxes)):
        boxes[n, 2] = boxes[n, 2] if boxes[n, 2] < 320 else 319
        boxes[n, 3] = boxes[n, 3] if boxes[n, 3] < 200 else 199
        
    

    # CAPTCHA
    captcha_list = []
    transform_character = transforms.Resize((args.char_size, args.char_size))
    for pos in position_list:
        single_word = captcha[..., pos[1]: pos[1]+pos[3], pos[0]:pos[0]+pos[2]]
        single_word = transform_character(single_word)
        captcha_list.append(single_word)

    captcha = torch.stack(captcha_list, dim=0)

    return captcha, boxes


def pipeline(captcha: torch.Tensor,
             label: torch.Tensor,
             model_seg: nn.Module,
             model_rec: nn.Module,
             device,
             args,
             index,
             ):
    captcha_mask = segment(captcha, model_seg, device)                       
    captcha_mask[captcha_mask > 0] = 1                                            # (3, H, W), binary
    
    save_image(captcha_mask, f"results/{index:04}_mask.png")
    
    captcha_seq, boxes_seq = crop_segmentation(captcha_mask, captcha_mask, args)  # (N, 3, H, W), (N, 4)
    print(boxes_seq)
   
    for i in range(len(boxes_seq)): 
        save_image(captcha_seq[i], f"results/{index:04}_char_{i}.png")
        
    prediction = recognize(captcha_seq, label, model_rec, device)                 # (N)

    return prediction, boxes_seq


def test():
    args = import_args()

    # Environment
    device_ids = range(args.num_gpus)
    device = torch.device(device_ids[0]) if args.num_gpus != 0 else torch.device('cpu')

    # Load Tokenizer
    with open(args.encoder_file, 'r') as encoder_file:
        encoder = json.load(encoder_file)

    with open(args.decoder_file, 'r') as decoder_file:
        decoder = json.load(decoder_file)

    # Load Model
    model_seg = UNet(n_channels=3, n_classes=2)
    model_rec = ResNet(pretrained=False, num_classes=3000)

    model_seg_ckpt = torch.load(args.ckpt_seg, map_location=device)
    model_rec_ckpt = torch.load(args.ckpt_rec, map_location=device)

    model_seg.load_state_dict(model_seg_ckpt)
    model_rec.load_state_dict(model_rec_ckpt)

    model_seg = model_seg.to(device)
    model_rec = model_rec.to(device)

    # List all CAPTCHA Image
    if args.captcha_dir:
        info_file = os.path.join(args.captcha_dir, 'test_info.json')
        info_file = open(info_file)
        data_info = json.load(info_file)

    elif args.captcha_file and args.captcha_label:
        data_info = [{'image_path': args.captcha_file,
                      'word_label_id': [encoder[word] for word in args.captcha_label]}]

    else:
        raise ValueError("either dir or file should not be None")

    # Predict
    transform = transforms.Compose([transforms.Resize((args.H, args.W)),
                                    transforms.ToTensor()])
    t = 0
    for sample in tqdm(data_info):
        t += 1
        # load image
        captcha_path = os.path.join(args.captcha_dir, 'test', sample['image_path']) if args.captcha_dir else sample['image_path']
        captcha_image = transform(Image.open(captcha_path))[0: 3]  # (3, H, W)

        # convert label: list to label: Tensor
        label = sample['word_label_id']
        label = torch.tensor(label).long()

        # predict the position and the character
        prediction, boxes = pipeline(captcha_image, label, model_seg, model_rec, device, args, t)

        # decode from token

        prediction = [decoder[str(token.item())] for token in prediction]
        captcha_image = torch.tensor(255 * captcha_image, dtype=torch.uint8)
        #print(prediction)
        recaptcha = torch.tensor(draw_bounding_boxes(captcha_image, boxes, labels=prediction, font='data/font/simsun.ttc', font_size=20), dtype=torch.float) / 255.0

        if not os.path.exists('./results'):
            os.makedirs('./results')
        result_name = sample['image_path'] if args.captcha_dir else "test.png"
        save_path = os.path.join('./results/', result_name)

        save_image(recaptcha, save_path)


if __name__ == '__main__':
    test()
