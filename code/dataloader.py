import torch
from torch.utils.data import Dataset
from torchvision import transforms

import os
from PIL import Image
import json


class LineCAPTCHA_mask(Dataset):
    def __init__(self, image_dir, split, info_file=None):
        self.info_file = info_file if info_file else os.path.join(image_dir, f'{split}_info.json')
        self.image_dir = image_dir
        self.split = split

        # self.data = []
        # self.load_data()
        self.transform = transforms.Compose([transforms.Resize((200, 320)),
                                             transforms.ToTensor()])
        info_file = open(self.info_file)
        self.data_info = json.load(info_file)

    def __getitem__(self, idx):
        """
        "data_id": "00000",
        "label": "\u7684",
        "label_id": 0,
        "image_path": "./test_00000.png"
        """
        sample = self.data_info[idx]
        # Load Images
        image_path = os.path.join(self.image_dir, self.split, sample['image_path'])
        image = self.transform(Image.open(image_path))
        image = image[0: 3]

        # Load mask
        mask_path = os.path.join(self.image_dir, self.split, sample['temp_image_path'])
        mask = self.transform(Image.open(mask_path))    # (4, H, W)
        mask = mask[0]                                  # (H, W)
        mask[mask != 0] = 1

        return {'image': image,  # (3, H, W) --batch--> (B, 3, H, W)
                'mask': mask,    # (H, W)    --batch--> (B, H, W)
                }

    def __len__(self):
        return len(self.data_info)



class LineCAPTCHA_box(Dataset):
    def __init__(self, image_dir, split, info_file=None):
        self.info_file = info_file if info_file else os.path.join(image_dir, f'{split}_info.json')
        self.image_dir = image_dir
        self.split = split

        # self.data = []
        # self.load_data()
        self.transform = transforms.Compose([transforms.Resize((200, 320)),
                                             transforms.ToTensor()])
        info_file = open(self.info_file)
        self.data_info = json.load(info_file)

    def __getitem__(self, idx):
        """
        "data_id": "00000",
        "label": "\u7684",
        "label_id": 0,
        "image_path": "./test_00000.png"
        """
        sample = self.data_info[idx]
        # Load Images
        image_path = os.path.join(self.image_dir, self.split, sample['image_path'])
        image = self.transform(Image.open(image_path))
        image = image[0: 3]

        # Box
        boxes = sample['bbox']
        boxes = torch.tensor(boxes)  # (N, 4), (x0, y0, x_offset, y_offset)

        # Load Images
        C, H, W = image.shape
        mask = torch.zeros(H, W).type(torch.LongTensor)
        for box in boxes:
            mask[box[1]: box[1] + box[3], box[0]: box[0] + box[2]] = 1

        return {'image': image,  # (3, H, W) --batch--> (B, 3, H, W)
                'mask': mask,    # (H, W)    --batch--> (B, H, W)
                }

    def __len__(self):
        return len(self.data_info)
