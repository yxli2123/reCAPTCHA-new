import torch
from torch.utils.data import Dataset
from torchvision.transforms import PILToTensor

import os
from PIL import Image
import json


class LineCAPTCHA(Dataset):
    def __init__(self, image_dir, split, num_character=2, info_file=None):
        self.info_file = info_file if info_file else os.path.join(image_dir, f'{split}_info.json')
        self.image_dir = image_dir
        self.split = split
        self.num_character = num_character

        # self.data = []
        # self.load_data()
        self.transform = PILToTensor()
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
        image = self.transform(Image.open(image_path)) / 255.0

        image = torch.stack(torch.chunk(image, self.num_character, dim=2), dim=0)

        # Tokenize Labels
        label = sample['label_id']
        label = torch.tensor(label)
        return {'image': image,  # (num_char, 3, H, W) --batch--> (B, num_char, 3, H, W)
                'label': label,  # (num_char)          --batch--> (B, num_char)
                'text': sample['label']
               }


    def __len__(self):
        return len(self.data_info)

    def load_data(self):
        transform = PILToTensor()
        info_file = open(self.info_file)
        data_info = json.load(info_file)
        """
        "data_id": "00000",
        "label": "\u7684",
        "label_id": 0,
        "image_path": "./test_00000.png"
        """
        for sample in data_info:
            # Load Images
            image_path = os.path.join(self.image_dir, self.split, sample['image_path'])
            image = transform(Image.open(image_path)) / 255.0

            image = torch.stack(torch.chunk(image, self.num_character, dim=2), dim=0)

            # Tokenize Labels
            label = sample['label_id']
            label = torch.tensor(label)
            self.data.append({'image': image,  # (num_char, 3, H, W) --batch--> (B, num_char, 3, H, W)
                              'label': label,  # (num_char)          --batch--> (B, num_char)
                              'text': sample['label']
                              })


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    test_set = LineCAPTCHA('../data/image/captcha_1ch_10/', 'test', 1)
    dataloader = DataLoader(test_set, batch_size=4, shuffle=True)
    for batch in dataloader:
        y_gt = batch['label']
        y_gt_ = y_gt.reshape(4*1)
        print(batch)


