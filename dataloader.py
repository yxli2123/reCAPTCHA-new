import torch
from torch.utils.data import Dataset
from torchvision import transforms

import os
from PIL import Image
import json

    
class CAPTCHA(Dataset):
    def __init__(self, image_dir, split, task='segmentation_box', info_file=None, args=None):
        self.info_file = info_file if info_file else os.path.join(image_dir, f'{split}_info.json')
        self.image_dir = image_dir
        self.task = task
        self.split = split

        self.padding = args.smax_char
        self.H = args.H
        self.W = args.W
        self.char_size = args.char_size

        self.transform = transforms.Compose([transforms.Resize((self.H, self.W)),
                                             transforms.ToTensor()])
        self.transform_character = transforms.Resize((self.char_size, self.char_size))
        
        info_file = open(self.info_file)
        self.data_info = json.load(info_file)

    def __getitem__(self, idx):
        """
        "data_id": "00000",
        "label": "\u7684",
        "word_label_id": 0,
        "image_path": "./test_00000.png"
        "temp_image_path": "./temp_test_00000.png"
        """
        sample = self.data_info[idx]

        # Load Images
        image_path = os.path.join(self.image_dir, self.split, sample['image_path'])
        image = self.transform(Image.open(image_path))
        image = image[0: 3]   # read only RGB, remove alpha channel

        # Load Masks
        mask_path = os.path.join(self.image_dir, self.split, sample['temp_image_path'])
        mask = self.transform(Image.open(mask_path))    # (4, H, W)
        mask = mask[0]                                  # (H, W)
        mask[mask != 0] = 1

        # Load Boxes
        boxes = sample['bbox']
        boxes = torch.tensor(boxes).long()              # (N, 4), (x0, y0, x_offset, y_offset)
        
        # Load Labels
        label = sample['word_label_id']                 # list of length=N

        if self.task == 'segmentation_box':
            C, H, W = image.shape
            label = torch.zeros(H, W).long()
            for box in boxes:
                label[box[1]: box[1] + box[3], box[0]: box[0] + box[2]] = 1

            pair = {'image': image,
                    'label': label}

        elif self.task == 'segmentation_mask':
            mask = mask.long()
            pair = {'image': image,
                    'label': mask}

        elif 'recognition' in self.task:
            # Crop images to get characters
            character_list = []
            for box in boxes:
                if 'bg' in self.task:
                    character = image[:, box[1]: box[1] + box[3], box[0]: box[0] + box[2]]
                elif 'clean' in self.task:
                    character = mask[box[1]: box[1] + box[3], box[0]: box[0] + box[2]]
                    character = torch.stack([character, character, character], dim=0)  # convert to RGB channel
                else:
                    character = image[:, box[1]: box[1] + box[3], box[0]: box[0] + box[2]]

                character = self.transform_box(character)  # (3, 50, 50)
                character_list.append(character)

            # Padding
            if len(label) < self.padding:
                label.append(label[0])
                character_list.append(character_list[0])
            elif len(label) > self.padding:
                raise ValueError("Padding number must be larger than length")

            label = torch.tensor(label).long()  # (padding, )
            character_list = torch.stack(character_list, dim=0)  # (padding, 3, 50, 50)

            pair = {'image': character_list,  # (padding, 3, H, W) --batch--> (B, padding, 3, H, W)
                    'label': label,           # (padding, )        --batch--> (B, padding)
                    }

        else:
            raise ValueError("Invalid task")

        return pair

    def __len__(self):
        return len(self.data_info)


