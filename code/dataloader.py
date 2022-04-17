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
        mask = mask.to(torch.long)

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
    
    
class LineCAPTCHA_rec(Dataset):
    def __init__(self, image_dir, split, info_file=None, padding=5):
        self.info_file = info_file if info_file else os.path.join(image_dir, f'{split}_info.json')
        self.image_dir = image_dir
        self.split = split
        self.padding = padding

        # self.data = []
        # self.load_data()
        self.transform = transforms.Compose([transforms.Resize((200, 320)),
                                             transforms.ToTensor()])
        self.transform_box = transforms.Resize((50, 50))
        
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
        image_path = os.path.join(self.image_dir, self.split, sample['temp_image_path'])
        image = self.transform(Image.open(image_path))
        image = image[0: 3]   # read only RGB, remove alpha channel

        # Box
        boxes = sample['bbox']
        boxes = torch.tensor(boxes)  # (N, 4), (x0, y0, x_offset, y_offset)
        
        # Labels
        label = sample['word_label_id']
        

        # Crop images to get characters
        character_list = []
        for box in boxes:
            character = image[..., box[1]: box[1] + box[3], box[0]: box[0] + box[2]]
            character = self.transform_box(character)      # (3, 50, 50)
            character_list.append(character)
        
        
        # Padding
        if len(label) < self.padding:
            label.append(label[0])
            character_list.append(character_list[0])
        elif len(label) > self.padding:
            raise ValueError("Padding number must be larger than length")
            
        
        label = torch.tensor(label).long()   # (padding, )
        character_list = torch.stack(character_list, dim=0)  # (padding, 3, 50, 50)
            

        return {'image': character_list,  # (padding, 3, H, W) --batch--> (B, padding, 3, H, W)
                'label': label,              # (padding, )        --batch--> (B, padding)
                }

    def __len__(self):
        return len(self.data_info)


