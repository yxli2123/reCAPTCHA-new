import random
import json
from turtle import color
from PIL import Image
from PIL import ImageDraw, ImageFont
import os
import numpy as np
import colorsys
from tqdm import tqdm

class ImageChar:
    def __init__(self,
                 backgroundDir='./bg_pics_small',
                 fontDir='./font'):
        
        self.fontDir = fontDir
        self.backgroundDir = backgroundDir
        self.image = None
        self.pos = None
        self.fontSize = None

    def randomBackground(self):
        bg_list = os.listdir(self.backgroundDir)
        background = f'{self.backgroundDir}/{random.choice(bg_list)}'
        
        with Image.open(background) as im:
            self.image = im.copy().convert('RGBA')
            
        self.fontSize = int(self.image.height * 0.15) 

    def randomPosition(self, num_char, min_dist=True):
        pos = []
        height = self.image.height
        width = self.image.width
    
        while True:
            x = random.choice(
                range(int(0.1*width), int(0.75*width))
                )
            y = random.choice(
                range(int(0.1*height), int(0.75*height))
                )
            
            add = True
            if min_dist is not None:
                for p in pos:
                    if abs(p[0] - x) < min_dist and abs(p[1] - y) < min_dist:
                        add = False
                        break
            if add:
                pos.append((x, y))
            if len(pos) >= num_char:
                break
        return pos
        

    def drawChar(self, pos, rotate, char, font, color=None):
        size = int(1.5 * self.fontSize)
        txt=Image.new('RGBA', (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(txt)
        if color==None:
            img_data = np.asarray(self.image)
            bg_color = np.average(np.average(img_data[pos[1]:pos[1]+size, pos[0]:pos[0]+size], axis=0), axis=0)
            r, g, b, _ = bg_color
            h, _, _ = colorsys.rgb_to_hsv(r/255, g/255, b/255)
            h = h+0.5 if h<0.5 else h-0.5
            r, g, b = colorsys.hsv_to_rgb(h, 1, 1)
            color = (int(r*255), int(g*255), int(b*255), 255)
        elif len(color)==3:
            color = (color[0], color[1], color[2], 255)
        draw.text((0, 0), char, font=font, fill=color)
        del draw
        w = txt.rotate(rotate, expand=True)
        self.image.paste(w, pos, w)
        return (pos[0], pos[1], w.height, w.width)

    def generate(self, text):
        self.randomBackground()
        num = len(text)
        pos = self.randomPosition(num, min_dist=1.5*self.fontSize)
        rotate = [random.randint(-45, 45) for _ in range(num)]
        font_name_list = os.listdir(self.fontDir)
        fonts = [ImageFont.truetype(os.path.join(self.fontDir, f), self.fontSize) for f in font_name_list]
        bbox = []
        for i in range(0, num):
            bbox.append(self.drawChar(pos[i], rotate[i], text[i], random.choice(fonts)))
        return bbox

    def save(self, path):
        self.image.save(path, 'png')


def generate(name='captcha_single', split='train', num=1000):
    data_dir = f"./image/{name}"
    split_dir = os.path.join(data_dir, split)
    if not os.path.exists(split_dir):
        os.makedirs(split_dir)

    with open('./text/character_3000.txt', 'r') as fp:
        word_list = fp.read().split()
        word_dict = {k: v for v, k in enumerate(word_list)}  # {"/u2341": 3}


    index_info = []
    ic = ImageChar()

    print(f'| Generating {name} {split}')
    for i in tqdm(range(num)):
        data_id = i
        img_path = f"{split_dir}/{split}_{data_id:05d}.png"
        num_char = random.choice([4, 5])
        word = ''
        for _ in range(num_char):
            word += random.choice(word_list)
        bbox = ic.generate(word)
        index_info.append({'data_id': f'{data_id:05d}',
                            'word_label': word,
                            'word_label_id': [word_dict[w] for w in word],
                            'bbox': bbox,
                            'image_path': f"{split}_{data_id:05d}.png"})
        ic.save(img_path)

    with open(os.path.join(data_dir, f'{split}_info.json'), 'w') as info_file:
        json.dump(index_info, info_file, indent=4)
        print(f"Finished {split} dataset.")


if __name__ == '__main__':
    generate("captcha_click", "test", 1000)
    generate("captcha_click", "valid", 2000)
    generate("captcha_click", "train", 30000)

    