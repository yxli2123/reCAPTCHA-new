import random
import json
from PIL import Image
from PIL import ImageDraw, ImageFont
import os
import numpy as np
import colorsys
from tqdm import tqdm

random.seed(0)


class ImageChar:
    def __init__(self,
                 backgroundDir='./bg_pics_small',
                 fontDir='./font'):

        self.fontDir = fontDir
        self.backgroundDir = backgroundDir
        self.image = None
        self.temp_image = None
        self.pos = None
        self.fontSize = None

    def randomBackground(self):
        bg_list = os.listdir(self.backgroundDir)
        background = f'{self.backgroundDir}/{random.choice(bg_list)}'

        with Image.open(background) as im:
            self.image = im.copy().convert('RGBA')
        self.temp_image = Image.new('RGBA', self.image.size, (0, 0, 0, 255))
        self.fontSize = int(self.image.height * 0.18)

    def randomPosition(self, num_char, min_dist=True):
        pos = []
        height = self.image.height
        width = self.image.width

        while True:
            x = random.choice(
                range(int(0.1 * width), int(0.75 * width))
            )
            y = random.choice(
                range(int(0.1 * height), int(0.75 * height))
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
        txt = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        txt_temp = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(txt)
        draw_temp = ImageDraw.Draw(txt_temp)
        if color == None:
            img_data = np.asarray(self.image)
            bg_color = np.average(np.average(img_data[pos[1]:pos[1] + size, pos[0]:pos[0] + size], axis=0), axis=0)
            r, g, b, _ = bg_color
            h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
            h = random.uniform(h + 0.25, h + 0.75)
            if h > 1:
                h -= 1
            if v < 0.55:
                v = random.uniform(0.7, 1)
            else:
                if s < 0.3:
                    v = random.choice([random.uniform(0, 0.3), random.uniform(0.7, 1)])
                else:
                    v = random.uniform(0.7, 1)
            s = random.uniform(0.7, 1)
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            color = (int(r * 255), int(g * 255), int(b * 255), 255)
        elif len(color) == 3:
            color = (color[0], color[1], color[2], 255)
        draw.text((0, 0), char, font=font, fill=color)
        draw_temp.text((0, 0), char, font=font, fill=(255, 255, 255, 255))
        del draw
        del draw_temp
        w = txt.rotate(rotate, expand=True)
        w_temp = txt_temp.rotate(rotate, expand=True)
        self.image.paste(w, pos, w)
        self.temp_image.paste(w_temp, pos, w_temp)
        return (pos[0], pos[1], w.height, w.width)

    def generate(self, text):
        self.randomBackground()
        num = len(text)
        pos = self.randomPosition(num, min_dist=1.5 * self.fontSize)
        rotate = [random.randint(-60, 60) for _ in range(num)]
        font_name_list = os.listdir(self.fontDir)
        fonts = [ImageFont.truetype(os.path.join(self.fontDir, f), self.fontSize) for f in font_name_list]
        bbox = []
        for i in range(0, num):
            bbox.append(self.drawChar(pos[i], rotate[i], text[i], random.choice(fonts)))
        return bbox

    def save(self, path1, path2):
        self.image.save(path1, 'png')
        self.temp_image.save(path2, 'png')


def generate(name='captcha_click', split='train', num=1000):
    data_dir = f"./{name}"
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
        img_path1 = f"{split_dir}/{split}_{data_id:05d}.png"
        img_path2 = f"{split_dir}/{split}_{data_id:05d}_temp.png"
        num_char = random.choice([4, 5])
        word = ''
        for _ in range(num_char):
            word += random.choice(word_list)
        bbox = ic.generate(word)
        index_info.append({'data_id': f'{data_id:05d}',
                           'word_label': word,
                           'word_label_id': [word_dict[w] for w in word],
                           'bbox': bbox,
                           'image_path': f"{split}_{data_id:05d}.png",
                           'temp_image_path': f"{split}_{data_id:05d}_temp.png"})
        ic.save(img_path1, img_path2)

    with open(os.path.join(data_dir, f'{split}_info.json'), 'w') as info_file:
        json.dump(index_info, info_file, indent=4)
        print(f"Finished {split} dataset.")


if __name__ == '__main__':
    generate("captcha_click", "test", 1000)
    generate("captcha_click", "valid", 2000)
    generate("captcha_click", "train", 30000)

