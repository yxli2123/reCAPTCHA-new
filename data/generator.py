import random
import json
from PIL import Image
from PIL import ImageDraw, ImageFont
import os


class ImageChar:
    def __init__(self,
                 height,
                 num_char,
                 fontDir='./font'):

        self.size = (height*num_char, height)
        self.fontSize = int(height * 0.8)
        self.fontDir = fontDir
        self.num_char = num_char

        self.bgColor = (255, 255, 255)
        self.fontColor = (0, 0, 0)
        self.fontPath = './font/simsun.ttc'
        self.randomConfig()

        self.font = ImageFont.truetype(self.fontPath, self.fontSize)
        self.image = Image.new('RGB', self.size, self.bgColor)

    def randomConfig(self):
        self.bgColor = (random.randint(200, 255), random.randint(200, 255), random.randint(200, 255))
        self.fontColor = (random.randint(0, 100), random.randint(0, 100), random.randint(0, 100))
        font_name_list = os.listdir(self.fontDir)
        fontPath = [os.path.join(self.fontDir, font_name) for font_name in font_name_list]
        self.fontPath = random.choice(fontPath)

    def rotate(self):
        self.image.rotate(random.randint(0, 45), expand=False)

    def drawText(self, pos, txt, fill):
        draw = ImageDraw.Draw(self.image)
        draw.text(pos, txt, font=self.font, fill=fill)
        del draw

    def randPoint(self):
        (width, height) = self.size
        return random.randint(0, width), random.randint(0, height)

    def randLine(self, num):
        draw = ImageDraw.Draw(self.image)
        for i in range(0, num):
            draw.line([self.randPoint(), self.randPoint()],
                      (random.randint(128, 255), random.randint(128, 255), random.randint(128, 255)))
        del draw

    def randChinese(self, text, num_line):
        gap = 8
        num = len(text)
        start = 2
        self.randLine(num_line//2)
        for i in range(0, num):
            x = start + self.fontSize * i + random.randint(2, gap) + gap * i
            self.drawText((x, random.randint(0, 5)), text[i], self.fontColor)
            self.randomConfig()
            self.rotate()
        self.randLine(num_line//2)

    def save(self, path):
        self.image.save(path, 'png')

    def generate(self, text, dirty):
        assert len(text) == self.num_char, "Number of char in the class doesn't much the text length"
        self.randChinese(text, dirty)


def generate(name='captcha_single', split='train', dirty=20):
    data_dir = f"./image/{name}_{dirty:02}"
    split_dir = os.path.join(data_dir, split)
    if not os.path.exists(split_dir):
        os.makedirs(split_dir)

    with open('./text/word_1ch.txt', 'r') as fp:
        word_list = fp.read().split()
        word_dict = {k: v for v, k in enumerate(word_list)}  # {"/u2341": 3}

    with open('./text/word_2ch.txt', 'r') as fp:
        text = fp.read().split()  # ["出生", "中国", "背景", ...]
        index_info = []

        repeat_time = 10 if split == 'train' else 1

        for t in range(repeat_time):
            for i, word in enumerate(text):
                data_id = t * len(text) + i
                img_path = f"{split_dir}/{split}_{data_id:05d}.png"

                index_info.append({'data_id': f'{data_id:05d}',
                                   'label': word,
                                   'label_id': [word_dict[word[0]], word_dict[word[1]]],
                                   'image_path': f"{split}_{data_id:05d}.png"})

                ic = ImageChar(height=40, num_char=2, fontDir='./font')
                ic.generate(word, dirty)
                ic.save(img_path)

    with open(os.path.join(data_dir, f'{split}_info.json'), 'w') as info_file:
        json.dump(index_info, info_file, indent=4)
        print(f"Finished {split} dataset.")


if __name__ == '__main__':
    generate("captcha_2ch", "test", 0)
    generate("captcha_2ch", "valid", 0)
    generate("captcha_2ch", "train", 0)

    generate("captcha_2ch", "test", 10)
    generate("captcha_2ch", "valid", 10)
    generate("captcha_2ch", "train", 10)

    generate("captcha_2ch", "test", 20)
    generate("captcha_2ch", "valid", 20)
    generate("captcha_2ch", "train", 20)

    generate("captcha_2ch", "test", 30)
    generate("captcha_2ch", "valid", 30)
    generate("captcha_2ch", "train", 30)

    generate("captcha_2ch", "test", 40)
    generate("captcha_2ch", "valid", 40)
    generate("captcha_2ch", "train", 40)

    generate("captcha_2ch", "test", 50)
    generate("captcha_2ch", "valid", 50)
    generate("captcha_2ch", "train", 50)

    generate("captcha_2ch", "test", 60)
    generate("captcha_2ch", "valid", 60)
    generate("captcha_2ch", "train", 60)

    generate("captcha_2ch", "test", 70)
    generate("captcha_2ch", "valid", 70)
    generate("captcha_2ch", "train", 70)

    generate("captcha_2ch", "test", 80)
    generate("captcha_2ch", "valid", 80)
    generate("captcha_2ch", "train", 80)


