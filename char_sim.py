# Copied from https://github.com/houbb/nlp-hanzi-similar/releases/tag/pythn
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import torch
import cv2 as cv
from PIL import Image
from PIL import ImageDraw, ImageFont
from sewar.full_ref import uqi
import numpy as np
from tqdm import tqdm

def similarity(char1, char2):
    font = ImageFont.truetype('./data/font/MSYHBD.TTC', 30)
    char_img1 = Image.new('L', (50, 50))
    draw1 = ImageDraw.Draw(char_img1)
    draw1.text((0, 0), char1, font=font)
    del draw1

    char_img2 = Image.new('L', (50, 50))
    draw2 = ImageDraw.Draw(char_img2)
    draw2.text((0, 0), char2, font=font)
    del draw2

    return uqi(np.asarray(char_img1), np.asarray(char_img2))


if __name__ == '__main__':
    with open('./data/text/decoder_3000.json', 'r') as f:
        decoder = json.load(f)
    chars = [v for k, v in decoder.items()]
    num_chars = len(chars)
    similarity_mat = torch.zeros((num_chars, num_chars))
    for i in tqdm(range(num_chars)):
        for j in range(i, num_chars):
            if j == i:
                similarity_mat[i][j] = 1.
                continue
            sim = similarity(decoder[str(i)], decoder[str(j)])
            similarity_mat[i][j] = sim
            similarity_mat[j][i] = sim
    torch.save(similarity_mat, f'similarity_{num_chars}.pt')
