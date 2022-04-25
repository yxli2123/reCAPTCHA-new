# Copied from https://github.com/houbb/nlp-hanzi-similar/releases/tag/pythn
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import torch
import cv2 as cv
from PIL import Image
from PIL import ImageDraw, ImageFont
# from sewar.full_ref import uqi
import numpy as np
from tqdm import tqdm

def similarity(char1, char2):
    font_size = 128
    img_size = int(font_size * 1.25)
    font = ImageFont.truetype('./data/font/MSYHBD.TTC', font_size)
    char_img1 = Image.new('L', (img_size, img_size))
    draw1 = ImageDraw.Draw(char_img1)
    draw1.text((0, 0), char1, font=font, fill=255)
    del draw1

    char_img2 = Image.new('L', (img_size, img_size))
    draw2 = ImageDraw.Draw(char_img2)
    draw2.text((0, 0), char2, font=font, fill=255)
    del draw2

    char_array1 = np.int32(np.asarray(char_img1).reshape(-1) / 255)
    char_array2 = np.int32(np.asarray(char_img2).reshape(-1) / 255)
    overlap = (char_array1 * char_array2).sum()
    sim = overlap / (char_array1.sum() + char_array2.sum() - overlap)
    return sim

if __name__ == '__main__':
    with open('./data/text/decoder_3000.json', 'r') as f:
        decoder = json.load(f)
    chars = [v for k, v in decoder.items()]
    num_chars = len(chars)
    similarity_mat = torch.ones((num_chars, num_chars))
    for i in tqdm(range(num_chars)):
        for j in range(i+1, num_chars):
            sim = similarity(decoder[str(i)], decoder[str(j)])
            similarity_mat[i][j] = sim
            similarity_mat[j][i] = sim
    torch.save(similarity_mat, f'similarity_{num_chars}.pt')
