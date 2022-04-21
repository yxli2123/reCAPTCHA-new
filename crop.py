import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def dist_point(px, py):
    return max(abs(px[0] - py[0]), abs(px[1] - py[1]))

def dist_line(pairx, pairy):
    def ordered(pair):
        if pair[0][0] > pair[1][0]:
            return (pair[1], pair[0])
        return pair
    ordered_pairx = ordered(pairx)
    ordered_pairy = ordered(pairy)
    return max(
        dist_point(ordered_pairx[0], ordered_pairy[0]),
        dist_point(ordered_pairx[1], ordered_pairy[1]),
    )

def line_length(pair):
    return dist_point(pair[0], pair[1])

def harris_detect(img):
    dst = cv.cornerHarris(img, 2, 3, 0.02)
    new_img = np.zeros_like(img)
    new_img[dst > 0.1*dst.max()] = 255
    return new_img, np.array(np.where(dst > 0.1*dst.max())).transpose()

def diag_detect(img, th=3):
    corner, points = harris_detect(img)
    dist_x = points[:, 0].reshape(-1, 1) - points[:, 0].reshape(1, -1)
    dist_y = points[:, 1].reshape(-1, 1) - points[:, 1].reshape(1, -1)

    diag_point_pairs1 = np.array(np.where(np.abs(dist_x - dist_y) <= th)).transpose()
    diag_point_pairs2 = np.array(np.where(np.abs(dist_x + dist_y) <= th)).transpose()
    diag_point_pairs = np.concatenate([diag_point_pairs1, diag_point_pairs2], axis=0)

    return [((points[p[0]][1], points[p[0]][0]), (points[p[1]][1], points[p[1]][0])) for p in diag_point_pairs]

def merge_pairs(diag_point_pairs, min_dist=10):
    merged_pairs = {}
    min_dist = 10

    for pair in diag_point_pairs:
        if merged_pairs == {}:
            merged_pairs[0] = [pair]
        new_k = True
        for k, v in merged_pairs.items():
            add_k = True
            for pk in v:
                if dist_line(pk, pair) >= min_dist:
                    add_k = False
                    break
            if add_k:
                new_k = False
                merged_pairs[k].append(pair)
                break
        if new_k:
            merged_pairs[len(merged_pairs)] = [pair]
    return [v[0] for k, v in merged_pairs.items()]

def diag_to_square(diag_pair):
    width = abs(diag_pair[0][0] - diag_pair[1][0])
    height = abs(diag_pair[0][1] - diag_pair[1][1])
    return (min(diag_pair[0][0], diag_pair[1][0]), min(diag_pair[0][1], diag_pair[1][1]), width, height)

def clean_pairs(merged_pairs, img):
    lengths = [line_length(p) for p in merged_pairs]
    cleaned_pairs = []
    for pair, length in zip(merged_pairs, lengths):
        if length <= 40 or length >= 80:
            continue
        center = (int((pair[0][1]+pair[1][1])/2), int((pair[0][0]+pair[1][0])/2))
        if not img[center]:
            continue
        binary = cv.threshold(img, 127, 255, cv.THRESH_BINARY)[1]
        rec = diag_to_square(pair)
        area = rec[2]*rec[3]
        rec_img = binary[rec[1]:rec[1]+rec[3], rec[0]:rec[0]+rec[2]]
        if rec_img.sum() / rec_img.max() <= 0.8* area:
            continue
        cleaned_pairs.append(pair)

    for i in range(len(cleaned_pairs)):
        if cleaned_pairs[i] is None:
            continue
        for j in range(i+1, len(cleaned_pairs)):
            if cleaned_pairs[j] is None:
                continue
            reci = diag_to_square(cleaned_pairs[i])
            recj = diag_to_square(cleaned_pairs[j])

            if reci[0] > recj[0] + recj[2] or\
               reci[1] > recj[1] + recj[3] or\
               reci[0] + reci[2] < recj[0] or\
               reci[1] + reci[3] < recj[1]:
                continue

            overlap_width = min(reci[0]+reci[2], recj[0]+recj[2]) - max(reci[0], recj[0])
            overlap_height = min(reci[1]+reci[3], recj[1]+recj[3]) - max(reci[1], recj[1])
            overlap = overlap_width*overlap_height
            areai = reci[2]*reci[3]
            areaj = recj[2]*recj[3]
            if overlap / (areai + areaj - overlap) >= 0.4:
                if areai < areaj:
                    cleaned_pairs[i] = cleaned_pairs[j]
                cleaned_pairs[j] = None
    return [p for p in cleaned_pairs if p is not None]


def crop(img):
    diag_point_pairs = diag_detect(img)
    cleaned_pairs = clean_pairs(merge_pairs(diag_point_pairs), img)
    return [diag_to_square(pair) for pair in cleaned_pairs]

def crop_stroke(img):
    binary = cv.threshold(img, 127, 255, cv.THRESH_BINARY)[1]
    binary.astype(np.uint8)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    closed = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel, iterations=3)
    contours, hier = cv.findContours(closed, cv.RETR_EXTERNAL,
                                    cv.CHAIN_APPROX_SIMPLE)
    recs = []
    for cnt in contours:
        rec = cv.boundingRect(cnt)
        width = int(1.5 * max(rec[2], rec[3]))
        x = int(rec[0] - min(0.25 * width, rec[0]))
        y = int(rec[1] - min(0.25 * width, rec[1]))
        recs.append((x, y, width, width))
    return recs

if __name__ == '__main__':
    img = cv.imread('/home/v-kehanwu/reCAPTCHA/tmp_image/0001.png')
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    print(crop(gray))
