import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import math


def crop_image(image, n_clusters=6, box_size=60):
    sample = np.where(image[..., 0] != 0)
    sample = np.array([[sample[0][i], sample[1][i]] for i in range(len(sample[0]))])

    k_means_class = KMeans(n_clusters=n_clusters)
    kmeans = k_means_class.fit(sample)
    center = kmeans.cluster_centers_.astype('int64')
    b = math.log10(kmeans.inertia_)
    print(b)

    image_stack = []
    H, W, C = image.shape
    for c in center:
        h, w = c[0], c[1]
        top = max(h - box_size // 2, 0)
        bottom = min(h + box_size // 2, H)
        left = max(w - box_size // 2, 0)
        right = min(w + box_size // 2, W)

        image_piece = image[top: bottom, left: right]
        image_stack.append(image_piece)

    image_stack = np.stack(image_stack)
    return image_stack


if __name__ == '__main__':
    img = Image.open('/Users/yxli/Downloads/results/segmentation/0007.png')
    img.save('o.png')
    img = np.array(img)
    img_s = crop_image(img)
    for i in range(len(img_s)):
        im = Image.fromarray(img_s[i])
        im.save(f'{i}.png')

