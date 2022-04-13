import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


def keep_smooth(image: np.ndarray, threshold=0.001):
    img_gradient = cv2.Laplacian(image, cv2.CV_64F)
    img_gradient = abs(img_gradient)
    img_gradient = img_gradient / img_gradient.max()
    #img_gradient[img_gradient > 0] = 1
    #img_gradient[img_gradient >= threshold] = 1
    #image_smooth = 1 - img_gradient
    image_smooth = (255 * img_gradient).astype('uint8')
    image_smooth = cv2.equalizeHist(image_smooth)
    image_smooth = cv2.medianBlur(image_smooth, 5)
    return image_smooth


def keep_variant(img_or, image: np.ndarray, kernel_size=10, threshold_min=10):
    H, W = image.shape
    for h in range(0, H - kernel_size):
        for w in range(0, W - kernel_size):
            kernel_h = image[h: h + kernel_size, w: w + kernel_size]
            #kernel_g = image[h: h + kernel_size, w: w + kernel_size, 1]
            #kernel_b = image[h: h + kernel_size, w: w + kernel_size, 2]
            std_h = kernel_h.std()
            #std_g = kernel_g.std()
            #std_b = kernel_b.std()
            if std_h < threshold_min:
                img_or[h: h + kernel_size, w: w + kernel_size] = 0

    return img_or


def mask(image_rgb):
    B = image_rgb[..., 0]
    G = image_rgb[..., 1]
    R = image_rgb[..., 2]

    diff_bg = abs(B - G)
    diff_gr = abs(G - R)
    diff_rb = abs(R - B)

    image_rgb[diff_bg < 30] = 0
    image_rgb[diff_gr < 30] = 0
    image_rgb[diff_rb < 30] = 0

    return image_rgb


def big_kernel(image, th=0, k_size_out=60, k_size_in=30):
    H, W, C = image.shape
    img_h = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[..., 0]
    cv2.imwrite('h.png', img_h)
    img_h = img_h.astype(np.float32)

    for h in range(H - k_size_out):
        for w in range(W - k_size_out):
            tl_i = h + (k_size_out - k_size_in) // 2, w + (k_size_out - k_size_in) // 2

            mean_i = img_h[tl_i[0]: tl_i[0] + k_size_in, tl_i[1]: tl_i[1] + k_size_in].mean()
            mean_a = img_h[h: h + k_size_out, w: w + k_size_out].mean()
            mean_o = (k_size_out**2 * mean_a - k_size_in**2 * mean_i) / (k_size_out**2 - k_size_in**2)
            diff = abs(mean_o - mean_i)
            print(h, w, diff)
            if diff < th:
                image[tl_i[0]: tl_i[0] + k_size_in, tl_i[1]: tl_i[1] + k_size_in] = 0

    return image


def main():
    image_name_list = os.listdir('/Users/yxli/Downloads/valid/')
    image_path_list = [os.path.join('/Users/yxli/Downloads/valid/', image_name) for image_name in image_name_list]
    for image_path, image_name in tqdm(zip(image_path_list, image_name_list)):
        image_bgr = cv2.imread(image_path)
        image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        image_h = image_hsv[..., 0]
        image_out = keep_variant(image_bgr, image_h, kernel_size=10, threshold_min=10)
        save_path = os.path.join('/Users/yxli/Downloads/valid_pre/', image_name)
        cv2.imwrite(save_path, image_out)


if __name__ == '__main__':
    main()
    """
    img = cv2.imread('/Users/yxli/Downloads/valid/valid_00020.png')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_r, img_g, img_b = img_rgb[..., 0], img_rgb[..., 1], img_rgb[..., 2]
    plt.imshow(img_rgb)
    plt.show()
    plt.imshow(img_r, cmap='gray', vmin=0, vmax=255)
    plt.show()
    plt.imshow(img_g, cmap='gray', vmin=0, vmax=255)
    plt.show()
    plt.imshow(img_b, cmap='gray', vmin=0, vmax=255)
    plt.show()

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_h, img_s, img_v = img_hsv[..., 0], img_hsv[..., 1], img_hsv[..., 2]
    plt.imshow(img_h, cmap='gray', vmin=0, vmax=255)
    plt.show()
    plt.imshow(img_s, cmap='gray', vmin=0, vmax=255)
    plt.show()
    plt.imshow(img_v, cmap='gray', vmin=0, vmax=255)
    plt.show()

    img_filter_h = keep_variant(img, img_h)
    plt.imshow(img_filter_h)
    plt.show()

    img_filter_r = keep_variant(img, img_r)
    plt.imshow(img_filter_r)
    plt.show()

    img_filter_g = keep_variant(img, img_g)
    plt.imshow(img_filter_g)
    plt.show()

    img_filter_b = keep_variant(img, img_b)
    plt.imshow(img_filter_b)
    plt.show()
    """
