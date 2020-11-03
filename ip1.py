import matplotlib.pyplot

matplotlib.use("TkAgg")

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io


class imageProcessing:
    def __init__(self, img):
        self.inputimg = img

    def show(self, outImg):
        plt.figure(figsize=(12, 3))
        plt.subplot(1, 2, 1)
        plt.title("input")
        plt.imshow(self.inputimg)

        plt.subplot(1, 2, 2)
        plt.title("answer")
        plt.imshow(outImg, cmap="gray")
        # plt.imshow(outImg)
        plt.show()

    def RGB2BGR(self):
        return self.inputimg[..., ::-1]

    def RGB2GRAY(self):
        _img = self.inputimg.copy().astype(np.float32)
        _img /= 255

        gray = 0.2126 * _img[..., 0] + 0.7152 * _img[..., 1] + 0.0722 * _img[..., 2]
        gray = np.clip(gray, 0, 255)
        return gray

    def binarization(self, th):
        _gray = self.RGB2GRAY() * 255
        _gray = _gray.astype(np.uint8)
        _gray = np.minimum(_gray // th, 1) * 255
        return _gray


if __name__ == "__main__":
    img_orig = io.imread(
        "https://yoyoyo-yo.github.io/Gasyori100knock/assets/imori_256x256.png"
    )
    # print(img_orig.shape)

    ip = imageProcessing(img_orig)
    outImg = ip.binarization(127)
    # outImg = ip.RGB2GRAY()
    ip.show(outImg)
