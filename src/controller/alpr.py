import datetime as dt
import cv2 as cv
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt


class Alpr:
    def __init__(self, options=None):
        self.img_name = ''
        self.options = options

    def process(self, img_name):
        self.img_name = img_name

        img = cv.imread('../base/' + self.img_name + '.png', 0)

        print('- Equalizing Histogram')
        equ = cv.equalizeHist(img)

        self.plot_histograms(img, equ)

        print('- Getting Gradients')
        kernel = np.ones((3, 3), np.float32) / 9
        filtered = cv.filter2D(img, -1, kernel)
        laplacian = cv.Laplacian(filtered, cv.CV_64F, ksize=5)

        cv.imwrite('../bin/' + self.img_name + '-laplacian.png', laplacian)

        print('- Applying Threshold')
        gausian = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

        print('- Saving Image')
        cv.imwrite('../bin/' + self.img_name + '-gausian-threshold.png', gausian)

    def plot_histograms(self, img, equ):
        hist = cv.calcHist([img], [0], None, [256], [0, 256])
        self.subplot(hist, 'Original Image', 211)

        hist = cv.calcHist([equ], [0], None, [256], [0, 256])
        self.subplot(hist, 'Equalized Image', 212)

        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig('../bin/' + self.img_name + '-histograms.png')
        plt.close()

    @staticmethod
    def subplot(hist, title, plot_num):
        plt.subplot(plot_num)
        plt.plot(hist)
        plt.xlim([0, 256])
        plt.title(title)
        plt.xticks([0, 64, 192, 256])
