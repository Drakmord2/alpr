import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


class PlotUtil:
    def __init__(self, img_name):
        self.img_name = img_name

    def plot_bgr_histograms(self, img, equ):
        hist = cv.calcHist([img], [0], None, [256], [0, 256])
        self.subplot(hist, 'Original Image', 211)

        hist = cv.calcHist([equ], [0], None, [256], [0, 256])
        self.subplot(hist, 'Equalized Image', 212)

        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig('../bin/' + self.img_name + '-histograms.png')
        plt.close()

    def subplot(self, hist, title, plot_num):
        plt.subplot(plot_num)
        plt.plot(hist)
        plt.xlim([0, 256])
        plt.title(title)
        plt.xticks([0, 64, 192, 256])
