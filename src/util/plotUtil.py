import datetime as dt
import cv2 as cv
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt


class PlotUtil:
    def __init__(self):
        pass

    def plot_bgr_histograms(self, img, equ):
        color = ('b', 'g', 'r')

        plt.subplot(211)
        for i, col in enumerate(color):
            hist = cv.calcHist([img], [i], None, [256], [0, 256])
            self.subplot(hist, 'Original Image')

        plt.subplot(212)
        for i, col in enumerate(color):
            hist = cv.calcHist([equ], [i], None, [256], [0, 256])
            self.subplot(hist, 'Equalized Image')

        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig('../bin/' + self.img_name + '-histograms.png')
        plt.close()

    def subplot(self, hist, title):
        plt.plot(hist)
        plt.xlim([0, 256])
        plt.title(title)
        plt.xticks([0, 64, 192, 256])
