import datetime as dt
import cv2 as cv
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt


class Alpr:
    def __init__(self, options=None):
        self.img_name = ''
        self.options = options

    def process_image(self, img_name):
        self.img_name = img_name

        img = cv.imread('../base/' + self.img_name + '.png')

        highlight = self.highlight_plate(img)

        print('- Writting image')
        cv.imwrite('../bin/' + self.img_name + '.png', highlight)

    def highlight_plate(self, img):
        print('- Equalizing histogram')
        img_yuv = cv.cvtColor(img, cv.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = cv.equalizeHist(img_yuv[:, :, 0])
        img_output = cv.cvtColor(img_yuv, cv.COLOR_YUV2BGR)

        print('- Applying smoothing filter')
        kernel = np.ones((3, 3), np.float32) / 9
        filtered = cv.filter2D(img_output, -1, kernel)
        laplacian = cv.Laplacian(filtered, cv.CV_64F, ksize=5)

        print('- Merging layers')
        sub = cv.subtract(img_output, laplacian, dtype=cv.CV_64F)
        result = cv.add(img_output, sub, dtype=cv.CV_64F)

        return result
