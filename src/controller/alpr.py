import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


class Alpr:
    def __init__(self, options=None):
        self.img_name = ''
        self.options = options

    def process_image(self, img_name):
        self.img_name = img_name

        img = cv.imread('../base/' + self.img_name + '.png', 0)

        highlight = self.highlight_plate(img)

        print('- Writting image')
        cv.imwrite('../bin/' + self.img_name + '.png', highlight)

    def highlight_plate(self, img):
        print('- Equalizing histogram')
        equalized = cv.equalizeHist(img)

        print('- Applying smoothing filter')
        kernel = np.ones((3, 3), np.float32) / 9
        filtered = cv.filter2D(equalized, -1, kernel)

        print('- Enhancing contrast')
        result = self.enhance_contrast(filtered)

        return result

    def enhance_contrast(self, img):
        kernel = np.ones((3, 3), np.uint8)

        top_hat = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel)
        black_hat = cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel)

        print('  - Removing noise from background')
        img_top = cv.add(img, top_hat)

        print('  - Removing noise from foreground')
        enhanced = cv.subtract(img_top, black_hat)

        return enhanced
