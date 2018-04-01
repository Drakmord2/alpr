import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from model.hmf import HomomorphicFilter
from model.histogram import Histogram
from util.plotUtil import PlotUtil


class Alpr:
    def __init__(self, options=None):
        self.img_name = ''
        self.options = options

    def process_image(self, img_name):
        self.img_name = img_name

        img = cv.imread('../base/' + self.img_name + '.png', 0)
        img = cv.resize(img, (0, 0), fx=0.8, fy=0.8)

        space = self.space_domain_filtering(img)
        freq = self.frequency_domain_filtering(space)

        print('- Writting image')
        cv.imwrite('../bin/' + self.img_name + '.png', freq)

    def frequency_domain_filtering(self, img):
        print('- Frequency-domain Filtering')

        print('  - Homomorphic Filter')
        hmf = HomomorphicFilter(img)
        filtered = hmf.filter()

        print('  - Equalizing histogram')
        result, _, _, _ = Histogram.equalize(filtered)

        return result

    def space_domain_filtering(self, img):
        print('- Space-domain Filtering')

        print('  - Equalizing histogram')
        equalized, _, _, _ = Histogram.equalize(img)

        print('  - Applying smoothing filter')
        kernel = np.ones((3, 3), np.float32) / 9
        filtered = cv.filter2D(equalized, -1, kernel)  # TODO Implement filter

        print('  - Improving contrast')
        result = self.enhance_contrast(filtered)  # TODO Implement functions

        return result

    def enhance_contrast(self, img):
        kernel = np.ones((3, 3), np.uint8)

        top_hat = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel)
        black_hat = cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel)

        print('  - Enhancing details')
        img_top = cv.add(img, top_hat)

        print('  - Removing noise from foreground')
        enhanced = cv.subtract(img_top, black_hat)

        return enhanced
