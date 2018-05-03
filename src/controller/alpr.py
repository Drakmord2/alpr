import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from model.hmf import HomomorphicFilter
from model.histogram import Histogram
from model.noise import Noise
from model.medianf import MedianFilter
from model.kmeans import Kmeans
from model.threshold import Threshold


class Alpr:
    def __init__(self, options=None):
        self.img_name = ''
        self.options = options

    def process_image(self, img_name):
        self.img_name = img_name

        img = cv.imread('../base/' + self.img_name + '.png', 0)
        img = cv.resize(img, (0, 0), fx=0.8, fy=0.8)
        cv.imwrite('../bin/' + self.img_name + '.png', img)

        filtered = self.noise_filtering(img)
        processed = self.frequency_domain_filtering(filtered)
        cv.imwrite('../bin/' + self.img_name + '-filtered.png', processed)

        threshold = self.segmentation(processed)
        cv.imwrite('../bin/' + self.img_name + '-threshold.png', threshold)

        self.morphology(threshold)

    def noise_filtering(self, img):
        print('- Filtering Noise')
        window = 1
        thresold = 1
        mf = MedianFilter(img, window, thresold)
        adaptive = mf.adaptive_filter()

        return adaptive

    def segmentation(self, img):
        print('- Segmentation')

        ret2, threshold = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        return threshold

    def frequency_domain_filtering(self, img):
        print('- Frequency-domain Filtering')

        print('  - Homomorphic Filter')
        hmf = HomomorphicFilter(img)
        filtered = hmf.filter()

        print('  - Equalizing Histogram')
        equalized, _, _, _ = Histogram.equalize(filtered)

        return equalized

    def morphology(self, img):
        print('- Morphology')

        kernel = [[0, 1, 0],
                  [1, 1, 1],
                  [0, 1, 0]]
        kernel = np.array(kernel, np.uint8)

        erosion = cv.erode(img, kernel, iterations=1)
        dilation = cv.dilate(img, kernel, iterations=1)
        opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
        closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
        gradient = cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel)

        top_hat = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel)
        black_hat = cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel)

        cv.imwrite('../bin/' + self.img_name + '-tophat.png', top_hat)
        cv.imwrite('../bin/' + self.img_name + '-blackhat.png', black_hat)
        cv.imwrite('../bin/' + self.img_name + '-erosion.png', erosion)
        cv.imwrite('../bin/' + self.img_name + '-dilation.png', dilation)
        cv.imwrite('../bin/' + self.img_name + '-opening.png', opening)
        cv.imwrite('../bin/' + self.img_name + '-closing.png', closing)
        cv.imwrite('../bin/' + self.img_name + '-gradient.png', gradient)

