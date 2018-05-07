import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from model.hmf import HomomorphicFilter
from model.histogram import Histogram
from model.noise import Noise
from model.medianf import MedianFilter
from model.kmeans import Kmeans
from model.threshold import Threshold
from model.morphology import Morphology


class Alpr:
    def __init__(self, options=None):
        self.img_name = ''
        self.options = options

    def process_image(self, img_name):
        self.img_name = img_name

        img = cv.imread('../base/' + self.img_name + '.png', 0)
        img = self.crop(img)

        filtered = self.noise_filtering(img)
        processed = self.frequency_domain_filtering(filtered)
        cv.imwrite('../bin/' + self.img_name + '-filtered.png', processed)

        threshold = self.segmentation(processed)
        cv.imwrite('../bin/' + self.img_name + '-threshold.png', threshold)

        self.morphology(threshold)

    def noise_filtering(self, img):
        print('- Filtering Noise')
        window = 1
        threshold = 1
        mf = MedianFilter(img, window, threshold)
        adaptive = mf.adaptive_filter()

        return adaptive

    def segmentation(self, img):
        print('- Segmentation')

        window = 5
        mean_c = 13
        th = Threshold(img)
        threshold = th.process_adaptive(window, mean_c)

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
        morph = Morphology()

        kernel = np.ones((3, 3))
        kernel = np.array(kernel, np.uint8)

        erosion = morph.erosion(img, kernel)
        dilation = morph.dilation(img, kernel)
        closing = morph.closing(img, kernel)
        opening = morph.opening(img, kernel)
        gradient = morph.gradient(img, kernel)
        top_hat = morph.top_hat(img, kernel)
        black_hat = morph.black_hat(img, kernel)

        cv.imwrite('../bin/' + self.img_name + '-erosion.png', erosion)
        cv.imwrite('../bin/' + self.img_name + '-dilation.png', dilation)
        cv.imwrite('../bin/' + self.img_name + '-closing.png', closing)
        cv.imwrite('../bin/' + self.img_name + '-opening.png', opening)
        cv.imwrite('../bin/' + self.img_name + '-gradient.png', gradient)
        cv.imwrite('../bin/' + self.img_name + '-tophat.png', top_hat)
        cv.imwrite('../bin/' + self.img_name + '-blackhat.png', black_hat)

    def crop(self, img):
        print('- Cropping')
        img = cv.resize(img, (0, 0), fx=0.8, fy=0.8)
        y, x = img.shape

        img = img[150:y-50, 0:x-100]

        cv.imwrite('../bin/' + self.img_name + '.png', img)

        return img
