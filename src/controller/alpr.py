import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from model.hmf import HomomorphicFilter
from model.histogram import Histogram
from model.noise import Noise
from model.medianf import MedianFilter


class Alpr:
    def __init__(self, options=None):
        self.img_name = ''
        self.options = options

    def process_image(self, img_name):
        self.img_name = img_name

        img = cv.imread('../base/' + self.img_name + '.png', 0)
        img = cv.resize(img, (0, 0), fx=0.8, fy=0.8)
        cv.imwrite('../bin/' + self.img_name + '-gray.png', img)

        img = self.noise_filtering(img)

        processed = self.frequency_domain_filtering(img)

        print('- Writting image')
        cv.imwrite('../bin/' + self.img_name + '.png', processed)

    def noise_filtering(self, img):
        print('- Applying noise')
        noise = Noise("salt-pepper", img)
        noisy = noise.apply()

        cv.imwrite('../bin/' + self.img_name + '-noisy.png', noisy)

        print('- Filtering noise')
        window = 1
        thresold = 1
        mf = MedianFilter(noisy, window, thresold)

        median = mf.filter()
        cv.imwrite('../bin/' + self.img_name + '-clean.png', median)

        adaptive = mf.adaptive_filter()
        cv.imwrite('../bin/' + self.img_name + '-clean-adpt.png', adaptive)

        mf = MedianFilter(img, window, thresold)
        clean = mf.adaptive_filter()

        return clean

    def frequency_domain_filtering(self, img):
        print('- Frequency-domain Filtering')

        print('  - Homomorphic Filter')
        hmf = HomomorphicFilter(img)
        filtered = hmf.filter()

        print('  - Equalizing histogram')
        equalized, _, _, _ = Histogram.equalize(filtered)

        return equalized

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
