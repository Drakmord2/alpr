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
from model.representation import Representation
from model.compression import Compression


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

        self.compress(processed)

        threshold = self.segmentation(processed)
        threshold = self.morphology(threshold)

        #  TODO Separar cada letra para representar e classificar (MSER?)
        self.representation(threshold)

    def compress(self, img):
        print('- Compression')
        compression = Compression(img)

        compressed, tree = compression.huffman()
        self.write(str(compressed), self.img_name + '-compressed.txt')

        decompressed = compression.decode(compressed, tree)
        cv.imwrite('../bin/' + self.img_name + '-huffman-decoded.png', decompressed)

    def noise_filtering(self, img):
        print('- Filtering Noise')
        window = 1
        threshold = 1
        mf = MedianFilter(img, window, threshold)
        adaptive = mf.adaptive_filter()

        return adaptive

    def segmentation(self, img):
        print('- Segmentation')
        th = Threshold(img)

        window = 11
        mean_c = 19
        threshold = th.process_adaptive(window, mean_c)
        cv.imwrite('../bin/' + self.img_name + '-threshold.png', threshold)

        return threshold

    def frequency_domain_filtering(self, img):
        print('- Frequency-domain Filtering')

        print('  - Homomorphic Filter')
        hmf = HomomorphicFilter(img)
        filtered = hmf.filter()

        cv.imwrite('../bin/' + self.img_name + '-filtered.png', filtered)

        return filtered

    def morphology(self, img):
        print('- Morphology')
        morph = Morphology()

        kernel = np.ones((5, 5))
        kernel = np.array(kernel, np.uint8)

        closing = morph.closing(img, kernel)

        cv.imwrite('../bin/' + self.img_name + '-closing.png', closing)

        return closing

    def representation(self, img):
        print('- Representation')

        moments = cv.moments(img, True)
        hu = cv.HuMoments(moments)
        hulog = -np.sign(hu) * np.log10(np.abs(hu))

        hustr = ''
        for i in range(len(hulog)):
            hustr = hustr + '\n\tI' + str(i+1) + ': ' + str(hu[i][0])

        print('   - Hu Moments: ', hustr)

        return hulog

    def crop(self, img):
        print('- Cropping')
        img = cv.resize(img, (0, 0), fx=0.8, fy=0.8)
        y, x = img.shape

        img = img[150:y-50, 100:x-500]

        cv.imwrite('../bin/' + self.img_name + '.png', img)

        return img

    def contour(self, img):
        print('- Contour')
        img = np.array(img, dtype=np.uint8)

        img_contours, contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

        for contour in contours:
            [x, y, w, h] = cv.boundingRect(contour)

            if h > 250 and w > 250:
                continue

            if h < 40 or w < 40:
                continue

            colors = self.contour_color()
            cv.rectangle(img, (x, y), (x + w, y + h), colors, 2)

        cv.imwrite('../bin/' + self.img_name + '-contours.png', img)

        return img_contours

    def save_contours(self, contours):
        files = []
        cont = 0
        for contour in contours:
            [x, y, w, h] = cv.boundingRect(contour)

            if h > 250 and w > 250:
                continue

            if h < 40 or w < 40:
                continue

            box = img[y:y + h, x:x + w]

            file_name = '../bin/' + self.img_name + '-contour-' + str(cont) + '.png'
            cv.imwrite(file_name, box)
            files.append(file_name)

            cont += 1

    def contour_color(self):
        b = np.random.randint(0, 200)
        g = np.random.randint(0, 200)
        r = np.random.randint(0, 200)

        colors = (b, g, r)

        return colors

    def write(self, data, name):
        try:
            file = open('../bin/'+name, 'w')
            file.write(data)
        except Exception:
            print('\n* File error\n')
        finally:
            if file:
                file.close()
