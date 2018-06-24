import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from util.csvUtil import CSVUtil
from model.hmf import HomomorphicFilter
from model.histogram import Histogram
from model.noise import Noise
from model.medianf import MedianFilter
from model.kmeans import Kmeans
from model.threshold import Threshold
from model.morphology import Morphology
from model.representation import Representation
from model.compression import Compression
from model.color import Color
from model.knn import KNN


class Alpr:
    def __init__(self, options=None):
        self.img_name = ''
        self.options = options

    def process_image(self, img_name):
        self.img_name = img_name

        img = cv.imread('../base/' + self.img_name + '.png', 0)
        # img = self.crop(img)

        filtered = self.noise_filtering(img)
        filtered = self.frequency_domain_filtering(filtered)

        threshold = self.segmentation(filtered)
        threshold = self.morphology(threshold)

        contours = self.contour(threshold)

        if contours:
            self.classification(contours)

    def noise_filtering(self, img):
        print('- Filtering Noise')
        window = 1
        threshold = 1
        mf = MedianFilter(img, window, threshold)
        adaptive = mf.adaptive_filter()

        return adaptive

    def frequency_domain_filtering(self, img):
        print('- Frequency-domain Filtering')

        print('  - Homomorphic Filter')
        hmf = HomomorphicFilter(img)
        filtered = hmf.filter()

        cv.imwrite('../bin/' + self.img_name + '-filtered.png', filtered)

        return filtered

    def segmentation(self, img):
        print('- Segmentation')
        th = Threshold(img)

        window = 13
        mean_c = 6
        threshold = th.process_adaptive(window, mean_c)
        cv.imwrite('../bin/' + self.img_name + '-threshold.png', threshold)

        return threshold

    def morphology(self, img):
        print('- Morphology')
        morph = Morphology()

        kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
        closing = morph.closing(img, kernel, True)
        cv.imwrite('../bin/' + self.img_name + '-closing.png', closing)

        return closing

    def contour(self, img):
        print('- Contour')
        img = np.array(img, dtype=np.uint8)

        img_contours, contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        boxes = []
        for contour in contours:
            [x, y, w, h] = cv.boundingRect(contour)

            proportion = h / w
            if 1.3 < proportion < 1.7:
                # if w > 60 or h > 60:
                #     continue
                #
                # # if w < 30 or h < 40:
                # if w < 10 or h < 10:
                #     continue

                colors = self.contour_color()
                cv.rectangle(img, (x, y), (x + w, y + h), colors, 2)
                boxes.append([x, y, w, h])

            if 4 < proportion < 5.5:
                # if w > 15 or h > 80:
                #     continue
                #
                # if w < 3 or h < 14:
                #     continue

                colors = self.contour_color()
                cv.rectangle(img, (x, y), (x + w, y + h), colors, 2)
                boxes.append([x, y, w, h])

        cv.imwrite('../bin/' + self.img_name + '-contour.png', img)
        if len(boxes) >= 7:
            boxes_files = self.save_boxes(img_contours, boxes)

            return boxes_files

        return False

    def representation(self, img, show=False):
        moments = cv.moments(img, True)
        hu = cv.HuMoments(moments)
        hulog = -np.sign(hu) * np.log10(np.abs(hu))

        if show:
            print('- Representation')
            hustr = ''
            for i in range(len(hulog)):
                hustr = hustr + '\n\tI' + str(i+1) + ': ' + str(hu[i][0])

            print('   - Hu Moments: ', hustr)

        return hulog

    def classification(self, contours):
        print('- Classification')
        knn = KNN()

        symbols = []
        for file in contours:
            symbol = cv.imread(file, 0)
            hulog = self.representation(symbol)

            symbol = knn.classify(hulog)
            if symbol:
                symbols.append(symbol)

        print('Captured: ', symbols)
        if len(symbols) == 7:
            print('  - Possible License Plate: ', symbols)

    def compress(self, img, binary=False):
        print('- Compression')

        compression = Compression(img)
        compressed, tree = compression.huffman()

        if binary:
            str_compr = compression.run_length(compressed)
            self.write_file(str_compr, self.img_name + '-thresholdcompressed.txt')
            return

        decompressed = compression.decode_huffman(compressed, tree)
        cv.imwrite('../bin/' + self.img_name + '-huffman-decoded.png', decompressed)

    def contour_color(self):
        b = np.random.randint(50, 200)
        g = np.random.randint(50, 200)
        r = np.random.randint(50, 200)

        colors = (b, g, r)

        return colors

    def save_boxes(self, img, boxes):
        files = []
        cont = 0

        boxes = sorted(boxes, key=lambda boxi: boxi[1], reverse=True)

        for box in boxes:
            [x, y, w, h] = box

            cropped_box = img[y:y + h, x:x + w]

            file_name = '../bin/boxes/' + self.img_name + '-contour-' + str(cont) + '.png'
            cv.imwrite(file_name, cropped_box)
            files.append(file_name)

            cont += 1

        return files

    def crop(self, img):
        print('- Cropping')
        img = cv.resize(img, (0, 0), fx=0.8, fy=0.8)
        y, x = img.shape

        img = img[150:y-50, 100:x-500]

        cv.imwrite('../bin/' + self.img_name + '.png', img)

        return img

    def write_file(self, data, name):
        try:
            file = open('../bin/'+name, 'w')
            file.write(data)
        except Exception:
            print('\n* File error\n')
        finally:
            if file:
                file.close()
