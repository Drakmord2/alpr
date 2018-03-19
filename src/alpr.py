import datetime as dt
import cv2 as cv
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt


class Alpr:
    def __init__(self, img_name):
        img = cv.imread('../base/'+img_name+'.png', 0)

        self.histogram(img, 'Original Image', 211)

        print('- Equalizing Histogram')
        equ = cv.equalizeHist(img)

        self.histogram(equ, 'Equalized Image', 212)

        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig('../bin/'+img_name+'-histograms.png')
        plt.close()

        print('- Getting Gradients')
        kernel = np.ones((3, 3), np.float32) / 9
        filtered = cv.filter2D(img, -1, kernel)
        laplacian = cv.Laplacian(filtered, cv.CV_64F, ksize=5)

        cv.imwrite('../bin/'+img_name+'-laplacian.png', laplacian)

        print('- Applying Threshold')
        gausian = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

        print('- Saving Image')
        cv.imwrite('../bin/'+img_name+'-gausian-threshold.png', gausian)

    @staticmethod
    def histogram(img, title, plot_num):
        hist = cv.calcHist([img], [0], None, [256], [0, 256])

        plt.subplot(plot_num)
        plt.plot(hist)
        plt.xlim([0, 256])
        plt.title(title)
        plt.xticks([0, 64, 192, 256])


if __name__ == '__main__':
    base = Path('../base')
    images = list(base.glob('**/*.png'))
    start_time = dt.datetime.utcnow()

    print('\t-- ALPR --')
    for i in images:
        file_name = i.parts[2]
        image_name = file_name.split('.png')[0]

        print('\n[ Processing '+file_name+' ] ')
        Alpr(image_name)

    end_time = dt.datetime.utcnow()
    total_time = (end_time - start_time).total_seconds()
    imageCount = len(images)

    print('\nStats:')
    print('- Images Processed: '+str(imageCount))
    print('- Time elapsed: '+str(total_time)+' seconds\n')
