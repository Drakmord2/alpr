
import cv2 as cv
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

img = cv.imread('../base/car.jpg')

if img is not None:
    color = ('b', 'g', 'r')

    for i, col in enumerate(color):
        histr = cv.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])

    plt.savefig('../bin/hist.png')
