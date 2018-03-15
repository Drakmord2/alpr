import cv2 as cv
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

img = cv.imread('../base/cars-1.png', 0)

hist = cv.calcHist([img], [0], None, [256], [0, 256])
plt.plot(hist)
plt.xlim([0, 256])
plt.savefig('../bin/histogram.png')

equ = cv.equalizeHist(img)
cv.imwrite('../bin/cars-1-equalized.png', equ)

hist_eq = cv.calcHist([equ], [0], None, [256], [0, 256])
plt.plot(hist_eq)
plt.xlim([0, 256])
plt.savefig('../bin/equalized-histogram.png')

ret, threshold = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
mean = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
gausian = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

cv.imwrite('../bin/cars-1-threshold.png', threshold)
cv.imwrite('../bin/cars-1-mean-threshold.png', mean)
cv.imwrite('../bin/cars-1-gausian-threshold.png', gausian)
