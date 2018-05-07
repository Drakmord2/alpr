import numpy as np
import cv2 as cv


class Threshold:
    def __init__(self, img, clusters=None, groups=None):
        self.img = img
        self.clusters = clusters
        self.groups = groups

    def process_adaptive(self, window, mean_c):
        if window % 2 != 1:
            window += 1

        window_size = window * window
        xlength, ylength = self.img.shape
        threshold = np.zeros(self.img.shape)

        for y in range(window, ylength - window):
            for x in range(window, xlength - window):

                x1, x2 = [x - window, x]
                y1, y2 = [y - window, y]
                window_array = self.img[x1:x2, y1:y2]

                pixels = np.reshape(window_array, window_size)

                mean = self.get_mean(pixels)
                t = mean - mean_c
                posx, posy = [x - 1 - window // 2, y - 1 - window // 2]

                threshold[posx, posy] = 0 if self.img[posx, posy] >= t else 255

        return threshold

    def get_mean(self, values):
        size = len(values)
        sum = values.sum()

        mean = sum // size

        return mean

    def process_kmeans(self):
        threshold = np.zeros(self.img.shape, dtype=np.int)
        m, n = self.img.shape

        for i in range(m):
            for j in range(n):
                pixel = self.img[i, j]
                threshold[i, j] = self.get_cluster_value(pixel)

        return threshold

    def process_otsu(self):
        pass

    def get_cluster_value(self, pixel):
        values = np.linspace(0, 255, self.groups, dtype=np.int)

        group = self.clusters[pixel]
        value = values[group]

        return value
