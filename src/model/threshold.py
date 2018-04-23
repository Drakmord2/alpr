import numpy as np
import cv2


class Threshold:
    def __init__(self, img, clusters, groups):
        self.img = img
        self.clusters = clusters
        self.groups = groups

    def process(self):
        threshold = np.zeros(self.img.shape, dtype=np.int)
        m, n = self.img.shape

        for i in range(m):
            for j in range(n):
                pixel = self.img[i, j]
                threshold[i, j] = self.get_cluster_value(pixel)

        return threshold

    def get_cluster_value(self, pixel):
        values = np.linspace(0, 255, self.groups, dtype=np.int)

        group = self.clusters[pixel]
        value = values[group]

        return value
