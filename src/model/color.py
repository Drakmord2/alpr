import numpy as np


class Color:
    def __init__(self, img=None):
        self.img = img

    def color_histogram(self, color_space):
        pass

    def filter(self):
        pass

    def threshold(self):
        pass

    def rgb2hsi(self):
        # Normalizar RGB para [0,1]

        #  H = {
        #         teta se B <= G,
        #         360 - teta se B > G
        #       }
        #  teta = cos^-1{ [1/2 * ((R-G) + (R-B))] / [(R-G)^2 + (R-B)(G-B)]^1/2 }
        #  S = 1 - 3 / (R+G+B)
        #  I = 1/3 * (R+G+B)

        pass

    def hsi2rgb(self):
        pass

    @staticmethod
    def rgb2cmy(self, image):
        #  C = 1 - R
        #  M = 1 - G
        #  Y = 1 - B

        x, y = image.shape
        for i in range(x):
            for j in range(y):
                new_image[i][j] = 0

        return new_image

    def cmy2rgb(self):
        pass
