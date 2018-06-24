import numpy as np
import cv2 as cv


class Morphology:
    def __init__(self):
        pass

    def erosion(self, image, kernel):
        image = image.copy()

        image = convolve(image, kernel, 0, 255)

        return image

    def dilation(self, image, kernel):
        image = image.copy()

        image = convolve(image, kernel, 255, 0)

        return image

    def closing(self, image, kernel, opencv=False):
        if opencv:
            image = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel)
            return image

        image = self.dilation(image, kernel)
        image = self.erosion(image, kernel)

        return image

    def opening(self, image, kernel):
        image = self.erosion(image, kernel)
        image = self.dilation(image, kernel)

        return image

    def gradient(self, image, kernel):
        dilation = self.dilation(image, kernel)
        erosion = self.erosion(image, kernel)

        image = np.subtract(dilation, erosion)

        return image

    def top_hat(self, image, kernel):
        opening = self.opening(image, kernel)

        image = np.subtract(image, opening)

        return image

    def black_hat(self, image, kernel):
        closing = self.closing(image, kernel)

        image = np.subtract(closing, image)

        return image


def convolve(image, kernel, min, max):
    xlength, ylength = image.shape

    for i in range(xlength):
        for j in range(ylength):
            if image[i, j] == min:
                if i > 0 and image[i - 1, j] == max:
                    image[i-1, j] = 2
                if j > 0 and image[i, j - 1] == max:
                    image[i, j-1] = 2
                if i + 1 < xlength and image[i + 1, j] == max:
                    image[i+1, j] = 2
                if j + 1 < ylength and image[i, j + 1] == max:
                    image[i, j+1] = 2

    for i in range(xlength):
        for j in range(ylength):
            if image[i][j] == 2:
                image[i][j] = min

    return image
