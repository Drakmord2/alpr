import numpy as np
import os
import cv2


class Noise:
    def __init__(self, type, image):
        self.type = type
        self.image = image

    def apply(self):
        type = {
            "salt-pepper": self.salt_pepper_noise,
            "gaussian": self.gaussian_noise
        }

        noisy_image = type.get(self.type)()

        return noisy_image

    def salt_pepper_noise(self):
        s_vs_p = 0.5
        amount = 0.03
        out = np.copy(self.image)

        num_salt = np.ceil(amount * self.image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in self.image.shape]
        out[coords] = 255

        num_pepper = np.ceil(amount * self.image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in self.image.shape]
        out[coords] = 0

        return out

    def gaussian_noise(self):
        row, col = self.image.shape
        mean = 0
        var = 0.2
        sigma = var ** 0.01

        gauss = np.random.normal(mean, sigma, (row, col))
        gauss = gauss.reshape(row, col)

        noisy = self.image + gauss

        return noisy
