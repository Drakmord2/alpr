import numpy as np
import cv2 as cv
from util.csvUtil import CSVUtil


class KNN:
    def __init__(self):
        self.type = 'KNN'

    def train(self):
        pass

    def classify(self, hulog):
        csv = CSVUtil()
        digits_moments = csv.read('../templates/moments.csv')

        hulog = hulog.flatten()
        for digit in digits_moments:
            moments = [float(digit['u1']), float(digit['u2']), float(digit['u3']), float(digit['u4']),
                       float(digit['u5']), float(digit['u6']), float(digit['u7'])]

            close = np.isclose(hulog, moments)

            if np.any(close):
                return digit['digit']

        return False
