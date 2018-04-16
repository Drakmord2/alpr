import numpy as np


class MedianFilter:
    def __init__(self, image, window, threshold=0):
        self.image = image
        self.window = window
        self.threshold = threshold

    def filter(self):
        w = 2 * self.window + 1
        xlength, ylength = self.image.shape
        vlength = w * w

        image_array = np.reshape(np.array(self.image, dtype=np.uint8), (xlength, ylength))

        for y in range(self.window, ylength - (self.window + 1)):
            for x in range(self.window, xlength - (self.window + 1)):

                filter_window = image_array[x - self.window:x + self.window + 1, y - self.window:y + self.window + 1]
                target_vector = np.reshape(filter_window, vlength)

                median = self.get_median(target_vector, vlength)

                image_array[x, y] = median

        image = np.reshape(image_array, (xlength, ylength))

        return image

    def adaptive_filter(self):
        w = 2 * self.window + 1
        xlength, ylength = self.image.shape
        vlength = w * w

        image_array = np.reshape(np.array(self.image, dtype=np.uint8), (xlength, ylength))

        for y in range(self.window, ylength - (self.window + 1)):
            for x in range(self.window, xlength - (self.window + 1)):

                filter_window = image_array[x - self.window:x + self.window + 1, y - self.window:y + self.window + 1]
                target_vector = np.reshape(filter_window, vlength)

                median = self.get_median(target_vector, vlength)

                if self.threshold <= 0:
                    image_array[x, y] = median
                else:
                    scale = np.zeros(vlength)
                    for n in range(vlength):
                        scale[n] = np.abs(int(target_vector[n]) - int(median))

                    scale = np.sort(scale)
                    index = vlength // 2
                    sk = 1.4826 * (scale[index])

                    if np.abs(int(image_array[x, y]) - int(median)) > (self.threshold * sk):
                        image_array[x, y] = median

        image = np.reshape(image_array, (xlength, ylength))

        return image

    def get_median(self, target_array, array_length):
        sorted_array = np.sort(target_array)
        index = array_length // 2

        median = sorted_array[index]

        return median
