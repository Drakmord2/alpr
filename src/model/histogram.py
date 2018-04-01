import numpy as np 


class Histogram:
    @staticmethod
    def histogram(img):
        m, n = img.shape
        
        h = [0.0] * 256
        for i in range(m):
            for j in range(n):
                h[img[i, j]] += 1
                
        return np.array(h)/(m*n)

    @staticmethod
    def cum_sum(h):
        # finds cumulative sum of a numpy array, list
        return [sum(h[:i+1]) for i in range(len(h))]

    @staticmethod
    def equalize(img):
        h = Histogram.histogram(img)

        cdf = np.array(Histogram.cum_sum(h))  # cumulative distribution function
        sk = np.uint8(255 * cdf)  # finding transfer function values

        s1, s2 = img.shape
        Y = np.zeros_like(img)

        for i in range(0, s1):
            for j in range(0, s2):
                Y[i, j] = sk[img[i, j]]

        H = Histogram.histogram(Y)

        # return transformed image, original and new histogram,
        # and transform function
        return Y, h, H, sk
