import cv2
import numpy as np
import scipy.fftpack


class HomomorphicFilter:
    def __init__(self, img):
        self.img = img

    def filter(self):
        rows = self.img.shape[0]
        cols = self.img.shape[1]

        # Image normalization
        normalized = np.array(self.img, dtype="float") / 255

        # Apply log(1 + I)
        log = np.log1p(normalized)

        # Create Gaussian mask
        sigma = 15
        M = 2*rows + 1
        N = 2*cols + 1
        (X, Y) = np.meshgrid(np.linspace(0, N-1, N), np.linspace(0, M-1, M))
        center_x = np.ceil(N/2)
        center_y = np.ceil(M/2)
        gaussian_numerator = (X - center_x)**2 + (Y - center_y)**2

        # Filters
        h_low = np.exp(- gaussian_numerator / (2 * sigma**2))
        h_high = 1 - h_low

        # Move the origin of filters
        h_low_shift = scipy.fftpack.ifftshift(h_low.copy())
        h_high_shift = scipy.fftpack.ifftshift(h_high.copy())

        # Fourier Transform
        i_freq = scipy.fftpack.fft2(log.copy(), (M, N))

        # Filtering and Inverse Fourier Transform
        low_pass = i_freq.copy() * h_low_shift
        out_low = scipy.real(scipy.fftpack.ifft2(low_pass, (M, N)))

        high_pass = i_freq.copy() * h_high_shift
        out_high = scipy.real(scipy.fftpack.ifft2(high_pass, (M, N)))

        # High-frequency emphasis filter
        alfa = 0.5
        beta = 1.5
        out = (alfa * out_low[0:rows, 0:cols]) + (beta * out_high[0:rows, 0:cols])

        # Apply exp(I) - 1
        hmf = np.expm1(out)

        # Image denormalization
        hmf = (hmf - np.min(hmf)) / (np.max(hmf) - np.min(hmf))
        result = np.array(hmf * 255, dtype="uint8")

        return result
