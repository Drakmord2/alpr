import cv2
import numpy as np
import scipy.fftpack


class HomomorphicFilter:
    def __init__(self, img):
        self.img = img

    def filter(self):
        rows = self.img.shape[0]
        cols = self.img.shape[1]

        # Convert image to 0 and 1, then do log(1 + I)
        log = np.log1p(np.array(self.img, dtype="float") / 255)

        # Create Gaussian mask of sigma = 10
        M = 2*rows + 1
        N = 2*cols + 1
        sigma = 10
        (X, Y) = np.meshgrid(np.linspace(0, N-1, N), np.linspace(0, M-1, M))
        center_x = np.ceil(N/2)
        center_y = np.ceil(M/2)
        gaussian_numerator = (X - center_x)**2 + (Y - center_y)**2

        # Low pass and high pass filters
        h_low = np.exp(- gaussian_numerator / (2*sigma*sigma))
        h_high = 1 - h_low

        # Move origin of filters so that it's at the top left corner to
        # match with the input image
        h_low_shift = scipy.fftpack.ifftshift(h_low.copy())
        h_high_shift = scipy.fftpack.ifftshift(h_high.copy())

        # Filter the image and crop
        func = scipy.fftpack.fft2(log.copy(), (M, N))
        out_low = scipy.real(scipy.fftpack.ifft2(func.copy() * h_low_shift, (M, N)))
        out_high = scipy.real(scipy.fftpack.ifft2(func.copy() * h_high_shift, (M, N)))

        # Set scaling factors and add
        gamma1 = 0.3
        gamma2 = 1.5
        out = gamma1*out_low[0:rows, 0:cols] + gamma2*out_high[0:rows, 0:cols]

        # Anti-log then rescale to [0,1]
        hmf = np.expm1(out)
        hmf = (hmf - np.min(hmf)) / (np.max(hmf) - np.min(hmf))
        hmf2 = np.array(255*hmf, dtype="uint8")

        return hmf2
