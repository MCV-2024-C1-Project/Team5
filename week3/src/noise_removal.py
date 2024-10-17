import cv2
import numpy as np
from math import log10, sqrt

def PSNR(original, denoised):
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) between the original image
    and the denoised image.

    Parameters:
        original (array): The original image.
        denoised (array): The denoised image.
    Returns:
        float: the PSNR value in decibles (dB). Higher value indicates less noise.
            Returns infinity if the MSE is zero.
    """
    mse = np.mean((original - denoised)**2)
    # Images are identical
    if mse == 0:
        return float('inf')
    
    psnr = 20 * log10(original.max() / sqrt(mse + 1e-7))
    return psnr
