import cv2
import numpy as np
from math import log10, sqrt
from skimage.metrics import structural_similarity as ssim

def PSNR(original, denoised):
    """
    Calculates the Peak Signal-to-Noise Ratio (PSNR) between the original image
    and the denoised image.

    Parameters:
        original (array): The original image.
        denoised (array): The denoised image.
    Returns:
        float: PSNR value in decibles (dB). A higher value indicates less noise.
            Returns infinity if the MSE is zero.
    """
    mse = np.mean((original - denoised)**2)
    # Images are identical
    if mse == 0:
        return float('inf')
    
    psnr = 20 * log10(original.max() / sqrt(mse + 1e-7))
    return psnr

def SSIM(original, denoised):
    """
    Calculates the Structural Similarity Index (SSIM) between the original image
    and the denoised image.

    Parameters:
        original (array): The original image.
        denoised (array): The denoised image.
    Returns:
        float: SSIM value. A higher value indicate less noise.
    """
    if len(original.shape) == 3:
        original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    if len(denoised.shape) == 3:
        denoised = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    
    ssim_value,_ = ssim(original, denoised, full=True)
    return ssim_value

def denoise_image(image, psnr_threshold: float = None, ssim_threshold: float = None):
    """
    Apply median filtering to the original image taking into account psnr and ssim thresholds.

    Parameters:
        image (array): The original image.
        psnr_threshold (float): PSNR threshold value. A higher threshold shows more similarity.
        ssim_threshold (float): SSIM threshold value. A higher threshold shows more similarity.
    Return:
        array: The denoised image if noise was detected.
    """
    # Apply median filtering to remove noise
    median_filtering = cv2.medianBlur(image, 3) # 3x3 kernel size

    assert psnr_threshold is not None or ssim_threshold is not None, 'At least one threshold must be provided'
    assert psnr_threshold is None or ssim_threshold is None, 'Only one threshold can be provided'

    if psnr_threshold is not None:
        psnr_value = PSNR(image, median_filtering)
        if psnr_value < psnr_threshold:
            return median_filtering
        return image
    
    ssim_value = SSIM(image, median_filtering)
    if ssim_value < ssim_threshold:
        return median_filtering
    return image