import cv2
from skimage import feature
from scipy.fftpack import dct

def LBP():
    return

class DCT():
    
    def zigzag():
        return

    def dct_hist(image, norm):
        """
        """
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Compute Discrete Fourier Transform
        image_dct = dct(dct(gray_image, axis=0, norm=norm), axis=1, norm=norm)

        return

def wavelet():
    return