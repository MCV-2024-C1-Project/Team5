import numpy as np
import cv2

from src.descriptors.base import Descriptor
from src.consts import ColorSpace
from src.utils import zigzag

class ImageDCTDescriptor(Descriptor):
    def __init__(
            self,
            image,
            colorspace: ColorSpace = ColorSpace.RGB,
        ):
        super().__init__(image, colorspace)
        self.values = self.compute_dct()
    
    def compute_dct(self):
        """
        Computes the Discrete Cosine Transform (DCT) for each channel of an RGB image.

        Parameters:
            image (numpy.ndarray): Input RGB image.

        Returns:
            dct_image_rgb (numpy.ndarray): The resulting DCT image for the RGB channels.
        """
        image = self.image

        # Initialize the DCT image for each channel
        self.dct_image = np.zeros_like(image)

        num_channels = len(self.channels)
        descriptors = []
        for channel in range(num_channels):  # Loop over the three color channels
            # Get the current channel
            image_channel = image[:, :, channel]

            imf = np.float32(image_channel)/255.0  # float conversion/scale
            dct = cv2.dct(imf)              # the dct
            dct_channel = np.uint8(dct*255.0)    # convert back to int

            # Store the DCT image for the current channel
            self.dct_image[:, :, channel] = dct_channel
            descriptors.append(zigzag(dct_channel))

        return descriptors