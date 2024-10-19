import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.fftpack import dct

from src.consts import ColorSpace
from src.descriptors.base import Descriptor
from typing import List, Callable, Optional

from src.utils import zigzag


class ImageTextureDescriptor(Descriptor):
    def __init__(
            self,
            image,
            colorspace: ColorSpace = ColorSpace.RGB,
            intervals: int = 7,
            rows: int = 4,
            columns: int = 4,
            channels: list = [[0, 1, 2]],
        ):
        super().__init__(image, colorspace)
        self.intervals = intervals
        self.rows = rows
        self.columns = columns
        self.channels = channels
        self.blocks = 0 # self.divide_image_into_blocks(rows, columns)
        self.values = self.compute_texture_descriptor('dct')

    def compute_texture_descriptor(self, method=str):
        if method=='lbp':
            return self.compute_lbp()
        elif method=='dct':
            return self.compute_dct()

    def compute_lbp(self):
        """
        Computes the Local Binary Pattern (LBP) image for each channel of an RGB image using a vectorized approach.

        Parameters:
            image (numpy.ndarray): Input RGB image.

        Returns:
            lbp_image_rgb (numpy.ndarray): The resulting LBP image for the RGB channels.
        """
        image = self.image

        # Initialize the LBP image for each channel
        lbp_image = np.zeros_like(image, dtype=np.uint8)

        # Define the 8 neighbors relative to the center pixel
        neighbors = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, 1), (1, 1), (1, 0),
            (1, -1), (0, -1)
        ]

        # Process each channel independently
        num_channels = len(self.channels)
        for channel in range(num_channels):  # Loop over the three color channels
            # Get the current channel
            image_channel = image[:, :, channel]

            # Initialize the LBP image for the current channel
            lbp_channel = np.zeros_like(image_channel, dtype=np.uint8)

            # Loop over each of the 8 neighbors
            for idx, (dy, dx) in enumerate(neighbors):
                # Shift the image to get the neighbor pixel values
                shifted = np.roll(np.roll(image_channel, shift=dy, axis=0), shift=dx, axis=1)

                # Compare neighbor pixels to the center pixel and update LBP image
                lbp_channel += ((shifted >= image_channel) << idx).astype(np.uint8)

            # Set border pixels to zero (they cannot be computed accurately)
            lbp_channel[0, :] = 0
            lbp_channel[-1, :] = 0
            lbp_channel[:, 0] = 0
            lbp_channel[:, -1] = 0

            # Store the LBP image for the current channel
            lbp_image[:, :, channel] = lbp_channel

        return lbp_image
    

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