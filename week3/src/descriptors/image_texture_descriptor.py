import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.fftpack import dct
from skimage.feature import local_binary_pattern

from src.consts import ColorSpace
from src.descriptors.base import Descriptor
from typing import List, Callable, Optional

from src.utils import zigzag
from src.metrics import DistanceType

class ImageTextureDescriptor(Descriptor):
    def __init__(
            self,
            image,
            colorspace: ColorSpace = ColorSpace.RGB,
            rows: int = 4,
            columns: int = 4,
            interval: int = 7,
            texture_method: str = 'dct'
        ):
        super().__init__(image, colorspace)
        self.channels = image.shape[2]
        self.rows = rows
        self.columns = columns
        self.interval = interval
        self.blocks = self.divide_image_into_blocks(rows, columns)
        self.values = self.compute_lbp_blocks(radius=2)

    def compute_texture_descriptor(self, method=str):
        if method=='lbp':
            return self.compute_lbp()
        elif method=='dct':
            return self.compute_dct()
        

    def compute_lbp_blocks(self, radius):
        """
        Computes the Local Binary Pattern (LBP) image for each channel of an RGB image using a vectorized approach.

        Parameters:
            image (numpy.ndarray): Input RGB image.
            radius (int): How far are the neighbours computed with the LBP

        Returns:
            lbp_image (numpy.ndarray): The resulting LBP image for the RGB channels.
        """
        blocks = self.blocks

        n_points = 8 * radius

        # Process each channel independently
        histograms_matrix = [[None for _ in range(self.columns)] for _ in range(self.rows)]

        for row in range(self.rows):
            for col in range(self.columns):
                histograms = []
                channels = cv2.split(blocks[row][col])
                fig, axs = plt.subplots(1, len(channels), figsize=(15, 5))  # 1 fila y len(channels) columnas

                # Iterar sobre cada canal y calcular el LBP
                for i, channel in enumerate(channels):
                    lbp_block = local_binary_pattern(channel, n_points, radius, method='uniform')
                    
                    # Mostrar el LBP en el subplot correspondiente
                    axs[i].imshow(lbp_block, cmap='gray')
                    axs[i].axis('off')
                    axs[i].set_title(f"LBP Image - Channel {i+1}")

                    hist, _ = np.histogram(lbp_block, bins=np.arange(0, 256, self.interval), density=True)
                    histograms.append(hist)
            
                # Mostrar la figura con todos los subplots
                plt.tight_layout()
                plt.show()

                # Concatenate all histograms for the current block
                concatenated_hist = np.concatenate(histograms)
                histograms_matrix[row][col] = concatenated_hist

        return histograms_matrix
    

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
    

    @staticmethod
    def _divide_image_into_blocks(rows: int, columns: int, image):
        """
        Divides the image into a grid of blocks based on the number of rows and columns.

        Parameters:
            rows (int): The number of rows to divide the image into.
            columns (int): The number of columns to divide the image into.

        Returns:
            blocks (list of lists): A 2D list (matrix) where each element is a block (sub-image)
            of the original image.
        """
        # Get image dimensions
        height, width = image.shape[:2]
        # Compute size of each block
        block_height = height // rows
        block_width = width // columns

        # Init a 2D list (matrix) to store the image blocks
        blocks = [[None for _ in range(columns)] for _ in range(rows)]

        # Loop thorugh each block position
        for i in range(rows):
            for j in range(columns):

                image_block = image[(i*block_height):(i*block_height) + block_height,
                                        (j*block_width):(j*block_width) + block_width]

                blocks[i][j] = image_block

        return blocks


    def divide_image_into_blocks(self, rows: int, columns: int):
        """
        Divides the image into a grid of blocks based on the number of rows and columns.

        Parameters:
            rows (int): The number of rows to divide the image into.
            columns (int): The number of columns to divide the image into.

        Returns:
            blocks (list of lists): A 2D list (matrix) where each element is a block (sub-image)
            of the original image.
        """
        image_blocks = self._divide_image_into_blocks(rows, columns, self.image)
        return image_blocks

    def plot_image_blocks(self):
        fig, axes = plt.subplots(self.rows, self.columns, figsize=(self.rows-1, self.columns-1))

        for i in range(self.rows):
            for j in range(self.columns):
                axes[i, j].imshow(self.blocks[i][j])
                axes[i, j].axis('off')

        plt.tight_layout()
        plt.show()

    
    def _compute_similarity_or_distance(self, descriptor2: 'ImageTextureDescriptor', func: Callable):
        # Potser hauriem de crear un descriptor per el DCT i un altre pel LBP
        # We should cut to N descriptors ?
        return func(self.values, descriptor2.values)
        
        
        