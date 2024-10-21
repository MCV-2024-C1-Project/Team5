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

class ImageLBPDescriptor(Descriptor):
    def __init__(
            self,
            image,
            colorspace: ColorSpace = ColorSpace.RGB,
            rows: int = 4,
            columns: int = 4,
            channels: List = [[0], [1], [2]],
            radius: int = 2
        ):
        super().__init__(image, colorspace)
        self.channels = image.shape[2]
        self.rows = rows
        self.columns = columns
        self.channels = channels
        self.radius = radius
        self.blocks = self.divide_image_into_blocks(rows, columns)
        self.values = self.compute_lbp_blocks()

    def compute_lbp_blocks(self):
        """
        Computes the Local Binary Pattern (LBP) image for each channel of an RGB image using a vectorized approach.

        Parameters:
            radius (int): How far are the neighbors computed with the LBP.

        Returns:
            histograms_matrix (list): A matrix containing concatenated histograms for each block.
        """
        blocks = self.blocks
        n_points = 8 * self.radius

        # Process each block independently
        histograms_matrix = [[None for _ in range(self.columns)] for _ in range(self.rows)]

        for row in range(self.rows):
            for col in range(self.columns):
                channels = cv2.split(blocks[row][col])

                # Iterar sobre cada canal y calcular el LBP
                lbp_image = np.zeros_like(blocks[row][col], dtype=np.uint8)
                # fig, axs = plt.subplots(1, len(channels), figsize=(15, 5))  # 1 fila y len(channels) columnas
                for i, channel in enumerate(channels):
                    lbp_block = local_binary_pattern(channel, n_points, self.radius, method='uniform')
                    lbp_image[:, :, i] = lbp_block

                    # Mostrar el LBP en el subplot correspondiente
                    # axs[i].imshow(lbp_block, cmap='gray')
                    # axs[i].axis('off')
                    # axs[i].set_title(f"LBP Image - Channel {i+1}")
                
                
                histograms = []
                for channel_group in self.channels:
                    # Compute histogram
                    hist = cv2.calcHist([lbp_image], channel_group, None,
                                        [n_points + 1] * len(channel_group),
                                        [0, n_points + 1] * len(channel_group))
                    hist = self.normalize(hist).flatten()  # Normalize and flatten
                    histograms.append(hist)

                # Concatenate all histograms for the current block
                concatenated_hist = np.concatenate(histograms)
                histograms_matrix[row][col] = concatenated_hist

        return histograms_matrix
    

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

    
    def _compute_similarity_or_distance(self, descriptor2: 'ImageLBPDescriptor', func: Callable):
        result = []
        for i, _ in enumerate(self.values):
            if isinstance(self.values, list):
                for j, _ in enumerate(self.values[0]):
                # Compute distance/similarity for sub-elements in the list
                    result.append(
                        func(self.values[i][j], descriptor2.values[i][j])
                    )
            else:
                result.append(
                    func(self.values[i], descriptor2.values[i])
                )
        return result
        