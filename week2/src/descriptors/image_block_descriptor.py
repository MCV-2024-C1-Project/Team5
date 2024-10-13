import numpy as np
import cv2
import matplotlib.pyplot as plt

from src.consts import ColorSpace
from src.descriptors.base import Descriptor
from typing import List, Callable, Optional


class ImageBlockDescriptor(Descriptor):
    def __init__(
            self,
            image,
            colorspace: ColorSpace = ColorSpace.RGB,
            intervals: list = [7],
            rows: int = 4,
            columns: int = 4,
            channels: list = [[0, 1, 2]]
        ):
        super().__init__(image, colorspace)
        self.intervals = intervals
        self.rows = rows
        self.columns = columns
        self.channels = channels
        self.blocks = self.divide_image_into_blocks(rows, columns)
        self.compute_image_histogram_descriptor(intervals, rows, columns, channels)

    def compute_image_histogram_descriptor(self, intervals: list = None, rows: int = None, columns: int = None, channels: list = None, mask = None):
        """
        Compute the image's histogram descriptor by dividing it into blocks and calculating
        the histograms for specified channel combinations in each block.

        Parameters:
            interval (int): The bin width for histogram computation.
            rows (int): The number of rows to divide the image into.
            columns (int): The number of columns to divide the image into.
            channels (list of lists): Each inner list contains channels to include in the histogram.

        Returns:
            Updates self.values with a 2D matrix containing concatenated histograms for each block.
        """
        self.intervals = intervals or self.intervals
        self.channels = channels or self.channels

        blocks = self.divide_image_into_blocks(rows, columns)  # Changes self.rows and columns
        mask_blocks = self._divide_image_into_blocks(rows, columns, mask) if mask is not None else None
        histograms_matrix = [[None for _ in range(self.columns)] for _ in range(self.rows)]

        for i in range(self.rows):
            for j in range(self.columns):
                block = blocks[i][j]
                if mask is None:
                    mask_block = np.ones(block.shape[:2], dtype=np.uint8) * 255
                else:
                    mask_block = mask_blocks[i][j]

                histograms = []
                for interval, channel_group in zip(self.intervals, self.channels):
                    # Compute the histogram for the specified channel group
                    hist = cv2.calcHist([block], channel_group, mask_block,
                                        [256 // interval] * len(channel_group),
                                        [0, 256] * len(channel_group))
                    hist = self.normalize(hist).flatten()  # Normalize and flatten
                    histograms.append(hist)

                # Concatenate all histograms for the current block
                concatenated_hist = np.concatenate(histograms)
                histograms_matrix[i][j] = concatenated_hist

        self.values = histograms_matrix
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


    def _compute_similarity_or_distance(self, descriptor2, func):
        return super()._compute_similarity_or_distance(descriptor2, func)


    def _compute_similarity_or_distance(self, descriptor2: 'ImageBlockDescriptor', func: Callable) -> List[float]:
        result = []
        assert self.intervals == descriptor2.intervals
        assert self.channels == descriptor2.channels

        for i, _ in enumerate(self.values):
            if isinstance(self.values, list):
                for j, _ in enumerate(self.values):
                # Compute distance/similarity for sub-elements in the list
                    result.append(
                        func(self.values[i][j], descriptor2.values[i][j])
                    )
            else:
                result.append(
                    func(self.values[i], descriptor2.values[i])
                )
        return result

    def __getitem__(self, i: int, j: Optional[int] = None):
        if j:
            return self.values[i][j]
        return self.values[i]

    def __len__(self):
        return len(self.values)