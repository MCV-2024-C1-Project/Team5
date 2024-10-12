import numpy as np
import matplotlib.pyplot as plt

from src.consts import ColorSpace
from src.descriptors.base import Descriptor
from typing import List, Callable, Optional


class ImageBlockDescriptor(Descriptor):
    def __init__(
            self,
            image,
            colorspace: ColorSpace = ColorSpace.RGB,
            interval: int = 1,
            rows: int = 4,
            columns: int = 4
        ):
        super().__init__(image, colorspace)
        self.interval = interval
        self.rows = rows
        self.columns = columns
        self.blocks = self.divide_image_into_blocks(rows, columns)
        self.compute_image_histogram_descriptor(interval, rows, columns)

    def compute_image_histogram_descriptor(self, interval: int, rows: int = None, columns: int = None):
        """
        Compute the image's histogram descriptor by diving it into blocks and calculating the RGB histograms
        for each block, normalizing and concatenating them into a descriptor matrix.

        Parameters:
            interval (int): The bin width for histogram computation.
            rows (int): The number of rows to divide the image into.
            columns (int): The number of columns to divide the image into.

        Returns:
            Updates self.values with a 2D matrix containing concatenated
            histograms for each block. 
        """
        self.interval = interval or self.interval

        blocks = self.divide_image_into_blocks(rows, columns) # Changes self.rows and colums
        histograms_matrix = [[None for _ in range(self.columns)] for _ in range(self.rows)]
        
        for i in range(self.rows):
            for j in range(self.columns):
                block = blocks[i][j]

                # Compute histograms for each channel
                hist_r, _ = np.histogram(block[:, :, 0], bins=np.arange(0, 256, self.interval))
                hist_g, _ = np.histogram(block[:, :, 1], bins=np.arange(0, 256, self.interval))
                hist_b, _ = np.histogram(block[:, :, 2], bins=np.arange(0, 256, self.interval))

                # Flatten and normalize histograms
                hist_r = self.normalize(hist_r.flatten())
                hist_g = self.normalize(hist_g.flatten())
                hist_b = self.normalize(hist_b.flatten())

                # Concatenate R, G and B histograms
                concatenated_hist = np.concatenate([hist_r, hist_b, hist_g])

                # Save the normalized histograms
                histograms_matrix[i][j] = concatenated_hist

        self.values = histograms_matrix
        return histograms_matrix

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
        # Get image dimensions
        height, width = self.image.shape[:2]

        # Compute size of each block
        block_height = height // rows
        block_width = width // columns

        # Init a 2D list (matrix) to store the image blocks
        blocks = [[None for _ in range(columns)] for _ in range(rows)]
        
        # Loop thorugh each block position
        for i in range(rows):
            for j in range(columns):

                image_block = self.image[(i*block_height):(i*block_height) + block_height, 
                                        (j*block_width):(j*block_width) + block_width]

                blocks[i][j] = image_block

        self.rows = rows
        self.columns = columns
        return blocks
    
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
        assert self.interval == descriptor2.interval

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