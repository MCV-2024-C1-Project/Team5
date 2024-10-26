import cv2
import numpy as np
from scipy.fftpack import dct
from typing import List, Callable

from src.noise_removal import denoise_image
from src.background_removal import get_mask_and_foreground
from src.descriptors.base import Descriptor
from src.consts import ColorSpace
from src.utils import zigzag

class ImageDCTDescriptor(Descriptor):
    def __init__(
            self,
            image,
            colorspace: ColorSpace = ColorSpace.RGB,
            N: int = 100,
            rows: int = 1,
            columns: int = 1,
            image_size: tuple = (64, 64)
        ):
        image = denoise_image(image)
        image = cv2.resize(image, image_size)
        super().__init__(image, colorspace)
        self.N = N
        self.rows = rows
        self.columns = columns
        self.values = self.compute_dct(N)

    def compute_dct(self, N):
        """
        Computes the Discrete Cosine Transform (DCT) for each block of the image.

        Parameters:
            N (int): Number of coefficients to return.

        Returns:
            descriptors (numpy.ndarray): The resulting DCT descriptors for the blocks.
        """
        # Initialize the descriptors array
        n_channels = 1 if len(self.image.shape) == 2 else self.image.shape[2]
        descriptors = np.zeros((self.rows, self.columns, n_channels*N))

        # Divide the image into blocks
        image_blocks = self.divide_image_into_blocks(self.rows, self.columns)

        for i in range(self.rows):
            for j in range(self.columns):
                norm_block = image_blocks[i][j] / 255.0
                dct_result = dct(dct(norm_block, type=2, norm='ortho', axis=0), type=2, norm='ortho', axis=1)
                limit = np.ceil(np.sqrt(N)).astype(int)
                if n_channels == 1:
                    descriptors[i, j, :] = zigzag(dct_result[:limit, :limit])[:N]
                else:
                    for k in range(n_channels):
                        descriptors[i, j, k*N:(k+1)*N] = zigzag(dct_result[:limit, :limit, k])[:N]

        self.N = N
        return descriptors

    def _compute_similarity_or_distance(self, descriptor2: 'ImageDCTDescriptor', func: Callable) -> List[float]:    
        assert self.colorspace == descriptor2.colorspace, "Colorspaces must be the same"
        assert self.N == descriptor2.N, "N must be the same"

        result = []
        for i in range(self.rows):
            for j in range(self.columns):
                result.append(func(self.values[i, j], descriptor2.values[i, j]))
        
        return result
    
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

        # Loop through each block position
        for i in range(rows):
            for j in range(columns):
                image_block = image[(i * block_height):(i * block_height) + block_height,
                                    (j * block_width):(j * block_width) + block_width]
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