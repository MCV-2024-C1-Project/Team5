import numpy as np

from src.image import Image, ColorSpace


class ImageBlockDescriptor(Image):
    def __init__(self, path: str, colorspace: ColorSpace = ColorSpace.RGB, interval: int = 1, rows: int = 4, columns: int = 4):
        super().__init__(path, colorspace, interval)
        self.rows = rows
        self.columns = columns

    def compute_image_histogram_descriptor(self, interval: int, rows: int = None, columns: int = None):
        self.interval = interval or self.interval
        self.rows = rows or self.rows
        self.columns = columns or self.columns

        histograms_matrix = [[None for _ in range(self.columns)] for _ in range(self.rows)]
        blocks = self.divide_img_into_blocks(self.rows, self.columns)

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

        self.histogram_descriptor = histograms_matrix

    def divide_img_into_blocks(self, rows: int, columns: int):
        """
        """
        # Get image dimensions
        height, width = self.image.shape[:2]

        # Compute size of each block
        block_height = height // rows
        block_width = width // columns

        # Divide la imagen en bloques
        blocks = [[None for _ in range(columns)] for _ in range(rows)]
        for i in range(rows):
            for j in range(columns):

                image_block = self.image[(i*block_height):(i*block_height) + block_height, 
                                        (j*block_width):(j*block_width) + block_width]

                blocks[i][j] = image_block

        return blocks