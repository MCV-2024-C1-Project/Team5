import numpy as np
import matplotlib.pyplot as plt
from typing import List

from src.consts import ColorSpace
from src.descriptors.base import Descriptor
from src.descriptors.image_block_descriptor import ImageBlockDescriptor


class ImagePyramidDescriptor(ImageBlockDescriptor):
    def __init__(
            self,
            image,
            colorspace: ColorSpace = ColorSpace.RGB,
            interval: int = 1,
            levels: int = 3
        ):
        Descriptor.__init__(self, image, colorspace)
        self.interval = interval
        self.num_levels = levels
        self.levels = []
        self.compute_image_histogram_descriptor(interval, levels)

    def plot_image_blocks(self, levels: List[int]):
        # TODO: Fix
        for i, level in enumerate(levels):
            blocks = self.levels[i]
            # ----- Plot the Image Blocks -----
            fig, axes = plt.subplots(level, level, figsize=(level, level))

            for i in range(level):
                for j in range(level):
                    axes[i, j].imshow(blocks[i][j])
                    axes[i, j].axis('off')

            plt.suptitle(f'Level {level}')
            plt.tight_layout(w_pad=0.1)
            plt.show()

    def compute_image_histogram_descriptor(self, interval: int, levels: List[int]):
        self.num_levels = levels or self.num_levels
        self.interval = interval or self.interval
        for level in self.num_levels:
            self.divide_image_into_blocks(rows=level, columns=level)
            histograms_matrix = super().compute_image_histogram_descriptor(interval=interval, rows=level, columns=level)
            self.levels.append(histograms_matrix)
