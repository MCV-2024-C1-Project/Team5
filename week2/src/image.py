from enum import Enum
from typing import Optional, List, Callable
import cv2
import matplotlib.pyplot as plt
import numpy as np
import platform
from src.metrics import DistanceType, SimilarityType, apk

class ColorSpace(Enum):
    gray = cv2.COLOR_BGR2GRAY
    RGB = cv2.COLOR_BGR2RGB
    HSV = cv2.COLOR_BGR2HSV
    CieLab = cv2.COLOR_BGR2Lab
    YCbCr = cv2.COLOR_BGR2YCrCb


class Image:
    def __init__(self, path: str, colorspace: ColorSpace = ColorSpace.RGB, interval: int = 1):
        self.path = path
        self.index = self._extract_index(path)
        self.original_image = cv2.imread(path)
        self.image = cv2.cvtColor(self.original_image, colorspace.value)
        self.colorspace = colorspace
        self.interval = None
        self.histogram_descriptor = []
        self.compute_image_histogram_descriptor(interval)


    def change_colorspace(self, new_colorspace: ColorSpace):
        self.image = cv2.cvtColor(self.original_image, new_colorspace.value)
        self.colorspace = new_colorspace

    def _extract_index(self, file_path):
        os_name = platform.system()

        if os_name == 'Windows':
            file_name = file_path.split('\\')[-1]
        else:
            file_name = file_path.split('/')[-1]

        name = file_name.split('.')[0]
        number = name.split('_')[-1]
        return int(number) 


    def compute_image_histogram_descriptor(self, interval: int):
        self.interval = interval
        # Separate the channels
        channels = cv2.split(self.image)

        # Create histogram
        histograms = []
        for channel in channels:
            # Compute histogram
            hist, _ = np.histogram(channel, bins=np.arange(0, 256, interval))  # Intervals of histogram given by bin_size
            
           # Normalize and flatten histograms
            hist = hist.flatten()
            hist = normalize(hist)
            histograms.append(hist)

        self.histogram_descriptor = histograms

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

    def compute_histogram_block_descriptor(self, interval: int, blocks, rows: int, columns: int):
        """
        """

        if interval is None:
            interval = self.interval
        else:
            self.interval = interval

        histograms_matrix = [[None for _ in range(columns)] for _ in range(rows)]

        for i in range(rows):
            for j in range(columns):
                block = blocks[i][j]

                # Compute histograms for each channel
                hist_r, _ = np.histogram(block[:, :, 0], bins=np.arange(0, 256, interval))
                hist_g, _ = np.histogram(block[:, :, 1], bins=np.arange(0, 256, interval))
                hist_b, _ = np.histogram(block[:, :, 2], bins=np.arange(0, 256, interval))

                # Flatten and normalize histograms
                hist_r = normalize(hist_r.flatten())
                hist_g = normalize(hist_g.flatten())
                hist_b = normalize(hist_b.flatten())

                # Concatenate R, G and B histograms
                concatenated_hist = np.concatenate([hist_r, hist_b, hist_g])

                # Save the normalized histograms
                histograms_matrix[i][j] = concatenated_hist
        
        # Save the matrix as histogram descriptor
        self.histogram_descriptor = histograms_matrix

    def plot_histograms(self, savepath: Optional[str] = None):
        channel_names = self.get_channel_names()

        fig, axs = plt.subplots(1, len(self.histogram_descriptor), figsize=(15, 5), sharey=True)

        max_freq = 0  # Max value between all hitograms
        colors = ['red', 'lime', 'blue']
        # TODO: It doesn't work with gray scale. Need to fix it
        for i, hist in enumerate(self.histogram_descriptor):
            axs[i].bar(range(len(hist)), hist, width=0.5, color=colors[i], alpha=0.7)
            axs[i].set_title(f'{channel_names[i]}')
            axs[i].set_xlabel('Bin number')
            axs[i].set_ylabel('Probability')
            axs[i].set_xlim(-1, len(hist))
            
            max_freq = max(max_freq, max(hist))*1.02
            axs[i].set_ylim(0, max_freq) 
            axs[i].grid(False)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        title = f'Color descriptors for each channel in {self.colorspace.name} colorspace (bin size={self.interval})'
        fig.suptitle(title, y=1.02, fontsize=15)

        # Save plot if savepath is provided
        if savepath:
            plt.savefig(f"{savepath}/{channel_names[i]}_{self.colorspace.name}_histogram.png")

        plt.show()

    def get_channel_names(self):
        # Asociate the colorspace to the names of their channels
        colorspace_dict = {
            'gray': ['Intensity'],
            'RGB': ['R', 'G', 'B'],
            'HSV': ['H', 'S', 'V'],
            'CieLab': ['L', 'a', 'b'],
            'YCbCr': ['Y', 'Cb', 'Cr']
        }
        return colorspace_dict[self.colorspace.name]

    def show(self):
        """
        Shows the image in the actual colorspace you're working with.
        """
        plt.imshow(self.image)
        plt.title(f'{self.colorspace} Image')
        plt.axis('off')
        plt.show()

    def show_original(self):
        """
        Shows the image in the RGB colorspace, just as they are stored originally in the database.
        """
        bgr_image = cv2.imread(self.path)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        plt.imshow(rgb_image)
        plt.title('RGB Image')
        plt.axis('off')
        plt.show()
    

    def _compute_similarity_or_distance(self, image2: 'Image', func: Callable) -> List[float]:
        result = []
        
        # Assert they have comparable histograms
        assert self.colorspace.name == image2.colorspace.name
        assert self.interval == image2.interval
        
        for i, _ in enumerate(self.histogram_descriptor):

            # Compute distance/similarity
            result.append(
                func(self.histogram_descriptor[i], image2.histogram_descriptor[i])
            )

        return result


    def compute_similarity(self, image2: 'Image', type=SimilarityType):
        return self._compute_similarity_or_distance(image2, type)


    def compute_distance(self, image2: 'Image', type=DistanceType) -> List[float]:
        return self._compute_similarity_or_distance(image2, type)
    

# Normalization of a histogram
def normalize(hist: List[np.array]) -> List[np.array]:
    """
    Normalizes the given histogram. The sum of all bins in the returned histogram is 1.

    Parameters:
    hist (List[np.array]): Histogram.

    Returns:
    hist (List[np.array]): Normalized histogram.
    """
    
    return hist / np.sum(hist)