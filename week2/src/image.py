from enum import Enum
from typing import Optional, List, Callable
import cv2
import matplotlib.pyplot as plt
import numpy as np
import platform

from src.metrics import DistanceType, SimilarityType

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

    def _compute_similarity_or_distance(self, image2: 'Image', func: Callable) -> List[float]:
        result = []

        # Assert they have comparable histograms
        assert self.colorspace.name == image2.colorspace.name
        assert self.interval == image2.interval

        for i, _ in enumerate(self.histogram_descriptor):
            if isinstance(self.histogram_descriptor, list):
                for j, _ in enumerate(self.histogram_descriptor):
                # Compute distance/similarity for sub-elements in the list
                    result.append(
                        func(self.histogram_descriptor[i][j], image2.histogram_descriptor[i][j])
                    )
            else:
                result.append(
                    func(self.histogram_descriptor[i], image2.histogram_descriptor[i])
                )
        return result


    def compute_similarity(self, image2: 'Image', type=SimilarityType):
        return self._compute_similarity_or_distance(image2, type)

    def compute_distance(self, image2: 'Image', type=DistanceType) -> List[float]:
        return self._compute_similarity_or_distance(image2, type)

    def compute_image_histogram_descriptor(self, interval: int):
        if interval is None:
            interval = self.interval
        else:
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
            hist = self.normalize(hist)
            histograms.append(hist)

        self.histogram_descriptor = histograms

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

    # Normalization of a histogram
    def normalize(self, hist: List[np.array]) -> List[np.array]:
        """
        Normalizes the given histogram. The sum of all bins in the returned histogram is 1.

        Parameters:
        hist (List[np.array]): Histogram.

        Returns:
        hist (List[np.array]): Normalized histogram.
        """

        return hist / np.sum(hist)