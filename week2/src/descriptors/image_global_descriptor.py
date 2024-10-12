from src.descriptors.base import Descriptor
from src.consts import ColorSpace

import cv2
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List, Callable

class ImageGlobalDescriptor(Descriptor):

    def __init__(
        self,
        image,
        colorspace: ColorSpace = ColorSpace.RGB,
        interval: int = 1,

    ):
        super().__init__(image, colorspace)
        self.colorspace = colorspace
        self.interval = interval
        self.compute_image_histogram_descriptor(interval)

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

        self.values = histograms
        return histograms

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
    

    def _compute_similarity_or_distance(self, descriptor2: 'ImageGlobalDescriptor', func: Callable) -> List[float]:
        result = []
        
        assert self.interval == descriptor2.interval
        
        for i, _ in enumerate(self.values):

            # Compute distance/similarity
            result.append(
                func(self.values[i], descriptor2.values[i])
            )

        return result


    def __getitem__(self, i: int):
        return self.values[i]

    def __len__(self):
        return len(self.values)