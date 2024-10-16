from typing import Callable, List
import cv2

from src.consts import ColorSpace
import numpy as np

from src.metrics import DistanceType, SimilarityType

class Descriptor:
    values: List
    colorspace: ColorSpace

    def __init__(self, image, colorspace: ColorSpace):
        self.original_image = image
        self.to_colorspace(colorspace)

    def _compute_similarity_or_distance(self, descriptor2: 'Descriptor', func: Callable):
        raise NotImplementedError()

    def compute_distance(self, descriptor2: 'Descriptor', type=DistanceType):
        return self._compute_similarity_or_distance(descriptor2, type)

    def compute_similarity(self, descriptor2: 'Descriptor', type=SimilarityType):
        return self._compute_similarity_or_distance(descriptor2, type)
    
    def to_colorspace(self, colorspace: ColorSpace):
        self.colorspace = colorspace
        self.image = cv2.cvtColor(self.original_image, colorspace.value)

        # Normalization of a histogram
    def normalize(self, hist: List[np.array]) -> List[np.array]:
        """
        Normalizes the given histogram. The sum of all bins in the returned histogram is 1.

        Parameters:
        hist (List[np.array]): Histogram.

        Returns:
        hist (List[np.array]): Normalized histogram.
        """
        sum_ = np.sum(hist)
        if sum_ < 10**-12:
            return hist
        return hist / sum_