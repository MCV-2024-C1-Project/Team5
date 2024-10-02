from typing import List
import numpy as np
from enum import Enum


# Distance metrics
def euclidean_distance(hist1: List[np.array], hist2: List[np.array]) -> float:
    """
    Euclidean distance is the straight-line distance between two points in Euclidean space.

    Parameters:
    hist1 (List[np.array]): The first histogram.
    hist2 (List[np.array]): The second histogram.

    Returns:
    float: The Euclidean distance between the two histograms.
    """
        
    return np.linalg.norm(hist1 - hist2)

def l1_distance(hist1: List[np.array], hist2: List[np.array]) -> float:
    """
    L1 distance is the sum of the absolute differences between corresponding bins
    in the histograms. It is less sensitive to large differences than Euclidean distance.

    Parameters:
    hist1 (List[np.array]): The first histogram.
    hist2 (List[np.array]): The second histogram.

    Returns:
    float: The L1 (Manhattan) distance.
    """

    return np.sum(np.abs(hist1 - hist2))

def chi2_distance(hist1: List[np.array], hist2: List[np.array]) -> float:
    """
    Chi-squared distance is useful to measure similarity between 2 feature matrices.

    Parameters:
    hist1 (List[np.array]): The first histogram.
    hist2 (List[np.array]): The second histogram.

    Returns:
    float: The Chi-squared distance between the two histograms.
    """

    return 0.5 * np.sum(np.square(hist1 - hist2) / (hist1 + hist2 + 1e-10))

class DistanceType(Enum):
    euclidean = euclidean_distance
    l1 = l1_distance
    chi2 = chi2_distance


# Similarity metrics
def hellinger_kernel_similarity(hist1: List[np.array], hist2: List[np.array]) -> float:
    """
    Hellinger kernel to quantify the similarity between two probability distributions (or histograms).

    Parameters:
    hist1 (List[np.array]): The first histogram.
    hist2 (List[np.array]): The second histogram.

    Returns:
    float: The Hellinger kernel similarity.
    """
    
    return np.sum(np.sqrt(hist1 * hist2))

def histogram_intersection_similarity(hist1: List[np.array], hist2: List[np.array]) -> float:
    """
    Computes the histogram intersection similarity between two histograms.

    Parameters:
    hist1 (List[np.array]): The first histogram.
    hist2 (List[np.array]): The second histogram.

    Returns:
    float: The histogram intersection similarity (between 0 and 1).
    """
    
    return np.sum(np.minimum(hist1, hist2))

class SimilarityType(Enum):
    hellinger_kernel = hellinger_kernel_similarity
    histogram_intersection = histogram_intersection_similarity
