from typing import List
import numpy as np
from enum import Enum
import cv2


# DISTANCE METRICS
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


# SIMILARITY METRICS
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

def bhattacharyya_similarity(hist1: List[np.array], hist2: List[np.array]) -> float:
    """
    Computes the Bhattacharyya similarity between two histograms.
    
    Parameters:
    hist1 (List[np.array]): The first histogram.
    hist2 (List[np.array]): The second histogram.
    
    Returns:
    float: The Bhattacharyya similarity (between 0 and 1).
    """
    hist1 = hist1.astype('float32')
    hist2 = hist2.astype('float32')

    distance = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)

    # Convert distance to similarity (lower distance means higher similarity)
    return 1 - distance

class SimilarityType(Enum):
    hellinger_kernel = hellinger_kernel_similarity
    histogram_intersection = histogram_intersection_similarity
    bhattacharyya = bhattacharyya_similarity


# EVALUATION METRICS
# Copied from https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py

def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.

    This function computes the average precision at k between two lists of
    items.

    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The average precision at k over the input lists

    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.

    This function computes the mean average prescision at k between two lists
    of lists of items.

    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The mean average precision at k over the input lists

    """
    result = []
    for act, pred in zip(actual, predicted):
        for i in range(len(pred)):
            print(act[i], pred[i])
            result.append(apk([act[i]], pred[i], k))
    return np.mean(result)