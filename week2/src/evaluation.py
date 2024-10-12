from typing import Dict, List
import numpy as np

from src.metrics import DistanceType, SimilarityType, apk
from src.image_dataset import ImageDataset

def compute_results_by_distance(bdd_dataset: ImageDataset,
                                query_dataset: ImageDataset,
                                ground_truth: List[List[int]],
                                distance_type: DistanceType = 
                                DistanceType.l1) -> Dict:
    """
    Compute the similarity results between a query dataset and a background dataset 
    using a specified similarity metric.

    Parameters:
        bdd_dataset (ImageDataset): The dataset containing images to compare against.
        query_dataset (ImageDataset): The dataset containing images to query.
        ground_truth (List[List[int]]): A list of ground truth indices for each query image.
        similarity_type (SimilarityType): The type of similarity metric to use for comparison.
        intersection).

    Returns:
        Dict: A dictionary containing:
            - 'apk1': A list of average precision at k=1 for each query image.
            - 'apk5': A list of average precision at k=5 for each query image.
            - 'similarities': A list of lists, where each sublist contains the similarity scores 
            for the top K images corresponding to each query image.
    """
    result = []
    distances_result = []
    for image in query_dataset:
        distances_list = []
        for image2 in bdd_dataset:
            distances = image.compute_distance(image2, type=distance_type)
            distance = np.mean(distances)
            distances_list.append(distance)
        top_k = np.argsort(distances_list)[:5]

        result.append([bdd_dataset[i].index for i in top_k])
        distances_result.append([distances_list[i] for i in top_k])

    return {
        'apk1': [apk(a,p,1) for a,p in zip(ground_truth, result)],
        'apk5': [apk(a,p,5) for a,p in zip(ground_truth, result)],
        'distances': distances_result
    }

def compute_results_by_similarity(bdd_dataset: ImageDataset, 
                                  query_dataset: ImageDataset, 
                                  ground_truth: List[List[int]], 
                                  similarity_type: SimilarityType = 
                                  SimilarityType.histogram_intersection) -> Dict:
    """
    Compute the similarity results between a query dataset and a background dataset 
    using a specified similarity metric.

    Parameters:
        bdd_dataset (ImageDataset): The dataset containing images to compare against.
        query_dataset (ImageDataset): The dataset containing images to query.
        ground_truth (List[List[int]]): A list of ground truth indices for each query image.
        similarity_type (SimilarityType): The type of similarity metric to use for comparison.
        intersection).

    Returns:
        Dict: A dictionary containing:
            - 'apk1': A list of average precision at k=1 for each query image.
            - 'apk5': A list of average precision at k=5 for each query image.
            - 'similarities': A list of lists, where each sublist contains the similarity scores 
            for the top K images corresponding to each query image.
    """
    result = []
    similarities_result = []
    for image in query_dataset:
        similarities_list = []
        for image2 in bdd_dataset:
            similarities = image.compute_similarity(image2, type=similarity_type)
            similarity = np.mean(similarities)
            similarities_list.append(similarity)
        top_k = np.argsort(similarities_list)[-5:][::-1]

        result.append([bdd_dataset[i].index for i in top_k])
        similarities_result.append([similarities_list[i] for i in top_k])
   
    return {
        'apk1': [apk(a,p,1) for a,p in zip(ground_truth, result)],
        'apk5': [apk(a,p,5) for a,p in zip(ground_truth, result)],
        'similarities': similarities_result
    }


def compute_spatial_pyramid(bdd_dataset: ImageDataset, 
                            query_dataset: ImageDataset, 
                            ground_truth: List[List[int]], 
                            similarity_type: SimilarityType = 
                            SimilarityType.histogram_intersection) -> Dict:
    """
    Compute the spatial pyramid at a certain level  between a query dataset and a background dataset 
    using a specified similarity metric.

    For each level, compute de topk images and append it to a list.
    In the end, count for the topk images with the highest votes among all levels.

    Parameters:
        bdd_dataset (ImageDataset): The dataset containing images to compare against.
        query_dataset (ImageDataset): The dataset containing images to query.
        ground_truth (List[List[int]]): A list of ground truth indices for each query image.
        similarity_type (SimilarityType): The type of similarity metric to use for comparison.
        intersection).
        levels (List[int]): Contains the levels we want to use. The number indicates the size of the grid we will us to divide the image by blocks.

    Returns:
        Dict: A dictionary containing:
            - 'apk1': A list of average precision at k=1 for each query image.
            - 'apk5': A list of average precision at k=5 for each query image.
            - 'similarities': A list of lists, where each sublist contains the similarity scores 
            for the top K images corresponding to each query image.
    """
    result = []
    similarities_result = []

    for image in query_dataset:
        similarities_list = []
        for image2 in bdd_dataset:
            similarities = image.compute_similarity(image2, type=similarity_type)
            similarity = np.mean(similarities)
            similarities_list.append(similarity)
        top_k = np.argsort(similarities_list)[-5:][::-1]

        result.append([bdd_dataset[i].index for i in top_k])
        similarities_result.append([similarities_list[i] for i in top_k])

    return {
        'apk1': [apk(a,p,1) for a,p in zip(ground_truth, result)],
        'apk5': [apk(a,p,5) for a,p in zip(ground_truth, result)],
        'similarities': similarities_result
    }