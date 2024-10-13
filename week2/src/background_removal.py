import os
import cv2
import numpy as np
import re
import pandas as pd


def imreconstruct(marker: np.ndarray, mask: np.ndarray, radius: int = 1):
    """Iteratively expand the markers white keeping them limited by the mask during each iteration.

    :param marker: Grayscale image where initial seed is white on black background.
    :param mask: Grayscale mask where the valid area is white on black background.
    :param radius Can be increased to improve expansion speed while causing decreased isolation from nearby areas.
    :returns A copy of the last expansion.
    Written By Semnodime.
    """
    kernel = np.ones(shape=(radius * 2 + 1,) * 2, dtype=np.uint8)
    while True:
        expanded = cv2.dilate(src=marker, kernel=kernel)
        cv2.bitwise_and(src1=expanded, src2=mask, dst=expanded)

        # Termination criterion: Expansion didn't change the image at all
        if (marker == expanded).all():
            return expanded
        marker = expanded

def imreconstruct_dual(marker: np.ndarray, mask: np.ndarray, radius: int = 1):
    """Iteratively shrink the markers while keeping them constrained by the mask during each iteration.

    :param marker: Grayscale image where initial seed is white on black background.
    :param mask: Grayscale mask where the valid area is white on black background.
    :param radius: Can be increased to improve shrinking speed while causing decreased isolation from nearby areas.
    :returns: A copy of the last shrinkage.
    Adapted from Semnodime stack overflow implementation.
    """
    kernel = np.ones(shape=(radius * 2 + 1,) * 2, dtype=np.uint8)
    while True:
        eroded = cv2.erode(src=marker, kernel=kernel)
        cv2.max(src1=eroded, src2=mask, dst=eroded)

        # Termination criterion: Erosion didn't change the image at all
        if (marker == eroded).all():
            return eroded
        marker = eroded


def get_mask_and_foreground(original_image):
    """
    Returns a binary mask of the input image.

    Parameters:
        original_image (numpy.ndarray): The input image in BGR format.
    
    Returns:
        numpy.ndarray: A binary mask of the input image, where the foreground is white 
                    (255) and the background is black (0).
    """
    # 1. Image to grayscale
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # 2. Gaussian Blurr
    blurred_immge = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # 3. Otsu's thresholding
    _, th2 = cv2.threshold(blurred_immge, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # 4. Invert the mask
    th2 = cv2.bitwise_not(th2)

    # 5. Morhoplogical operations
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(th2, cv2.MORPH_CLOSE, kernel)

        # Choose largest connected component
    # Find connected components
    num_labels, labels_im = cv2.connectedComponents(mask)

    # If there's more than one connected component, find the largest
    if num_labels > 1:
        largest_component = 1  # Label 0 is the background
        max_size = 0

        for i in range(1, num_labels):
            component_size = np.sum(labels_im == i)
            if component_size > max_size:
                max_size = component_size
                largest_component = i

        # Create a new mask for the largest component
        mask = np.zeros_like(mask)
        mask[labels_im == largest_component] = 255

    # 5. Morhoplogical operations
    kernel = np.ones((70, 70), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # # 6. Dual Reconstruction
    pad_width = 10
    marker = np.pad(np.ones((mask.shape[0] - pad_width*2, mask.shape[1] - pad_width*2), dtype=np.uint8)*255, pad_width=pad_width, mode='constant', constant_values=0)
    mask = imreconstruct_dual(marker, mask)

    # # 7. Opening to remove small gaps between the background
    kernel = np.ones((70, 70), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 8. Set a minimmum margin in the background of 10 pixels
    margin = 10
    mask[:margin, :] = 0  # Top margin
    mask[-margin:, :] = 0  # Bottom margin
    mask[:, :margin] = 0  # Left margin
    mask[:, -margin:] = 0  # Right margin

    # Create foreground by setting background pixels to black
    foreground = original_image.copy()
    foreground[mask == 0] = [0, 0, 0]

    # Return the final mask
    return foreground, mask

def evaluate_pixel_mask(mask_path, groundtruth_path):
    """
    Evaluates the performance of a binary mask against the ground truth mask.
    
    Parameters:
        mask_path (str): Path to the generated mask image.
        groundtruth_path (str): Path to the ground truth image.

    Returns:
        precision (float): Precision score.
        recall (float): Recall score.
        f1_score (float): F1 score.
    """
    # Load mask and ground truth images
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    groundtruth = cv2.imread(groundtruth_path, cv2.IMREAD_GRAYSCALE)

    # Check that images are loaded correctly
    if mask is None:
        raise ValueError(f"Could not load mask image from {mask_path}")
    if groundtruth is None:
        raise ValueError(f"Could not load ground truth image from {groundtruth_path}")

    # Flatten arrays to 1D and in [0, 1] range
    mask_flat = mask.flatten() // 255
    groundtruth_flat = groundtruth.flatten() // 255

    true_positive = np.sum((mask_flat == 1) & (groundtruth_flat == 1))
    false_positive = np.sum((mask_flat == 1) & (groundtruth_flat == 0))
    false_negative = np.sum((mask_flat == 0) & (groundtruth_flat == 1))

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score

def evaluate_masks(masks_path, grountruth_paths):
    """
    Evaluates the performance of a binary mask against the ground truth mask for all dataset images.

    Parameters:
        mask_path (str): Path to the generated mask image.
        groundtruth_path (str): Path to the ground truth image.

    Returns:
        tuple: A tuple containing the average precision, average recall, and average F1 score for all mask images.
    """
    total_precision = 0
    total_recall = 0
    total_f1_score = 0
    
    for mask in os.listdir(masks_path):
        if mask.endswith('.png'):
            mask_filename = mask.split('.')[0]

            try:
                precision, recall, f1_score = evaluate_pixel_mask(os.path.join(masks_path, mask), 
                                                                  os.path.join(grountruth_paths, mask_filename + '.png'))

                total_precision += precision
                total_recall += recall
                total_f1_score += f1_score
            except ValueError as e:
                print(f"Error processing {mask}: {e}")

    # To be changed wwhen creating the dataset with the generated masks
    masks_number = len([mask for mask in os.listdir(masks_path) if mask.endswith('.png')])

    return (total_precision / masks_number, 
            total_recall / masks_number, 
            total_f1_score / masks_number)

# MAIN TESTING
if __name__ == '__main__':

    BASE_PATH = os.path.join(re.search(r'.+(Team5)', os.getcwd())[0], 'week2')
    os.chdir(BASE_PATH)
    DATA_DIRECTORY = '../data'
    precision, recall, f1_score = evaluate_masks("data_results/masks", f'{DATA_DIRECTORY}/qsd2_w2')

    results = pd.DataFrame({
        'Metric': ['Precision', 'Recall', 'F1 Score'],
        'Value': [precision, recall, f1_score]
    })
    print(results)