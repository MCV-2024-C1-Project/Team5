import os
import cv2
import numpy as np
import re

def get_mask(original_image):
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
    kernel = np.ones((5, 5), np.uint8) # Should it be a rectangle?
    morph_mask = cv2.morphologyEx(th2, cv2.MORPH_CLOSE, kernel)

    # ...

    # Return the final mask
    return th2

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

    # Flatten arrays to 1D
    mask_flat = mask.flatten() // 255
    groundtruth_flat = groundtruth.flatten() // 255

    true_positive = np.dot(mask_flat, groundtruth_flat)
    false_positive = np.dot(mask_flat, 1 - groundtruth_flat)
    true_negative = np.dot(1 - mask_flat, 1 - groundtruth_flat)
    false_negative = np.dot(groundtruth_flat, 1 - mask_flat)

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score

def evaluate_mask(masks_path, grountruth_paths):
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
            precision, recall, f1_score = evaluate_pixel_mask(os.path.join(masks_path, mask),
                                                            os.path.join(grountruth_paths, mask_filename + '.png'))

            total_precision += precision
            total_recall += recall
            total_f1_score += f1_score

    # To be changed wwhen creating the dataset with the generated masks
    masks_number = len([mask for mask in os.listdir(masks_path) if mask.endswith('.png')])

    return (total_precision / masks_number, 
            total_recall / masks_number, 
            total_f1_score / masks_number)

# Testing
if __name__ == '__main__':
    BASE_PATH = os.path.join(re.search(r'.+(Team5)', os.getcwd())[0], 'week2')
    os.chdir(BASE_PATH)
    DATA_DIRECTORY = '../data'
    evaluate_mask(f'{DATA_DIRECTORY}/qsd2_w2', f'{DATA_DIRECTORY}/qsd2_w2')
