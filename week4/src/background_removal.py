import os
import cv2
import numpy as np
import re
import pandas as pd
from copy import deepcopy
from skimage import filters

from src.noise_removal import denoise_image

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

def apply_filter(image, filter_type):
    # Noise reduction filters
    if filter_type == 'median':
        return cv2.medianBlur(image, 5)
    elif filter_type == 'gaussian':
        return cv2.GaussianBlur(image, (3, 3), 0)

    # Gradient-based edge detection filters
    elif filter_type == 'laplacian':
        return cv2.Laplacian(image, cv2.CV_64F)
    elif filter_type == 'prewitt':
        kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        kernely = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        return cv2.filter2D(image, -1, kernelx) + cv2.filter2D(image, -1, kernely)
    elif filter_type == 'sobel':
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        return cv2.magnitude(sobelx, sobely)
    elif filter_type == 'roberts':
        # Roberts cross kernels
        kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
        kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)

        # Apply the kernels to the image
        grad_x = cv2.filter2D(image, cv2.CV_64F, kernel_x)
        grad_y = cv2.filter2D(image, cv2.CV_64F, kernel_y)

        # Calculate the gradient magnitude
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        return grad_magnitude

    # Advanced edge detection filters
    elif filter_type == 'scharr':
        scharrx = cv2.Scharr(image, cv2.CV_64F, 1, 0)
        scharry = cv2.Scharr(image, cv2.CV_64F, 0, 1)
        return cv2.magnitude(scharrx, scharry)
    elif filter_type == 'canny':
        threshold1 = 30
        threshold2 = 50
        return cv2.Canny(image, threshold1, threshold2)
    elif filter_type == "gabor":
        # Apply the Gabor filter
        frequency = 0.52  # Frequency of the sinusoidal wave
        theta = np.pi / 2  # Orientation of the Gabor filter 
        # Gabor kernel
        real, imag = filters.gabor(image, frequency=frequency, theta=theta)
        # Calculate the magnitude of the Gabor response
        return np.sqrt(real**2 + imag**2)

    elif filter_type == 'identity':
        return image
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")

# Canny Edge Detector: Known for its multi-stage algorithm that includes noise reduction, gradient calculation, non-maximum suppression, and edge tracking by hysteresis.
# Kirsch Operator: Uses a set of eight convolution kernels to detect edges in different directions.
# Frei-Chen Filter: A set of nine masks used to detect edges and other features in an image.
# Robinson Compass Masks: Similar to the Kirsch operator but uses different masks for edge detection in various directions.


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize the input image to the range [0, 255].

    Parameters:
    image (np.ndarray): Input image to be normalized.

    Returns:
    np.ndarray: Normalized image.
    """
    image_min = image.min()
    image_max = image.max()

    # Shift and scale normalization
    normalized_image = (image - image_min) / (image_max - image_min) * 255

    return normalized_image


def get_mask_and_foreground(original_image, enhancing_factor=0, th2_method='grabcut', equalize=False, closing_size=17,
                            opening_size=47, adaptative_area=15, grabcut_iters=5):
    """
    Returns a binary mask of the input image.
laplacian
    Parameters:
        original_image (numpy.ndarray): The input image in BGR format.
    
    Returns:
        numpy.ndarray: A binary mask of the input image, where the foreground is white 
                    (255) and the background is black (0).
    """
    # 1. Denoise the image
    original_image = denoise_image(original_image)
    if equalize:
        for channel in range(original_image.shape[2]):
            original_image[:, :, channel] = cv2.equalizeHist(original_image[:, :, channel])

    # 2. Image to grayscale
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # 3. Edge enhancing
    if np.absolute(enhancing_factor) < 0.01:
        edge_enhanced_image = normalize_image(gray_image).astype(np.uint8)
    else:
        sobel_filtered_image = apply_filter(gray_image, 'sobel')
        edge_enhanced_image = normalize_image(gray_image)-enhancing_factor*normalize_image(sobel_filtered_image)
        edge_enhanced_image = normalize_image(edge_enhanced_image).astype(np.uint8)
    if equalize:
        edge_enhanced_image = cv2.equalizeHist(edge_enhanced_image)

    # 4. Thresholding (with downsampling)
    if th2_method == 'otsu':
        _, th2 = cv2.threshold(edge_enhanced_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    elif th2_method == 'adaptative':
        th2 = cv2.adaptiveThreshold(
            edge_enhanced_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, adaptative_area, 2
        )
    elif th2_method == 'grabcut':
        # Set parameters
        np.random.seed(42)
        grabcut_margin = 10
        grabcut_iters = 5  # Adjust based on your needs
        resize_factor = 0.25  # Adjust based on desired speed-up (0.5 = 50% reduction)

        small_image = cv2.resize(deepcopy(original_image), (0, 0), fx=resize_factor, fy=resize_factor)

        # Initialize mask and models
        mask = np.zeros(small_image.shape[:2], np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        # Define the smaller rectangle for grabCut
        rect = (int(grabcut_margin * resize_factor), int(grabcut_margin * resize_factor),
                int(small_image.shape[1] - grabcut_margin * resize_factor),
                int(small_image.shape[0] - grabcut_margin * resize_factor))

        # Run grabCut on the smaller image
        cv2.grabCut(small_image, mask, rect, bgd_model, fgd_model, grabcut_iters, cv2.GC_INIT_WITH_RECT)

        # Resize mask back to the original image size
        resized_mask = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Generate the final mask
        mask2 = np.where((resized_mask == 2) | (resized_mask == 0), 1, 0).astype('uint8')
        th2 = mask2 * 255

    else:
        raise ValueError(f"Invalid thresholding method: {th2_method}. Please use 'otsu' or 'adaptative'.")

    # 5. Invert the mask
    th2 = cv2.bitwise_not(th2)
    margin = 10
    th2[:margin, :] = 0  # Top margin
    th2[-margin:, :] = 0  # Bottom margin
    th2[:, :margin] = 0  # Left margin
    th2[:, -margin:] = 0  # Right margin

    # 6. Morhoplogical operations
    kernel = np.ones((opening_size, opening_size), np.uint8)
    opening_mask = cv2.morphologyEx(th2, cv2.MORPH_OPEN, kernel)

    # 6..1 Dual Reconstruction
    pad_width = 10
    marker = np.pad(np.ones((opening_mask.shape[0] - pad_width*2, opening_mask.shape[1] - pad_width*2), dtype=np.uint8)*255, pad_width=pad_width, mode='constant', constant_values=0)
    reconstruct = imreconstruct_dual(marker, opening_mask)

    # 6.2. Morhoplogical operations
    kernel = np.ones((closing_size, closing_size), np.uint8)
    mask = cv2.morphologyEx(reconstruct, cv2.MORPH_CLOSE, kernel)


    # 7. Set a minimmum margin in the background of 10 pixels
    margin = 10
    mask[:margin, :] = 0  # Top margin
    mask[-margin:, :] = 0  # Bottom margin
    mask[:, :margin] = 0  # Left margin
    mask[:, -margin:] = 0  # Right margin

    foreground = original_image.copy()
    foreground[mask == 0] = [0, 0, 0]
    return foreground, mask


def evaluate_pixel_mask(mask_path, groundtruth_path):
    """
    Evaluates the performance of a binary mask against the ground truth mask.

    Parameters:
        mask_path (Union[str, np.ndarray]): Path to the generated mask image.
        groundtruth_path (Union[str, np.ndarray]): Path to the ground truth image.

    Returns:
        precision (float): Precision score.
        recall (float): Recall score.
        f1_score (float): F1 score.
    """
    if isinstance(mask_path, str) and isinstance(groundtruth_path, str):
        # Load mask and ground truth images
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        groundtruth = cv2.imread(groundtruth_path, cv2.IMREAD_GRAYSCALE)
    elif isinstance(mask_path, np.ndarray) and isinstance(groundtruth_path, np.ndarray):
        mask = mask_path
        groundtruth = groundtruth_path
    else:
        raise ValueError("Invalid input types for mask_path and groundtruth_path. Expected str or np.ndarray.")

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


def evaluate_masks(masks_dir, grountruth_dir):
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
    
    for mask in os.listdir(masks_dir):
        if mask.endswith('.png'):
            try:
                mask_path = os.path.join(masks_dir, mask)
                mask_filename = mask.split('.')[0]
                gt_path = os.path.join(grountruth_dir, mask_filename + '.png')

                precision, recall, f1_score = evaluate_pixel_mask(mask_path, gt_path)

                total_precision += precision
                total_recall += recall
                total_f1_score += f1_score
            except ValueError as e:
                print(f"Error processing {mask}: {e}")

    # To be changed wwhen creating the dataset with the generated masks
    masks_number = len([mask for mask in os.listdir(masks_dir) if mask.endswith('.png')])

    return (total_precision / masks_number, 
            total_recall / masks_number, 
            total_f1_score / masks_number)

# MAIN TESTING
if __name__ == '__main__':

    BASE_PATH = os.path.join(re.search(r'.+(Team5)', os.getcwd())[0], 'week3')
    os.chdir(BASE_PATH)
    DATA_DIRECTORY = '../data'
    precision, recall, f1_score = evaluate_masks("data_results/masks", f'{DATA_DIRECTORY}/qsd2_w3')

    results = pd.DataFrame({
        'Metric': ['Precision', 'Recall', 'F1 Score'],
        'Value': [precision, recall, f1_score]
    })
    print(results)


def background_removal(image):
    # Get mask and foreground
    foreground, mask = get_mask_and_foreground(image)

    # Find connected components
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8))

    # Sort components based on their position (x, y)
    indexes_sorted = np.argsort(stats[:, 1] + stats[:, 0])

    # Remove the background component (first index)
    indexes_sorted = indexes_sorted[1:]

    # Filter out small components based on area
    min_area = 0.05 * image.shape[0] * image.shape[1]
    indexes_filtered = indexes_sorted[stats[indexes_sorted, cv2.CC_STAT_AREA] > min_area]

    # Create an output mask for the filtered components
    result = []
    for index in indexes_filtered:
        output_mask = np.zeros_like(mask)
        output_mask[labels_im == index] = 255
        cut_image = cv2.bitwise_and(image, image, mask=output_mask)

        x, y, w, h, _ = stats[index]
        
        # Return only image within the bounding box
        result.append(cut_image[y:y+h, x:x+w])


    
    return result
