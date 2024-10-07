import cv2
import numpy as np

def get_mask(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    mask = cv2.inRange(image, np.array([35, 40, 40]), np.array([85, 255, 255]))
    
    # Binary mask: foreground (1), background (0)
    binary_mask = mask > 0

    return binary_mask
