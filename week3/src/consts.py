from enum import Enum
import cv2

class DescriptorType(Enum):
    Block = 'block'
    Global = 'global'
    LBP = 'lpb'
    DCT = 'dct'


class ColorSpace(Enum):
    gray = cv2.COLOR_BGR2GRAY
    RGB = cv2.COLOR_BGR2RGB
    HSV = cv2.COLOR_BGR2HSV
    CieLab = cv2.COLOR_BGR2Lab
    YCbCr = cv2.COLOR_BGR2YCrCb