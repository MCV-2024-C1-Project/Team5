import cv2
import matplotlib.pyplot as plt
import platform
from typing import Dict

from src.consts import DescriptorType
from src.descriptors.image_block_descriptor import ImageBlockDescriptor
from src.descriptors.image_global_descriptor import ImageGlobalDescriptor
from src.descriptors.image_lbp_descriptor import ImageLBPDescriptor
from src.descriptors.image_dct_descriptor import ImageDCTDescriptor

from src.metrics import DistanceType, SimilarityType


class Image:
    def __init__(self, path: str, descriptor_type: DescriptorType, params: Dict):
        self.path = path
        self.index = self._extract_index(path)
        self.original_image = cv2.imread(path)
        self.descriptor_type = descriptor_type
        self.params = params
        self.descriptors = self._get_descriptors(descriptor_type, params)

    def modify_image(self, new_image):
        self.original_image = new_image
        self.descriptors = self._get_descriptors(self.descriptor_type, self.params)

    def _get_descriptors(self, descriptor_type: DescriptorType, params: Dict):
        if descriptor_type == DescriptorType.Block:
            return ImageBlockDescriptor(self.original_image, **params)
        elif descriptor_type == DescriptorType.Global:
            return ImageGlobalDescriptor(self.original_image, **params)
        elif descriptor_type == DescriptorType.LBP:
            return ImageLBPDescriptor(self.original_image, **params)
        elif descriptor_type == DescriptorType.DCT:
            return ImageDCTDescriptor(self.original_image, **params)
        else:
            raise NotImplementedError(f'No descriptor implemented for {descriptor_type}')

    def _extract_index(self, file_path):
        os_name = platform.system()

        if os_name == 'Windows':
            file_name = file_path.split('\\')[-1]
        else:
            file_name = file_path.split('/')[-1]

        name = file_name.split('.')[0]
        number = name.split('_')[-1]
        return int(number)

    def show(self):
        """
        Shows the image in the RGB colorspace, just as they are stored originally in the database.
        """
        bgr_image = cv2.imread(self.path)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        plt.imshow(rgb_image)
        plt.title('RGB Image')
        plt.axis('off')
        plt.show()

    def compute_distance(self, image2: 'Image', type=DistanceType):
        return self.descriptors.compute_distance(image2.descriptors, type)

    def compute_similarity(self, image2: 'Image', type=SimilarityType):
        return self.descriptors.compute_similarity(image2.descriptors, type)