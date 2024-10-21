
import os
from typing import Dict
from tqdm import tqdm

from src.image import Image
from src.consts import ColorSpace, DescriptorType
from src.descriptors.image_block_descriptor import ImageBlockDescriptor
from src.descriptors.image_lbp_descriptor import ImageLBPDescriptor

class ImageDataset:
    def __init__(
            self,
            directory_path: str,
            descriptor_type: DescriptorType,
            params: Dict
        ):
        self.directory_path = directory_path
        self.descriptor_type = descriptor_type
        self.params = params
        self.images = self.load_dataset()


    def load_dataset(self):
        """
        Load all '.jpg' images from the specified directory and create Image instances for each.
        Stores the Image instances in self.images.
        """

        # List all images in the directory that have '.jpg' extension
        image_filenames = [f for f in os.listdir(self.directory_path) if f.endswith('.jpg')]

        result = [
            Image(os.path.join(self.directory_path, filename), self.descriptor_type, self.params)
            for filename in tqdm(image_filenames)
        ]
        return sorted(result, key=lambda x: x.index)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        return self.images[index]