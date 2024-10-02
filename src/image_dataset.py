import os

from src.image import Image, ColorSpace

class ImageDataset:
    def __init__(self, directory_path: str, colorspace: ColorSpace = ColorSpace.RGB, interval: int = 1):
        self.directory_path = directory_path
        self.colorspace = colorspace
        self.interval = interval
        self.descriptors = self.load_dataset()

    def load_dataset(self):
        """
        Load all '.jpg' images from the specified directory and create Image instances for each.
        Stores the Image instances in self.descriptors.
        """

        # List all images in the directory that have '.jpg' extension
        image_filenames = [f for f in os.listdir(self.directory_path) if f.endswith('.jpg')]

        # Create Image instances for each image and store them in self.descriptors
        result = []
        for image_filename in image_filenames:
            image_path = os.path.join(self.directory_path, image_filename)
            image_instance = Image(image_path, self.colorspace, self.interval)  # Assuming Image class takes the image path as an argument
            result.append(image_instance)
        return result
