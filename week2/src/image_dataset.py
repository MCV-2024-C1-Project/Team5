
import os

from src.image import Image, ColorSpace
from src.image_block_descriptor import ImageBlockDescriptor

class ImageDataset:
    def __init__(self, directory_path: str, colorspace: ColorSpace = ColorSpace.RGB, interval: int = 1, histogram_type: str = 'block',
                 rows: int = 4, columns: int = 4):
        self.directory_path = directory_path
        self.colorspace = colorspace
        self.histogram_type = histogram_type
        self.interval = interval
        self.rows = rows
        self.columns = columns
        self.images = self.load_dataset()


    def load_dataset(self):
        """
        Load all '.jpg' images from the specified directory and create Image instances for each.
        Stores the Image instances in self.images.
        """

        # List all images in the directory that have '.jpg' extension
        image_filenames = [f for f in os.listdir(self.directory_path) if f.endswith('.jpg')]

        # Create Image instances for each image and store them in self.images
        result = []
        indexes = []
        for image_filename in image_filenames:
            image_path = os.path.join(self.directory_path, image_filename)
            if self.histogram_type == 'block':
                #TODO: Maybe it's time to move paremeters to a configuration file...
                image_instance = ImageBlockDescriptor(image_path, self.colorspace, self.interval, rows=self.rows, columns=self.columns)
            elif self.histogram_type == 'global':
                image_instance = Image(image_path, self.colorspace, self.interval)
            else:
                raise ValueError(f'Invalid histogram type: {self.histogram_type}. Use "block" or "global".')

            image_instance.compute_image_histogram_descriptor(self.interval)

            result.append(image_instance)
            indexes.append(image_instance.index)
        zipped = zip(indexes, result)
        sorted_zipped = sorted(zipped, key=lambda x: x[0])
        _, sorted_result = zip(*sorted_zipped)
        return sorted_result
    
    
    def change_colorspace(self, new_colorspace: ColorSpace):
        """
        Change the colorspace of all images in the dataset without need of computing again the histograms.
        """
        self.colorspace = new_colorspace
        
        # Change colospace for each image in the dataset
        for image in self.images:
            image.change_colorspace(new_colorspace)

    def change_interval(self, new_interval: int):
        for image in self.images:
            image.compute_image_histogram_descriptor(new_interval)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        return self.images[index]