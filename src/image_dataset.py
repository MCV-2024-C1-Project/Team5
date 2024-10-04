import os

from src.image import Image, ColorSpace

class ImageDataset:
    def __init__(self, directory_path: str, colorspace: ColorSpace = ColorSpace.RGB, interval: int = 1):
        self.directory_path = directory_path
        self.colorspace = colorspace
        self.interval = interval
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
            image_instance = Image(image_path, self.colorspace, self.interval)  # Assuming Image class takes the image path as an argument
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


    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        return self.images[index]