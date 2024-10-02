import argparse

from image import Image
from metrics import DistanceType, SimilarityType

def load_data_path(data_path):
    return data_path

def main():
    
    # If loading the data using Windows, you may need to use '../' instead of './'
    image = Image('./data/BBDD/bbdd_00000.jpg', interval=1)
    image.plot_histograms()
    image.show()

    image2 = Image('./data/BBDD/bbdd_00002.jpg', interval=1)
    image2.plot_histograms()
    image2.show()

    # Distances
    euclidean = image.compute_distance(image2, type=DistanceType.euclidean)
    l1 = image.compute_distance(image2, type=DistanceType.l1)
    chi2 = image.compute_distance(image2, type=DistanceType.chi2)
    print(f'Euclidean Distance: {euclidean}\n'
        f'L1 Distance: {l1}\n'
        f'Chi^2 Distance: {chi2}\n\n')

    # Similarities
    hellinger_kernel = image.compute_similarity(image2, type=SimilarityType.hellinger_kernel)
    histogram_intersection = image.compute_similarity(image2, type=SimilarityType.histogram_intersection)
    print(f'Hellinger Kernel Similarity: {hellinger_kernel}\n'
        f'Histogram Intersection Similarity: {histogram_intersection}')


if __name__=='__main__':
    # User argparse to handle command-line arguments
    # parser = argparse.ArgumentParser(description='Select data path')
    # parser.add_argument("data_path", help="Path to the data")

    # # Parse arguments
    # args = parser.parse_args()
    
    # try:
    #     data_path = load_data_path(args.data_path)
    # except Exception as e:
    #     print(e)
    #     exit(1)

    main()