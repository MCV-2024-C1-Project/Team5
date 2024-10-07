# Read data
import zipfile

def extract(path_to_zip_file: str, directory_to_extract_to: str) -> None:
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)

DATA_DIRECTORY = './data'
extract(f'{DATA_DIRECTORY}/BBDD.zip', DATA_DIRECTORY)
extract(f'{DATA_DIRECTORY}/qsd1_w1.zip', DATA_DIRECTORY)