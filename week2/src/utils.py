# Read data
import zipfile

def extract(path_to_zip_file: str, directory_to_extract_to: str) -> None:
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)
