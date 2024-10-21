import zipfile
import numpy as np

def extract(path_to_zip_file: str, directory_to_extract_to: str) -> None:
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)

def zigzag(a: np.array):
    return np.concatenate([
        np.diagonal(a[::-1,:], i)[::(2*(i % 2)-1)] 
        for i in range(1-a.shape[0], a.shape[0])
    ])


