"""
Code adopted from pix2pixHD:
https://github.com/NVIDIA/pix2pixHD/blob/master/data/image_folder.py
"""
import os
from tqdm import tqdm

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    print('Loading dataset...')
    for root, _, fnames in tqdm(sorted(os.walk(dir))):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images


def make_dataset_from_set_of_dirs(dirs):
    if not isinstance(dirs, list):
        images = make_dataset(dirs)
    else:
        images = []
        for d in dirs:
            images.extend(make_dataset(d))

    return images
