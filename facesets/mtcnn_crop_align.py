import os
import argparse

import warnings
warnings.filterwarnings('ignore')

from glob import glob
import numpy as np
import imageio
from tqdm import tqdm

from PIL import Image
Image.MAX_IMAGE_PIXELS = 933120000    # workaround over PIL.Image.DecompressionBombError exception

# from turbojpeg import TurboJPEG
import pandas as pd
import imageio
import cv2
from skimage import transform as trans
from joblib import Parallel, delayed


def ldms_transform(img, landmark, image_size):
    """Code of this function is originally taken from Tencent/TFace repository.
    """
    assert landmark.shape[0] == 68 or landmark.shape[0] == 5
    assert landmark.shape[1] == 2
    if landmark.shape[0] == 68:
        landmark5 = np.zeros((5, 2), dtype=np.float32)
        landmark5[0] = (landmark[36] + landmark[39]) / 2    # left eye
        landmark5[1] = (landmark[42] + landmark[45]) / 2    # right eye
        landmark5[2] = landmark[30]    # nose
        landmark5[3] = landmark[48]    # mouth_left
        landmark5[4] = landmark[54]    # mouth_right
    else:
        landmark5 = landmark
    tform = trans.SimilarityTransform()
    src = np.array([
      [30.2946, 51.6963],
      [65.5318, 51.5014],
      [48.0252, 71.7366],
      [33.5493, 92.3655],
      [62.7299, 92.2041]],
      dtype=np.float32)
    src[:, 0] += 8.0

    src[:, 0] *= image_size[1] / 112.0
    src[:, 1] *= image_size[0] / 112.0

    # print('src:', src)

    tform.estimate(landmark5, src)
    M = tform.params[0:2, :]
    img = cv2.warpAffine(img, M, (image_size[1], image_size[0]), 
                         borderValue=0.0)
    return img


def run(img_fns, check_exists=False):
    from src import detect_faces    # initializing networks here

    for name in tqdm(img_fns):
        if not name.lower().endswith('.jpg') and not name.lower().endswith('.png'):
            continue
        
        tag_folder = name.split(os.sep)[-2]
        try:
            img = imageio.imread(name)
        except:    # broken file
            print('Error when reading a file', name)
            continue
            # raise Exception('')
        
        if len(img.shape) == 2:
            img = img[..., np.newaxis]
            img = np.repeat(img, repeats=3, axis=2)
        if img.shape[2] == 4:
            img = img[..., :3]

        h, w = img.shape[:2]
        w_added = 0
        if h > w:
            img = np.hstack([np.zeros_like(img)[:, :(h - w) // 2],
                             img,
                             np.zeros_like(img)[:, :(h - w) // 2 + (h - w) % 2]])
            w_added = (h - w) // 2
        
        h_added = 0
        if w > h:
            img = np.vstack([np.zeros_like(img)[:(w - h) // 2],
                             img,
                             np.zeros_like(img)[:(w - h) // 2 + (w - h) % 2]])
            h_added = (w - h) // 2

        try:
            bounding_boxes, landmarks = detect_faces(img, 
                                                     min_face_size=100,
                                                     thresholds=[0.9, 0.9, 0.9],
                                                     nms_thresholds=[0.9, 0.9, 0.9])
        except ValueError:
            print('ValueError from detector caught on a too small image:')
            print('name:', name)
            print('img shape:', img.shape)
            continue

        if len(landmarks) == 0:
            print(f'img {name} no faces found')
            continue
        for face_no, ldms_vec in enumerate(landmarks):
            ldms_mx = ldms_vec.reshape(5, 2, order='F')
            ldms_vec = ldms_mx.ravel('F')

            img_crop = ldms_transform(img, ldms_mx, (args.out_res, args.out_res))

            # <in_dir>/<tag>/<name>.jpg -> <out_dir>/<tag>/<name>_<face_no>.jpg
            out_bn = name.split(os.sep)[-1] + '_' + str(face_no) + '.jpg'
            out_name = os.path.join(args.out_dir, tag_folder, out_bn)

            os.makedirs(os.path.dirname(out_name), exist_ok=True)
            imageio.imwrite(out_name, img_crop)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Crop & align images by landmarks in a given folder.")
    parser.add_argument("--in_dir", type=str, help='folder with subdirectories, each containing .jpg and/or .png images')
    parser.add_argument("--in_list", type=str, 
        help='if provided, this list (.txt file) will be used instead of scanning '
             'through subdirs of in_dir. in_dir is thus ignored.')
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--out_res", type=int, default=112)
    parser.add_argument("--n_threads", type=int, default=1, help='number of parallel threads to use. Note that it proportionally increases the amount of GPU memory used.')
    parser.add_argument("--mtcnn_pytorch_path", type=str, help='specify the path to mtcnn-pytorch source code')
    parser.add_argument("--check_exists", action='store_true', default=False, help='if True, not overwrite already processed photos in out_dir')
    args = parser.parse_args()

    import sys
    sys.path.append(args.mtcnn_pytorch_path)

    if args.in_list is not None:
        print('--in_list provided => ignoring --in_dir and --in_dir_frac')
        img_fns = open(args.in_list).read().splitlines()
        print('# images in the list:', len(img_fns))
    else:        
        img_fns = list(glob(os.path.join(args.in_dir, '**', '*.jpg'), recursive=True))
        print('# images found:', len(img_fns))

    if args.check_exists:
        print('Filtering out already processed photos...')
        img_fns_corr = []

        for name in tqdm(img_fns):
            if not name.lower().endswith('.jpg') and not name.lower().endswith('.png'):
                continue
                
            tag_folder = name.split(os.sep)[-2]
            sample_face_no = 0
            sample_out_bn = name.split(os.sep)[-1] + '_' + str(sample_face_no) + '.jpg'
            sample_out_name = os.path.join(args.out_dir, tag_folder, sample_out_bn)
            if os.path.exists(sample_out_name):
                continue
                
            img_fns_corr.append(name)
        
        img_fns = img_fns_corr
        print('# images left for the processing:', len(img_fns))

    print('Starting crop & align...')
    print('You will see several progress bars writing to stdout at the same time (one for each thread).')
    Parallel(n_jobs=args.n_threads)(
        delayed(run)(img_fns[int(len(img_fns) / args.n_threads * i) : int(len(img_fns) / args.n_threads * (i + 1))], check_exists=args.check_exists)
        for i in range(args.n_threads)
    )
    print('Done.')
