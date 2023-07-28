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


def run(img_fns, ldms):
    for i, name in tqdm(enumerate(img_fns), total=len(img_fns)):
        if not name.lower().endswith('.jpg') and not name.lower().endswith('.png'):
            print('Skipping', name, 'because it is not a jpg or png file.')
            continue
        
        group_folder = name.split(os.sep)[-3]
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

        ldms_mx = ldms[i].reshape(5, 2)

        img_crop = ldms_transform(img, ldms_mx, (args.out_res, args.out_res))

        # <in_dir>/<tag>/<name>.jpg -> <out_dir>/<tag>/<name>_<face_no>.jpg
        out_bn = os.path.basename(name)
        out_name = os.path.join(args.out_dir, 'test', 'data', group_folder, tag_folder, out_bn)

        os.makedirs(os.path.dirname(out_name), exist_ok=True)
        imageio.imwrite(out_name, img_crop)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Crop & align images by landmarks in a given folder.")
    parser.add_argument("--in_dir", type=str, help='RFW folder')
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--out_res", type=int, default=112)
    parser.add_argument("--n_threads", type=int, default=1, help='number of parallel threads to use.')
    args = parser.parse_args()

    for grp in ('African', 'Asian', 'Caucasian', 'Indian'):
        print('Processing', grp)

        ldms_file_content = open(os.path.join(args.in_dir, 'test', 'txts', grp, f'{grp}_lmk.txt')).read().splitlines()
        ldms_file_content = [line.split('\t') for line in ldms_file_content]
        img_fns = [os.path.join(args.in_dir, line[0][1:]) for line in ldms_file_content]    # [1:] -- removing leading slash
        ldms = np.stack([
            np.array([float(el) for el in line[2:]]) 
            for line in ldms_file_content
        ], axis=0)

        print('Starting crop & align...')
        print('You will see several progress bars writing to stdout at the same time (one for each thread).')
        Parallel(n_jobs=args.n_threads)(
            delayed(run)(img_fns[int(len(img_fns) / args.n_threads * i) : int(len(img_fns) / args.n_threads * (i + 1))],
                         ldms[int(len(img_fns) / args.n_threads * i) : int(len(img_fns) / args.n_threads * (i + 1))])
            for i in range(args.n_threads)
        )
        print('Done.')
