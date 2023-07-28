import os
from glob import glob
import argparse
import bcolz
import numpy as np
from tqdm import tqdm
import imageio


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Packing a dataset into bcolz format.')

    parser.add_argument('--data_path', type=str, help='path to the folder containing the dataset of images')
    parser.add_argument('--out_path',  type=str, help='path to the output folder where the bcolz array will be stored as a subfolder')
    
    args = parser.parse_args()

    os.makedirs(args.out_path, exist_ok=True)

    for race in ('African', 'Asian', 'Caucasian', 'Indian'):
        test_set_name = f'RFW_{race}'
        pairs_path = os.path.join(args.data_path, 'test', 'txts', race, f'{race}_pairs.txt')

        pairs = open(pairs_path).read().splitlines()
        pairs = [el.split('\t') for el in pairs]
        # pairs format: 
        #      - positive pair: (person_id, img_no_1, img_no_2)
        #      - negative pair: (person_id_1, img_no_1, person_id_2, img_no_2)

        carray = bcolz.carray(np.zeros([0, 3, 112, 112], dtype=np.float32), chunklen=1, mode='w', rootdir=os.path.join(args.out_path, test_set_name))
        issame = []

        for pair in tqdm(pairs):
            if len(pair) == 3:
                # positive pair
                person_id, img_no_1, img_no_2 = pair
                issame_tag = True
                img_no_1, img_no_2 = int(img_no_1), int(img_no_2)
                src_fn = os.path.join(args.data_path, 'test', 'data', race, person_id, f'{person_id}_{img_no_1:04}.jpg')
                tgt_fn = os.path.join(args.data_path, 'test', 'data', race, person_id, f'{person_id}_{img_no_2:04}.jpg')
            elif len(pair) == 4:
                # negative pair
                person_id_1, img_no_1, person_id_2, img_no_2 = pair
                issame_tag = False
                img_no_1, img_no_2 = int(img_no_1), int(img_no_2)
                src_fn = os.path.join(args.data_path, 'test', 'data', race, person_id_1, f'{person_id_1}_{img_no_1:04}.jpg')
                tgt_fn = os.path.join(args.data_path, 'test', 'data', race, person_id_2, f'{person_id_2}_{img_no_2:04}.jpg')
            

            src_img = imageio.imread(src_fn)
            tgt_img = imageio.imread(tgt_fn)
            
            src_img = (src_img.astype(np.float32) / 255.0) * 2 - 1    # into [-1, 1] (correct scale for face-evolve)
            tgt_img = (tgt_img.astype(np.float32) / 255.0) * 2 - 1    # into [-1, 1] (correct scale for face-evolve)

            src_img = src_img.transpose(2, 0, 1)    # into (C, H, W)
            tgt_img = tgt_img.transpose(2, 0, 1)    # into (C, H, W)

            carray.append(src_img)
            carray.append(tgt_img)
            issame.append(issame_tag)

        issame = np.array(issame)
        carray.flush()

        np.save(os.path.join(args.out_path, f'{test_set_name}_list.npy'), issame)
