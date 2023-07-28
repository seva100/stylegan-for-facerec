import argparse
import os
from glob import glob


if __name__ == '__main__':
    parser = argparse.ArgumentParser("")
    parser.add_argument("--dataset_path", type=str, nargs='+', help='should be one or several paths to the folder(s), each containing all extracted images in a format <video_id>/<frame_name>.jpg')
    parser.add_argument("--out_list_path", type=str, help='where to save the output .txt index of files')
    
    args = parser.parse_args()

    all_fn = []
    for path in args.dataset_path:
        all_fn_in_path = list(glob(os.path.join(path, '*', '*.jpg')))
        all_fn.extend(all_fn_in_path)
    
    all_fn = list(sorted(all_fn))
    print(f'# files found in {len(args.dataset_path)} provided dirs:', len(all_fn))

    with open(args.out_list_path, 'w') as fout:
        for fn in all_fn:
            fout.write(fn + '\n')
