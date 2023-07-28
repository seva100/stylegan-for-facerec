import os
import argparse
import scrapetube
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Videos links scraper.")
    parser.add_argument("--info_file", type=str, help="path to file with lines of type '<channel_id> <name>'")
    parser.add_argument("--out_dir", type=str)
    
    args = parser.parse_args()

    info = open(args.info_file).read().splitlines()
    os.makedirs(args.out_dir, exist_ok=True)

    for i, line in enumerate(info):
        channel_name, channel_id = line.split(' ')
        print('Channel #', i)
        print(channel_id, channel_name)

        videos = scrapetube.get_channel(channel_id)
        
        with open(os.path.join(args.out_dir, f'{channel_name}-ids-scrapetube.txt'), 'a') as fout:
            for video in tqdm(videos, total=1000000):
                fout.write(video['videoId'] + '\n')
        
        print('-' * 50)
