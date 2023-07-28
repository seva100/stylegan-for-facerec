import os
import sys
import json
import subprocess
from glob import glob
import random
import contextlib
import joblib
from joblib import Parallel, delayed
import argparse
from tqdm import tqdm
from pytube import YouTube


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def run(video_id, start_segment=0, end_segment=20 * 60):
    out_path = os.path.join(out_dir, video_id)
    if os.path.exists(out_path):
        print(f'{out_path} exists and contains a video or frames -- skipping downloading')
    else:    
        # ------ Downloading a video

        video_id = video_id.strip()
        # yt = YouTube(f'http://youtube.com/watch?v={video_id}')    # no time limit
        yt = YouTube(f'https://youtube.com/watch?v={video_id}?start={start_segment}&end={end_segment}')
        try:
            print(f'https://youtube.com/watch?v={video_id}?start={start_segment}&end={end_segment}')
            (yt.streams
                    .filter(progressive=True, file_extension='mp4')
                    .order_by('resolution')
                    .desc()
                    .first()
                    .download(out_path))
        except Exception as e: 
            print('video', video_id, ' -- exception occured while downloading:')
            print(e)

    # ----- Extracting frames

    video_fn = list(glob(os.path.join(out_path, '*.mp4')))
    if len(video_fn) == 0:
        print(f'video {video_id} was supposed to be download but not found in a folder (an error might have occured during downloading) -> skipping')
        return
    video_fn = video_fn[0]
    # print('extracting frames:', video_fn)
    video_dir = os.path.dirname(video_fn)
    cmd = (f"ffmpeg -t 00:{(end_segment - start_segment) // 60}:{(end_segment - start_segment) % 60} -i {os.path.join(video_dir, '*.mp4')} "
           f"-r 0.1 {os.path.join(video_dir, r'frames_%04d.jpg')}")
 
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    out, err = process.communicate()
    if process.returncode == 0:
        # print("Done")
        pass
    else:
        print(f"video {video_id} -- exception occured while extracting frames")
        print(out)
        print(err)
    
    # ----- Deleting the video
    if os.path.exists(video_fn):
        os.remove(video_fn)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Videos downloader and frames extractor.")
    parser.add_argument("--list_dir", type=str)
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--n_threads", type=int, default=4)
    
    args = parser.parse_args()
    list_dir = args.list_dir
    out_dir = args.out_dir
    n_threads = args.n_threads

    start_segment = 0
    end_segment = 60 * 20

    os.makedirs(out_dir, exist_ok=True)
    
    list_fns = list(glob(os.path.join(list_dir, '*.txt')))

    video_ids = []
    for list_fn in list_fns: 
        video_ids.extend(open(list_fn).read().splitlines())
    print('# ids collected:', len(video_ids))

    # DEBUG
    video_ids = video_ids[:1000]

    random.shuffle(video_ids)
    
    with tqdm_joblib(tqdm(desc="Downloading videos", total=len(video_ids))) as progress_bar:
        Parallel(n_jobs=n_threads)(delayed(run)(cur_id, start_segment, end_segment) for cur_id in video_ids)
