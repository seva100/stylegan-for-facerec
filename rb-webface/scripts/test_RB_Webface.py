import os
from glob import glob
from PIL import Image
from collections import defaultdict
import imageio
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import importlib
import skimage
import skimage.transform
import scipy as sp
import scipy.spatial
import sklearn
import sklearn.metrics
import contextlib
import joblib
from joblib import Parallel, delayed

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from einops import rearrange

import sys
sys.path.append('.')    # project root (if running as rb-webface/scripts/test_RB_WebFace.py)
# sys.path.append('../..')    # project root (if running from rb-webface/scripts folder)

from backbone.model_resnet import ResNet_50, ResNet_101, ResNet_152
from backbone.model_irse import IR_50, IR_101, IR_152, IR_SE_50, IR_SE_101, IR_SE_152
from head.metrics import ArcFace, CosFace, SphereFace, Am_softmax
from loss.focal import FocalLoss
from util.utils import make_weights_for_balanced_classes, get_val_data, separate_irse_bn_paras, separate_resnet_bn_paras, warm_up_lr, schedule_lr, perform_val, get_time, buffer_val, AverageMeter, accuracy
from backbone.restyle_psp import pSp


def initialize_model(config_name, checkpoint):
    sys.path.append(os.path.dirname(config_name))
    config_name = os.path.basename(config_name)

    config_name = config_name.replace('.py', '')
    config_name = config_name.replace('/', '.')

    config = importlib.import_module(config_name)
    configurations = config.configurations
    cfg = configurations[1]

    SEED = cfg['SEED'] # random seed for reproduce results
    torch.manual_seed(SEED)

    BACKBONE_NAME = cfg['BACKBONE_NAME'] # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']

    INPUT_SIZE = cfg['INPUT_SIZE']

    ENCODER_CHECKPOINT = cfg.get('ENCODER_CHECKPOINT', None)
    ENCODER_AVG_IMAGE = cfg.get('ENCODER_AVG_IMAGE', None)
    ENCODER_INPUT_SIZE = cfg.get('ENCODER_INPUT_SIZE', 112)
    ENCODER_ADDITIONAL_DROPOUT = cfg.get('ENCODER_ADDITIONAL_DROPOUT', None)


    #======= model & loss & optimizer =======#
    if BACKBONE_NAME == 'IR_34_ReStyle':
        BACKBONE = pSp(encoder_type='BackboneEncoder34',
                       size=ENCODER_INPUT_SIZE,
                       checkpoint_path=ENCODER_CHECKPOINT,
                       avg_image=ENCODER_AVG_IMAGE,
                       include_dropout=ENCODER_ADDITIONAL_DROPOUT)
    if BACKBONE_NAME == 'IR_50_ReStyle':
        BACKBONE = pSp(size=ENCODER_INPUT_SIZE,
                       checkpoint_path=ENCODER_CHECKPOINT,
                       avg_image=ENCODER_AVG_IMAGE,
                       include_dropout=ENCODER_ADDITIONAL_DROPOUT)
    if BACKBONE_NAME == 'IR_100_ReStyle':
        BACKBONE = pSp(encoder_type='BackboneEncoder100',
                       size=ENCODER_INPUT_SIZE,
                       checkpoint_path=ENCODER_CHECKPOINT,
                       avg_image=ENCODER_AVG_IMAGE,
                       include_dropout=ENCODER_ADDITIONAL_DROPOUT)

    if os.path.exists(checkpoint) and os.path.isfile(checkpoint):
        print("Loading Backbone Checkpoint '{}'".format(checkpoint))
        BACKBONE.load_state_dict(torch.load(checkpoint))
    else:
        raise Exception('checkpoint cannot be opened')
    
    BACKBONE.eval()
    BACKBONE.to('cuda:0')
    
    return BACKBONE



class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, paths_list=None):
        super().__init__()
        self.paths = paths_list
        
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize([128, 128]),
            torchvision.transforms.CenterCrop([112, 112]),
            
            torchvision.transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)


def l2_normalize(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


def calc_embeddings(backbone, names, data_dir, batch_size=50):
    all_emb = []

    absnames = [os.path.join(data_dir, name) for name in names]
    dataset_train = ImageDataset(absnames)

    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=batch_size, pin_memory=True,
        num_workers=8, drop_last=False, shuffle=False, 
        collate_fn=None
    )
    
    with torch.no_grad():
        for img_t in tqdm(train_loader):

            img_t = img_t.to('cuda:0')

            emb = backbone(img_t)
            emb = l2_normalize(emb)
            all_emb.append(emb.cpu().data.numpy())
        
    all_emb = np.concatenate(all_emb, axis=0)
    return all_emb


def calc_FNMR(pos_emb, threshold, n_names_per_grp=5):
    fnmr = 0
    pairs_seen = 0

    for i in tqdm(range(0, len(pos_emb), n_names_per_grp)):

        pos_emb_cur_id_group = pos_emb[i:i + n_names_per_grp]

        # u*v / ||u|| / ||v||:
        group_pdist = 1 - sp.spatial.distance.pdist(pos_emb_cur_id_group, 
                                                    metric='cosine')    
        # group_pdist is a compressed matrix -- 
        # -- only upper-diagonal values are returned

        # score of 1 means super-close embeddings <=> the same people
        # score of -1 means super-far embeddings <=> different people
        # we now that all these pairs are of the same people, 
        # so we calculate the number of misses (when score is less than the threshold).
        fnmr += (group_pdist < threshold).sum()
        pairs_seen += group_pdist.size

    # FNMR = 1 - 1/N * sum(u_i - T), where u_i is a genuine score (the more it is, the more probable the embeddings are of the same person)
    #      = 1/N * sum([u_i < T])
    #      = <fraction of predictions that it's different people across all positive pairs>
    fnmr /= pairs_seen
    return fnmr


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


def calc_FMR(neg_emb, threshold, n_jobs=1, batch_size=1000):

    def FMR_calc_chunk(i, threshold, batch_size=1000):
        D_chunk = 1 - sp.spatial.distance.cdist(neg_emb[i:i + batch_size], 
                                                neg_emb, 
                                                metric='cosine')
        
        i_grid, j_grid = np.meshgrid(np.arange(batch_size), np.arange(D_chunk.shape[1]), indexing='ij')
        rows_seen = i
        upper_trapezoid = (j_grid) > (i_grid + rows_seen)
        
        D_chunk_condensed = D_chunk[upper_trapezoid]
        
        # false match rate: fraction of cases when neg pair is predicted is same people
        # => `fmr += (D_chunk_condensed > T).sum()
        fmr = (D_chunk_condensed > threshold).sum()
        pairs_seen = (D_chunk_condensed > threshold).size
    #     print(fmr)
        
        # false match rate: fraction of cases when neg pair is predicted is same people
        # cosine distance is inverted => `fmr += (D_chunk_condensed < T).sum()
        return fmr, pairs_seen

    with tqdm_joblib(tqdm(desc="pairwise distances", 
                          total=len(neg_emb) // batch_size)) as progress_bar:
        results = Parallel(n_jobs=n_jobs)(delayed(FMR_calc_chunk)(i, threshold, batch_size=batch_size) 
                                     for i in range(0, len(neg_emb), batch_size))
    
    fmr = 0
    pairs_seen = 0
    for fmr_part, pairs_seen_part in results:
        fmr += fmr_part
        pairs_seen += pairs_seen_part
    fmr /= pairs_seen
    
    return fmr


def evaluate_model(config_name, checkpoint, data_dir, test_names_dir, cpu_batch_size=1000, cpu_n_jobs=8, gpu_batch_size=50):

    tpr_at3, tpr_at4 = dict(), dict()

    print('initializing model...')
    backbone = initialize_model(config_name, checkpoint)

    for grp_no in range(4):
        
        names_for_pos_pairs = open(os.path.join(test_names_dir, f'pos_pairs_samples_{race2class[grp_no]}.txt')).read().splitlines()
        names_for_neg_pairs = open(os.path.join(test_names_dir, f'neg_pairs_samples_{race2class[grp_no]}.txt')).read().splitlines()
        
        all_fpr = []
        all_fnr = []
        
        # The range covers FPR between {1e-3, 1e-4} for the "strong" model (trained on 10% of the labeled dataset or more).
        # For weaker models (e.g. trained on 1% of the labeled dataset), the range should be adjusted to higher values (e.g. [0.9, 1.0]).
        # Higher number of thresholds in the array yields more precise TPR@{1e-3, 1e-4} values.
        all_thresholds = np.linspace(0.3, 0.6, num=20)

        print('calculating embeddings for positive names')
        pos_emb = calc_embeddings(backbone, names_for_pos_pairs, data_dir,  batch_size=gpu_batch_size)
        print('calculating embeddings for negative names')
        neg_emb = calc_embeddings(backbone, names_for_neg_pairs, data_dir, batch_size=gpu_batch_size)

        for threshold in all_thresholds:
            print('trying threshold', threshold)
            print('calculating FNMR...')
            fnmr = calc_FNMR(pos_emb, threshold, n_names_per_grp=5)    # n_names_per_grp: how many consecutive entries in 
                                                                       # samples_pos_pairs_*.txt correspond to the same person
            print('calculating FMR...')
            fmr = calc_FMR(neg_emb, threshold, batch_size=cpu_batch_size, n_jobs=cpu_n_jobs)
            print('fnmr', fnmr, 'fmr', fmr)

            all_fnr.append(fnmr)
            all_fpr.append(fmr)
        
        # # plotting the TPR curve given various FPR
        # plt.rcParams.update({'font.size': 12,})
        # plt.figure(figsize=(5, 5))
        # plt.semilogx(all_fpr, 1 - np.array(all_fnr))
        # plt.title(f'ROC (RB-WebFace {race2class[grp_no]})')
        # plt.xlim(left=1e-8, right=1e-1)
        # plt.xlabel('FPR')
        # plt.ylabel('TPR')
        # plt.savefig(f'ROC_{race2class[grp_no]}_pretr_ours_three.pdf')
        # plt.savefig(f'ROC_{race2class[grp_no]}_pretr_ours_three.jpg')
        
        print('=' * 20)
        print('Group ', race2class[grp_no])
        print('TPR@FPR=1e-3', 1 - np.interp(1e-3, all_fpr[::-1], all_fnr[::-1]))
        print('TPR@FPR=1e-4', 1 - np.interp(1e-4, all_fpr[::-1], all_fnr[::-1]))
        print()

        tpr_at3[race2class[grp_no]] = (1 - np.interp(1e-3, all_fpr[::-1], all_fnr[::-1]))
        tpr_at4[race2class[grp_no]] = (1 - np.interp(1e-4, all_fpr[::-1], all_fnr[::-1]))
    
    return tpr_at3, tpr_at4


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing the network on RB-WebFace.')
    parser.add_argument('--data_path', type=str, default='../', help='path to the folder containing WebFace dataset of images')
    parser.add_argument('--partition_path', type=str, default='../', help='path to the folder containing lists of samples forming RB-WebFace pairs')
    parser.add_argument('--model_ckpt_path', type=str, help='path to the checkpoint (training code saves it as "Backbone_....pth")')
    parser.add_argument('--config_name', type=str, help='config corresponding to the loaded model. Only some of the entries will be read (see `initialize_model(...)`).)')
    parser.add_argument('--cpu_batch_size', type=int, default=1000, help='batch size for FMR and FNMR calculation. The larger it is, the more RAM is going to be used. At least 100 is recommended.')
    parser.add_argument('--cpu_n_jobs', type=int, default=2, help='number of parallel threads used for FMR calculation. Adjust according to # CPU cores and RAM. At least 8 is recommended.')
    parser.add_argument('--gpu_batch_size', type=int, default=50, help='batch size for embeddings calculation. The larger it is, the more GPU memory is going to be used. At least 25 is recommended.')

    args = parser.parse_args()

    class2race = {
        'African': 0,
        'Asian': 1,
        'Caucasian': 2,
        'Indian': 3
    }
    race2class = {v: k
                for k, v in class2race.items()}
    
    tpr_at3, tpr_at4 = evaluate_model(args.config_name, args.model_ckpt_path, args.data_path, args.partition_path,
                                      cpu_batch_size=args.cpu_batch_size, cpu_n_jobs=args.cpu_n_jobs, gpu_batch_size=args.gpu_batch_size)
    # printing is done inside evaluate_model(...)
