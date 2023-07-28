import torch
import torch.nn as nn
from util.utils import get_val_data, perform_val
from backbone.restyle_psp import pSp

from tqdm import tqdm
import os
import argparse
import importlib


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--config', type=str)

    args = parser.parse_args()

    #======= hyperparameters & data loaders =======#
    import sys
    config_name = args.config
    sys.path.append(os.path.dirname(config_name))
    config_name = os.path.basename(config_name)

    config_name = config_name.replace('.py', '')
    config_name = config_name.replace('/', '.')

    print('config_name:', config_name)
    # exec(f'from {config_name} import configurations')
    config = importlib.import_module(config_name)
    configurations = config.configurations
    cfg = configurations[1]

    SEED = cfg['SEED'] # random seed for reproduce results
    torch.manual_seed(SEED)

    DATA_ROOT = cfg['DATA_ROOT'] # the parent root where your train/val/test data are stored
    MODEL_ROOT = cfg['MODEL_ROOT'] # the root to buffer your checkpoints
    LOG_ROOT = cfg['LOG_ROOT'] # the root to log your train/val status
    BACKBONE_RESUME_ROOT = cfg['BACKBONE_RESUME_ROOT'] # the root to resume training from a saved checkpoint
    HEAD_RESUME_ROOT = cfg['HEAD_RESUME_ROOT']  # the root to resume training from a saved checkpoint

    BACKBONE_NAME = cfg['BACKBONE_NAME'] # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
    HEAD_NAME = cfg['HEAD_NAME'] # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
    LOSS_NAME = cfg['LOSS_NAME'] # support: ['Focal', 'Softmax']

    INPUT_SIZE = cfg['INPUT_SIZE']
    RGB_MEAN = cfg['RGB_MEAN'] # for normalize inputs
    RGB_STD = cfg['RGB_STD']
    EMBEDDING_SIZE = cfg['EMBEDDING_SIZE'] # feature dimension
    BATCH_SIZE = cfg['BATCH_SIZE']
    DROP_LAST = cfg['DROP_LAST'] # whether drop the last batch to ensure consistent batch_norm statistics
    LR = cfg['LR'] # initial LR
    NUM_EPOCH = cfg['NUM_EPOCH']
    WEIGHT_DECAY = cfg['WEIGHT_DECAY']
    MOMENTUM = cfg['MOMENTUM']
    STAGES = cfg['STAGES'] # epoch stages to decay learning rate
    CCROP_AT_VAL = cfg.get('CCROP_AT_VAL', True)

    DEVICE = cfg['DEVICE']
    MULTI_GPU = cfg['MULTI_GPU'] # flag to use multiple GPUs
    # GPU_ID = cfg['GPU_ID'] # specify your GPU ids
    GPU_ID = [0] # specify your GPU ids

    ENCODER_CHECKPOINT = cfg.get('ENCODER_CHECKPOINT', None)
    ENCODER_AVG_IMAGE = cfg.get('ENCODER_AVG_IMAGE', None)
    ENCODER_INPUT_SIZE = cfg.get('ENCODER_INPUT_SIZE', 112)
    ENCODER_ADDITIONAL_DROPOUT = cfg.get('ENCODER_ADDITIONAL_DROPOUT', None)
    
    PIN_MEMORY = cfg['PIN_MEMORY']
    NUM_WORKERS = cfg['NUM_WORKERS']
    print("=" * 60)
    print("Overall Configurations:")
    print(cfg)
    print("=" * 60)

    lfw, cfp_ff, cfp_fp, agedb, calfw, cplfw, vgg2_fp, lfw_issame, cfp_ff_issame, cfp_fp_issame, agedb_issame, calfw_issame, cplfw_issame, vgg2_fp_issame, rfw, rfw_issame = get_val_data(DATA_ROOT)

    #======= model =======#
    if BACKBONE_NAME == 'IR_34_ReStyle':
        BACKBONE = pSp(encoder_type='BackboneEncoder34',
                       size=ENCODER_INPUT_SIZE,
                       checkpoint_path=ENCODER_CHECKPOINT,
                       avg_image=ENCODER_AVG_IMAGE,
                       include_dropout=False)
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
        
    print("=" * 60)
    # print(BACKBONE)
    print("{} Backbone Generated".format(BACKBONE_NAME))
    print("=" * 60)

    # optionally resume from a checkpoint
    if BACKBONE_RESUME_ROOT and HEAD_RESUME_ROOT:
        print("=" * 60)
        if os.path.isfile(BACKBONE_RESUME_ROOT) and os.path.isfile(HEAD_RESUME_ROOT):
            print("Loading Backbone Checkpoint '{}'".format(BACKBONE_RESUME_ROOT))
            BACKBONE.load_state_dict(torch.load(BACKBONE_RESUME_ROOT))
        else:
            print("No Checkpoint Found at '{}' and '{}'. Please Have a Check or Continue to Train from Scratch".format(BACKBONE_RESUME_ROOT, HEAD_RESUME_ROOT))
        print("=" * 60)

    if args.checkpoint:    # easier interface to specify a checkpoint
        print("=" * 60)
        if os.path.exists(args.checkpoint) and os.path.isfile(args.checkpoint):
            print("Loading Backbone Checkpoint '{}'".format(args.checkpoint))
            BACKBONE.load_state_dict(torch.load(args.checkpoint))
        else:
            raise Exception('checkpoint cannot be opened')
        print("=" * 60)

    if MULTI_GPU:
        # multi-GPU setting
        BACKBONE = nn.DataParallel(BACKBONE, device_ids = GPU_ID)
        BACKBONE = BACKBONE.to(DEVICE)
    else:
        # single-GPU setting
        BACKBONE = BACKBONE.to(DEVICE)

    print("=" * 60)
    epoch = 1    # mock value

    print("Performing evaluation...")
    if lfw is not None:
        accuracy_lfw, best_threshold_lfw, roc_curve_lfw = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, lfw, lfw_issame)
        print('LFW accuracy:', accuracy_lfw)

    if cfp_ff is not None:
        accuracy_cfp_ff, best_threshold_cfp_ff, roc_curve_cfp_ff = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, cfp_ff, cfp_ff_issame)
        print('CFP_FF accuracy:', accuracy_cfp_ff)

    if cfp_fp is not None:
        accuracy_cfp_fp, best_threshold_cfp_fp, roc_curve_cfp_fp = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, cfp_fp, cfp_fp_issame)
        print('CFP_FP accuracy:', accuracy_cfp_fp)

    if agedb is not None:
        accuracy_agedb, best_threshold_agedb, roc_curve_agedb = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, agedb, agedb_issame)
        print('AgeDB accuracy:', accuracy_agedb)

    if calfw is not None:
        accuracy_calfw, best_threshold_calfw, roc_curve_calfw = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, calfw, calfw_issame)
        print('CALFW accuracy:', accuracy_calfw)

    if cplfw is not None:
        accuracy_cplfw, best_threshold_cplfw, roc_curve_cplfw = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, cplfw, cplfw_issame)
        print('CPLFW accuracy:', accuracy_cplfw)
    
    if vgg2_fp is not None:
        accuracy_vgg2_fp, best_threshold_vgg2_fp, roc_curve_vgg2_fp = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, vgg2_fp, vgg2_fp_issame, dset_name="VGGFace2")
        print('VGG2_FP accuracy:', accuracy_vgg2_fp)

    if rfw is not None:     
        print("=" * 60)
        for ethnicity in ('African', 'Asian', 'Caucasian', 'Indian'):
            accuracy_rfw_part, best_threshold_rfw_part, roc_curve_rfw_part = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, rfw[ethnicity], rfw_issame[ethnicity], dset_name="RFW_" + ethnicity, ccrop=CCROP_AT_VAL)
            
            print(f"Evaluation: RFW {ethnicity} Acc: {accuracy_rfw_part}")
        print("=" * 60)
