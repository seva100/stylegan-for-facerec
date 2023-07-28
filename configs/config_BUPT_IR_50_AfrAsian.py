import os
import torch
import numpy as np

EXP_NAME = 'BUPT_IR_50_AfrAsian'

configurations = {
    1: dict(
        SEED = 900, # random seed for reproduce results

        EXP_NAME = EXP_NAME,
        DATA_ROOT = '<path to the folder containing BUPT-BalancedFace and test datasets as subfolders>', # the parent root where your train/val/test data are stored
        TRAIN_IMAGES_FOLDER = 'bupt-balancedface',    # name of the subfolder of DATA_ROOT containing BUPT-BalancedFace dataset
        MODEL_ROOT = os.path.join('exps/model/', EXP_NAME),    # checkpoint save dir
        LOG_ROOT = os.path.join('exps/log', EXP_NAME),         # the root to log your train/val status
        BACKBONE_RESUME_ROOT = '',  # the root to resume training from a saved checkpoint
        HEAD_RESUME_ROOT = '',      # the root to resume training from a saved checkpoint
        OPTIMIZER_RESUME_ROOT = '', # the root to resume training from a saved checkpoint
        
        BACKBONE_NAME = 'IR_50_ReStyle',    # should be used both with and without encoder checkpoint
        HEAD_NAME = 'ArcFace',              # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
        LOSS_NAME = 'Focal',
        
        ENCODER_CHECKPOINT = '<path to the encoder checkpoint after the Stage 2>',
        ENCODER_AVG_IMAGE = '<path to the avg image calculated during the Stage 2 training (see avg_image.jpg in the Stage 2 experiment folder)>',
        ENCODER_INPUT_SIZE = 112,
        ENCODER_ADDITIONAL_DROPOUT = 0.15,
        
        INPUT_SIZE = [112, 112],
        RGB_MEAN = [0.5, 0.5, 0.5],
        RGB_STD = [0.5, 0.5, 0.5],
        EMBEDDING_SIZE = 512,
        BATCH_SIZE = 100,    # in total for all gpus
        DROP_LAST = True, # whether drop the last batch to ensure consistent batch_norm statistics
        FREEZE_BACKBONE_EPOCHS = 3,

        # START_EPOCH = 5,   # set this parameter to resume training from a specific epoch
        # LIMIT_TRAIN_SAMPLES = 10000,    # set this parameter to limit the number of samples seen in each epoch (e.g. to run validation more frequently)

        LR = 0.03, # initial LR

        NUM_EPOCH = 100,        # total epoch number (use the firt 1/25 epochs to warm up)
        WEIGHT_DECAY = 2e-3,    # do not apply to batch_norm parameters
        MOMENTUM = 0.9,
        STAGES = np.array([10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120]) + 5,     # epoch stages to decay learning rate
        WARMUP = False,

        LAYER_DECAY = None,

        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        MULTI_GPU = True,       # flag to use multiple GPUs; if you choose to train with single GPU, you should first run "export CUDA_VISILE_DEVICES=device_id" to specify the GPU card you want to use
        GPU_ID = [0],     # specify your GPU ids
        PIN_MEMORY = True,
        NUM_WORKERS = 8,
),
}