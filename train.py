import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from backbone.model_resnet import ResNet_50, ResNet_101, ResNet_152
from backbone.model_irse import IR_50, IR_101, IR_152, IR_SE_50, IR_SE_101, IR_SE_152
from backbone.restyle_psp import pSp
from head.metrics import ArcFace, CosFace, SphereFace, Am_softmax
from loss.focal import FocalLoss
from util.utils import get_val_data, separate_irse_bn_paras, \
    separate_resnet_bn_paras, warm_up_lr, schedule_lr, perform_val, get_time, AverageMeter, accuracy, collate_fn_ignore_none, buffer_val
from dataset import FacesDataset

# from tensorboardX import SummaryWriter    # visualization is done in wandb in this code, but one could enable tensorboard as in face-evolve repo
import wandb
from tqdm import tqdm
import os


if __name__ == '__main__':
    import argparse
    import importlib

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.py')
    
    args = parser.parse_args()
    config_name = args.config
    config_name = config_name.replace('.py', '')
    config_name = config_name.replace('/', '.')

    print('config_name:', config_name)
    # exec(f'from {config_name} import configurations')
    config = importlib.import_module(config_name)
    configurations = config.configurations

    #======= hyperparameters & data loaders =======#
    cfg = configurations[1]

    SEED = cfg['SEED'] # random seed for reproduce results
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    DATA_ROOT = cfg['DATA_ROOT'] # the parent root where your train/val/test data are stored
    TRAIN_IMAGES_FOLDER = cfg['TRAIN_IMAGES_FOLDER']
    MODEL_ROOT = cfg['MODEL_ROOT'] # the root to buffer your checkpoints
    LOG_ROOT = cfg['LOG_ROOT'] # the root to log your train/val status
    BACKBONE_RESUME_ROOT = cfg['BACKBONE_RESUME_ROOT'] # the root to resume training from a saved checkpoint
    HEAD_RESUME_ROOT = cfg['HEAD_RESUME_ROOT']  # the root to resume training from a saved checkpoint
    OPTIMIZER_RESUME_ROOT = cfg.get('OPTIMIZER_RESUME_ROOT', None)  # the root to resume training from a saved checkpoint
    EXP_NAME = cfg['EXP_NAME']
    PROJECT_NAME = cfg.get('PROJECT_NAME', 'face-evolve')

    BACKBONE_NAME = cfg['BACKBONE_NAME'] # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
    HEAD_NAME = cfg['HEAD_NAME'] # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
    LOSS_NAME = cfg['LOSS_NAME'] # support: ['Focal', 'Softmax']
    OPTIMIZER_NAME = cfg.get('OPTIMIZER_NAME', 'SGD') # support: ['Focal', 'Softmax']
    OPTIMIZER_SWITCH = cfg.get('OPTIMIZER_SWITCH', None) # only makes sense for AdamSGD
    ARCFACE_S = cfg.get('ARCFACE_S', 64.0)
    START_EPOCH = cfg.get('START_EPOCH', 0)

    INPUT_SIZE = cfg['INPUT_SIZE']
    RGB_MEAN = cfg['RGB_MEAN'] # for normalize inputs
    RGB_STD = cfg['RGB_STD']
    EMBEDDING_SIZE = cfg['EMBEDDING_SIZE'] # feature dimension
    BATCH_SIZE = cfg['BATCH_SIZE']
    LIMIT_TRAIN_SAMPLES = cfg.get('LIMIT_TRAIN_SAMPLES', None)
    LIMIT_TRAIN_BATCHES = None if LIMIT_TRAIN_SAMPLES is None else LIMIT_TRAIN_SAMPLES // BATCH_SIZE
    DROP_LAST = cfg['DROP_LAST'] # whether drop the last batch to ensure consistent batch_norm statistics
    FREEZE_BACKBONE_EPOCHS = cfg.get('FREEZE_BACKBONE_EPOCHS', None)

    LR = cfg['LR'] # initial LR
    NUM_EPOCH = cfg['NUM_EPOCH']
    WEIGHT_DECAY = cfg['WEIGHT_DECAY']
    MOMENTUM = cfg['MOMENTUM']
    STAGES = cfg['STAGES'] # epoch stages to decay learning rate
    WARMUP = cfg.get('WARMUP', True)
    LAYER_DECAY = cfg.get('LAYER_DECAY', None)
    CCROP_AT_VAL = cfg.get('CCROP_AT_VAL', True)

    DEVICE = cfg['DEVICE']
    MULTI_GPU = cfg['MULTI_GPU'] # flag to use multiple GPUs
    GPU_ID = cfg['GPU_ID'] # specify your GPU ids
    PIN_MEMORY = cfg['PIN_MEMORY']
    NUM_WORKERS = cfg['NUM_WORKERS']

    ENCODER_CHECKPOINT = cfg.get('ENCODER_CHECKPOINT', None)
    ENCODER_AVG_IMAGE = cfg.get('ENCODER_AVG_IMAGE', None)
    ENCODER_INPUT_SIZE = cfg.get('ENCODER_INPUT_SIZE', 112)

    print("=" * 60)
    print("Overall Configurations:")
    print(cfg)
    print("=" * 60)

    # writer = SummaryWriter(LOG_ROOT) # writer for buffering intermedium results
    os.makedirs(MODEL_ROOT, exist_ok=True)
    os.makedirs(LOG_ROOT, exist_ok=True)

    wandb.init(project=PROJECT_NAME, config=cfg)
    wandb.run.name = EXP_NAME
    wandb.define_metric("train_loss", summary="min")
    wandb.define_metric("RFW_African_Accuracy", summary="max")
    wandb.define_metric("RFW_Asian_Accuracy", summary="max")
    wandb.define_metric("RFW_Caucasian_Accuracy", summary="max")
    wandb.define_metric("RFW_Indian_Accuracy", summary="max")

    train_transform = transforms.Compose([ # refer to https://pytorch.org/docs/stable/torchvision/transforms.html for more build-in online data augmentation
        transforms.Resize([int(128 * INPUT_SIZE[0] / 112), int(128 * INPUT_SIZE[0] / 112)]), # smaller side resized
        # transforms.Resize([int(160 * INPUT_SIZE[0] / 112), int(160 * INPUT_SIZE[0] / 112)]), # smaller side resized
        transforms.RandomCrop([INPUT_SIZE[0], INPUT_SIZE[1]]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = RGB_MEAN,
                             std = RGB_STD),
    ])

    # dataset_train = datasets.ImageFolder(os.path.join(DATA_ROOT, 'imgs'), train_transform)
    print('Initializing primary dataset...')
    dataset_train = FacesDataset(root=os.path.join(DATA_ROOT, TRAIN_IMAGES_FOLDER), 
                                    transform=train_transform)
    num_pictures_orig = len(dataset_train)
    print('Ready')

    NUM_CLASS = len(dataset_train.classes)
    print("Number of Training Classes: {}".format(NUM_CLASS))

    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size = BATCH_SIZE, sampler = None, pin_memory = PIN_MEMORY,
        num_workers = NUM_WORKERS, drop_last = DROP_LAST, shuffle = True, collate_fn = collate_fn_ignore_none
    )

    lfw, cfp_ff, cfp_fp, agedb, calfw, cplfw, vgg2_fp, lfw_issame, cfp_ff_issame, cfp_fp_issame, agedb_issame, calfw_issame, cplfw_issame, vgg2_fp_issame, rfw, rfw_issame = get_val_data(DATA_ROOT)


    #======= model & loss & optimizer =======#
    if BACKBONE_NAME == 'ResNet_50': 
        BACKBONE = ResNet_50(INPUT_SIZE)
    if BACKBONE_NAME == 'ResNet_101': 
        BACKBONE = ResNet_101(INPUT_SIZE)
    if BACKBONE_NAME == 'ResNet_152': 
        BACKBONE = ResNet_152(INPUT_SIZE)
    if BACKBONE_NAME == 'IR_50': 
        BACKBONE = IR_50(INPUT_SIZE)
    if BACKBONE_NAME == 'IR_101': 
        BACKBONE = IR_101(INPUT_SIZE)
    if BACKBONE_NAME == 'IR_152': 
        BACKBONE = IR_152(INPUT_SIZE)
    if BACKBONE_NAME == 'IR_SE_50': 
        BACKBONE = IR_SE_50(INPUT_SIZE)
    if BACKBONE_NAME == 'IR_SE_101': 
        BACKBONE = IR_SE_101(INPUT_SIZE)
    if BACKBONE_NAME == 'IR_SE_152':
        BACKBONE = IR_SE_152(INPUT_SIZE)
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
                       include_dropout=False)
    if BACKBONE_NAME == 'IR_100_ReStyle':
        BACKBONE = pSp(encoder_type='BackboneEncoder100',
                       size=ENCODER_INPUT_SIZE,
                       checkpoint_path=ENCODER_CHECKPOINT,
                       avg_image=ENCODER_AVG_IMAGE,
                       include_dropout=False )
    print("=" * 60)
    # print(BACKBONE)
    print("{} Backbone Generated".format(BACKBONE_NAME))
    print("=" * 60)

    HEAD_DICT = {'ArcFace': ArcFace(in_features = EMBEDDING_SIZE, out_features = NUM_CLASS, device_id = GPU_ID, s = ARCFACE_S),
                 'CosFace': CosFace(in_features = EMBEDDING_SIZE, out_features = NUM_CLASS, device_id = GPU_ID),
                 'SphereFace': SphereFace(in_features = EMBEDDING_SIZE, out_features = NUM_CLASS, device_id = GPU_ID),
                 'Am_softmax': Am_softmax(in_features = EMBEDDING_SIZE, out_features = NUM_CLASS, device_id = GPU_ID)}
    HEAD = HEAD_DICT[HEAD_NAME]
    print("=" * 60)
    # print(HEAD)
    print("{} Head Generated".format(HEAD_NAME))
    print("=" * 60)

    if BACKBONE_NAME.find("IR") >= 0:
        backbone_paras_only_bn, backbone_paras_wo_bn = separate_irse_bn_paras(BACKBONE) # separate batch_norm parameters from others; do not do weight decay for batch_norm parameters to improve the generalizability
        _, head_paras_wo_bn = separate_irse_bn_paras(HEAD)
    else:
        backbone_paras_only_bn, backbone_paras_wo_bn = separate_resnet_bn_paras(BACKBONE) # separate batch_norm parameters from others; do not do weight decay for batch_norm parameters to improve the generalizability
        _, head_paras_wo_bn = separate_resnet_bn_paras(HEAD)

    if OPTIMIZER_NAME == 'SGD':
        OPTIMIZER = optim.SGD([{'params': backbone_paras_wo_bn + head_paras_wo_bn, 'weight_decay': WEIGHT_DECAY}, {'params': backbone_paras_only_bn}], lr = LR, momentum = MOMENTUM)
    elif OPTIMIZER_NAME == 'Adam':
        OPTIMIZER = optim.Adam([{'params': backbone_paras_only_bn + backbone_paras_wo_bn + head_paras_wo_bn}], lr=LR)

    print("=" * 60)
    print(OPTIMIZER)
    print("Optimizer Generated")
    print("=" * 60)

    # optionally resume from a checkpoint
    if BACKBONE_RESUME_ROOT and HEAD_RESUME_ROOT:
        print("=" * 60)
        if os.path.isfile(BACKBONE_RESUME_ROOT) and os.path.isfile(HEAD_RESUME_ROOT):
            print("Loading Backbone Checkpoint '{}'".format(BACKBONE_RESUME_ROOT))
            BACKBONE.load_state_dict(torch.load(BACKBONE_RESUME_ROOT))
            print("Loading Head Checkpoint '{}'".format(HEAD_RESUME_ROOT))
            HEAD.load_state_dict(torch.load(HEAD_RESUME_ROOT))
        else:
            print("No Checkpoint Found at '{}' and '{}'. Please Have a Check or Continue to Train from Scratch".format(BACKBONE_RESUME_ROOT, HEAD_RESUME_ROOT))
        print("=" * 60)

    # optimizer_to(OPTIMIZER, DEVICE)

    if MULTI_GPU:
        # multi-GPU setting
        BACKBONE = nn.DataParallel(BACKBONE, device_ids = GPU_ID)
        BACKBONE = BACKBONE.to(DEVICE)
    else:
        # single-GPU setting
        BACKBONE = BACKBONE.to(DEVICE)
    
    if OPTIMIZER_RESUME_ROOT:
        if os.path.isfile(OPTIMIZER_RESUME_ROOT):
            print("Loading Optimizer Checkpoint '{}'".format(OPTIMIZER_RESUME_ROOT))
            OPTIMIZER.load_state_dict(torch.load(OPTIMIZER_RESUME_ROOT))
        else:
            print("No Checkpoint Found at '{}'. Please Have a Check or Continue to Train from Scratch".format(OPTIMIZER_RESUME_ROOT))

        #TODO get rid of LOSS_DICT
    if LOSS_NAME == 'Focal':
        LOSS = FocalLoss()
    elif LOSS_NAME == 'Softmax':
        LOSS = nn.CrossEntropyLoss()
    
    print("=" * 60)
    print(LOSS)
    print("{} Loss Generated".format(LOSS_NAME))
    print("=" * 60)

    #======= train & validation & save checkpoint =======#
    # DISP_FREQ = len(train_loader) // 100 # frequency to display training loss & acc
    DISP_FREQ = len(train_loader) // 10 # frequency to display training loss & acc
    # DISP_FREQ = len(train_loader) # frequency to display training loss & acc

    NUM_EPOCH_WARM_UP = NUM_EPOCH // 25  # use the first 1/25 epochs to warm up
    NUM_BATCH_WARM_UP = len(train_loader) * NUM_EPOCH_WARM_UP  # use the first 1/25 epochs to warm up
    batch = 0  # batch index

    for epoch in range(START_EPOCH, NUM_EPOCH):
                
        if epoch in STAGES:
            print('epoch in STAGES, changing lr')
            schedule_lr(OPTIMIZER)
        
        BACKBONE.train()
        HEAD.train()

        if FREEZE_BACKBONE_EPOCHS is not None:
            if epoch <= FREEZE_BACKBONE_EPOCHS:
                print('freezing backbone; optimizing head only')
                BACKBONE.module.encoder.input_layer.requires_grad_(True)
                BACKBONE.module.encoder.body.requires_grad_(False)
                BACKBONE.module.encoder.output_layer.requires_grad_(True)

            else:
                print('optimizing both backbone & head')
                BACKBONE.module.encoder.input_layer.requires_grad_(True)
                BACKBONE.module.encoder.body.requires_grad_(True)
                BACKBONE.module.encoder.output_layer.requires_grad_(True)

        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        if LIMIT_TRAIN_BATCHES is None:
            progress_bar = tqdm(iter(train_loader), total=len(train_loader))
        else:
            progress_bar = tqdm(iter(train_loader), total=LIMIT_TRAIN_BATCHES)
        
        batch_in_epoch = 0

        for inputs, labels in progress_bar:

            if WARMUP and (epoch + 1 <= NUM_EPOCH_WARM_UP) and (batch + 1 <= NUM_BATCH_WARM_UP):
                warm_up_lr(batch, NUM_BATCH_WARM_UP, LR, OPTIMIZER)
            
            # compute output
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE).long()
            
            features = BACKBONE(inputs)

            outputs = HEAD(features, labels)

            if LOSS_NAME == 'Softmax':
                loss = LOSS(outputs, labels)
                loss_components = None
            else:
                loss, loss_components = LOSS(outputs, labels)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, labels, topk = (1, 5))
            losses.update(loss.data.item(), inputs.size(0))
            top1.update(prec1.data.item(), inputs.size(0))
            top5.update(prec5.data.item(), inputs.size(0))

            # compute gradient and do SGD step
            OPTIMIZER.zero_grad()
            loss.backward()

            OPTIMIZER.step()
            
            # dispaly training loss & acc every DISP_FREQ
            if ((batch + 1) % DISP_FREQ == 0) and batch != 0:
                print("=" * 60)
                print('Epoch {}/{} Batch {}/{}\t'
                      'Training Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Training Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Training Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch + 1, NUM_EPOCH, batch + 1, len(train_loader) * NUM_EPOCH, loss = losses, top1 = top1, top5 = top5))
                print("=" * 60)
            
            # display train loss every certain number of steps
            if batch % 10 == 0:
                wandb.log({"train_loss": loss.item(),
                           "step": batch * BATCH_SIZE})

            # visualize input images
            # if batch % 10 == 0:
            #     inputs_to_log = make_grid(inputs[:50]) * 0.5 + 0.5
            #     inputs_to_log = (inputs_to_log.cpu().data.numpy() * 255).astype(np.uint8)
            #     inputs_to_log = rearrange(inputs_to_log, 'c h w -> h w c')
            #     wandb.log({"img_to_check": wandb.Image(inputs_to_log)
            #     })

            batch += 1 # batch index
            batch_in_epoch += 1

            if LIMIT_TRAIN_BATCHES is not None and batch_in_epoch >= LIMIT_TRAIN_BATCHES:
                break

        # training statistics per epoch
        epoch_loss = losses.avg
        epoch_acc = top1.avg
        epoch_acc_top5 = top5.avg
        wandb.log({"train_loss_ep": epoch_loss,
                   "train_acc_ep": epoch_acc,
                   "train_acc_top5_ep": epoch_acc_top5,
                   "step": batch * BATCH_SIZE,
                   "epoch": epoch + 1})
        print("=" * 60)
        print('Epoch: {}/{}\t'
              'Training Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Training Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
              'Training Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
            epoch + 1, NUM_EPOCH, loss = losses, top1 = top1, top5 = top5))
        print("=" * 60)

        # perform validation & save checkpoints per epoch
        print("=" * 60)
        print("Performing evaluation and saving checkpoints...")

        if lfw is not None:
            accuracy_lfw, best_threshold_lfw, roc_curve_lfw = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, lfw, lfw_issame)
            buffer_val(wandb, "LFW", accuracy_lfw, best_threshold_lfw, roc_curve_lfw, epoch + 1)
            print('LFW accuracy:', accuracy_lfw)

        if cfp_ff is not None:
            accuracy_cfp_ff, best_threshold_cfp_ff, roc_curve_cfp_ff = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, cfp_ff, cfp_ff_issame)
            buffer_val(wandb, "CFP_FF", accuracy_cfp_ff, best_threshold_cfp_ff, roc_curve_cfp_ff, epoch + 1)
            print('CFP_FF accuracy:', accuracy_cfp_ff)

        if cfp_fp is not None:
            accuracy_cfp_fp, best_threshold_cfp_fp, roc_curve_cfp_fp = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, cfp_fp, cfp_fp_issame)
            buffer_val(wandb, "CFP_FP", accuracy_cfp_fp, best_threshold_cfp_fp, roc_curve_cfp_fp, epoch + 1)
            print('CFP_FP accuracy:', accuracy_cfp_fp)

        if agedb is not None:
            accuracy_agedb, best_threshold_agedb, roc_curve_agedb = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, agedb, agedb_issame)
            buffer_val(wandb, "AgeDB", accuracy_agedb, best_threshold_agedb, roc_curve_agedb, epoch + 1)
            print('AgeDB accuracy:', accuracy_agedb)

        if calfw is not None:
            accuracy_calfw, best_threshold_calfw, roc_curve_calfw = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, calfw, calfw_issame)
            buffer_val(wandb, "CALFW", accuracy_calfw, best_threshold_calfw, roc_curve_calfw, epoch + 1)
            print('CALFW accuracy:', accuracy_calfw)

        if cplfw is not None:
            accuracy_cplfw, best_threshold_cplfw, roc_curve_cplfw = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, cplfw, cplfw_issame)
            buffer_val(wandb, "CPLFW", accuracy_cplfw, best_threshold_cplfw, roc_curve_cplfw, epoch + 1)
            print('CPLFW accuracy:', accuracy_cplfw)
        
        if vgg2_fp is not None:
            accuracy_vgg2_fp, best_threshold_vgg2_fp, roc_curve_vgg2_fp = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, vgg2_fp, vgg2_fp_issame, dset_name="VGGFace2")
            buffer_val(wandb, "VGGFace2_FP", accuracy_vgg2_fp, best_threshold_vgg2_fp, roc_curve_vgg2_fp, epoch + 1)
            print('VGG2_FP accuracy:', accuracy_vgg2_fp)

        if rfw is not None:        
            print("=" * 60)
            for ethnicity in ('African', 'Asian', 'Caucasian', 'Indian'):
                accuracy_rfw_part, best_threshold_rfw_part, roc_curve_rfw_part = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, rfw[ethnicity], rfw_issame[ethnicity], dset_name="RFW_" + ethnicity, ccrop=CCROP_AT_VAL)
                buffer_val(wandb, "RFW_" + ethnicity, accuracy_rfw_part, best_threshold_rfw_part, roc_curve_rfw_part, epoch + 1, batch * BATCH_SIZE)

                print(f"Evaluation: RFW {ethnicity} Acc: {accuracy_rfw_part}")
            print("=" * 60)

        # save checkpoints per epoch
        print('Saving checkpoint...')
        if MULTI_GPU:
            torch.save(BACKBONE.module.state_dict(), os.path.join(MODEL_ROOT, "Backbone_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(BACKBONE_NAME, epoch + 1, batch, get_time())))
            torch.save(HEAD.state_dict(), os.path.join(MODEL_ROOT, "Head_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(HEAD_NAME, epoch + 1, batch, get_time())))
            torch.save(OPTIMIZER.state_dict(), os.path.join(MODEL_ROOT, "Optimizer_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(HEAD_NAME, epoch + 1, batch, get_time())))
        else:
            torch.save(BACKBONE.state_dict(), os.path.join(MODEL_ROOT, "Backbone_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(BACKBONE_NAME, epoch + 1, batch, get_time())))
            torch.save(HEAD.state_dict(), os.path.join(MODEL_ROOT, "Head_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(HEAD_NAME, epoch + 1, batch, get_time())))
            torch.save(OPTIMIZER.state_dict(), os.path.join(MODEL_ROOT, "Optimizer_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(HEAD_NAME, epoch + 1, batch, get_time())))
