from xml.etree.ElementInclude import include
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn

from .verification import evaluate

from datetime import datetime
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
from PIL import Image
import bcolz
import io
import os
from einops import rearrange
from tqdm import tqdm
from backbone.restyle_psp_helpers import AdaConv2d_faster

# Support: ['get_time', 'l2_norm', 'make_weights_for_balanced_classes', 'get_val_pair', 'get_val_data', 'separate_irse_bn_paras', 'separate_resnet_bn_paras', 'warm_up_lr', 'schedule_lr', 'de_preprocess', 'hflip_batch', 'ccrop_batch', 'gen_plot', 'perform_val', 'buffer_val', 'AverageMeter', 'accuracy']


def _initialize_weights(model):
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                # print('[initializing conv]')
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, AdaConv2d_faster):
                # print('[initializing adaconv]')
                nn.init.xavier_normal_(m.kernel_base)
                nn.init.xavier_normal_(m.kernel_mask)
                # nn.init.constant_(m.fuse_mark.data, val=-1)
            elif isinstance(m, nn.BatchNorm2d):
                # print('[initializing bn]')
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                # print('[initializing linear]')
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()


def get_time():
    return (str(datetime.now())[:-10]).replace(' ', '-').replace(':', '-')


def l2_norm(input, axis = 1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output


def make_weights_for_balanced_classes(images, nclasses):
    '''
        Make a vector of weights for each image in the dataset, based
        on class frequency. The returned vector of weights can be used
        to create a WeightedRandomSampler for a DataLoader to have
        class balancing when sampling for a training batch.
            images - torchvisionDataset.imgs
            nclasses - len(torchvisionDataset.classes)
        https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3
    '''
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1  # item is (img-data, label-id)
    weight_per_class = [0.] * nclasses
    N = float(sum(count))  # total number of images
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]

    return weight


def get_val_pair(path, name):
    carray = bcolz.carray(rootdir = os.path.join(path, name), mode = 'r')
    issame = np.load('{}/{}_list.npy'.format(path, name))

    return carray, issame


def get_val_data(data_path):

    lfw, lfw_issame = None, None
    # uncomment this line if you want to test on LFW dataset:
    # lfw, lfw_issame = get_val_pair(data_path, 'lfw')
    
    # replace "None, None" with the respective code piece on the right
    #  to also test on the these datasets
    cfp_ff, cfp_ff_issame = None, None      # get_val_pair(data_path, 'cfp_ff')
    cfp_fp, cfp_fp_issame = None, None      # get_val_pair(data_path, 'cfp_fp')
    agedb_30, agedb_30_issame = None, None  # get_val_pair(data_path, 'agedb_30')
    calfw, calfw_issame = None, None        # get_val_pair(data_path, 'calfw')
    calfw, calfw_issame = None, None        # get_val_pair(data_path, 'calfw')
    cplfw, cplfw_issame = None, None        # get_val_pair(data_path, 'cplfw')
    vgg2_fp, vgg2_fp_issame = None, None    # get_val_pair(data_path, 'vgg2_fp')

    # new test sets can be included here the same way. 
    #  The second parameter to get_val_pair(...) defines what the subfolder in DATA_ROOT should be called where the test set needs to be saved.
    #  Make sure to properly align and crop images in the datasets to 112x112 (see Readme).

    rfw, rfw_issame = dict(), dict()
    for ethnicity in ('African', 'Asian', 'Indian', 'Caucasian'):
        part, part_issame = get_val_pair(data_path, 'RFW_' + ethnicity)
        rfw[ethnicity] = part
        rfw_issame[ethnicity] = part_issame
    
    return lfw, cfp_ff, cfp_fp, agedb_30, calfw, cplfw, vgg2_fp, lfw_issame, cfp_ff_issame, cfp_fp_issame, agedb_30_issame, calfw_issame, cplfw_issame, vgg2_fp_issame, rfw, rfw_issame


def separate_irse_bn_paras(modules):
    if not isinstance(modules, list):
        # modules = [*modules.modules()]
        modules = [*modules.named_modules()]
    paras_only_bn = []
    paras_wo_bn = []
    # for layer in modules:
    for name, layer in modules:
        if 'model' in str(layer.__class__).lower():
            continue
        if 'container' in str(layer.__class__).lower():
            continue
        if 'backbone' in str(layer.__class__).lower():
            continue
        else:
            if 'batchnorm' in str(layer.__class__).lower():
                # print('FOUND batchnorm')
                paras_only_bn.extend([*layer.parameters()])
            else:
                paras_wo_bn.extend([*layer.parameters()])

    return paras_only_bn, paras_wo_bn


def separate_irse_bn_paras_with_occurs(modules, include_string=None, exclude_string=None):
    if not isinstance(modules, list):
        # modules = [*modules.modules()]
        modules = [*modules.named_modules()]
    paras_only_bn = []
    paras_wo_bn = []
    for name, layer in modules:
        if include_string is not None and include_string not in name:
            continue
        if exclude_string is not None and exclude_string in name:
            continue
        
        if 'model' in str(layer.__class__).lower():
            continue
        if 'container' in str(layer.__class__).lower():
            continue
        if 'backbone' in str(layer.__class__).lower():
            continue
        else:
            if 'batchnorm' in str(layer.__class__).lower():
                # print('FOUND batchnorm')
                paras_only_bn.extend([*layer.parameters()])
            else:
                paras_wo_bn.extend([*layer.parameters()])

    return paras_only_bn, paras_wo_bn


def separate_resnet_bn_paras(modules):
    all_parameters = modules.parameters()
    paras_only_bn = []

    for pname, p in modules.named_parameters():
        if pname.find('bn') >= 0:
            paras_only_bn.append(p)
            
    paras_only_bn_id = list(map(id, paras_only_bn))
    paras_wo_bn = list(filter(lambda p: id(p) not in paras_only_bn_id, all_parameters))
    
    return paras_only_bn, paras_wo_bn


def warm_up_lr(batch, num_batch_warm_up, init_lr, optimizer):
    for params in optimizer.param_groups:
        params['lr'] = batch * init_lr / num_batch_warm_up

    # print(optimizer)


def schedule_lr(optimizer):
    for params in optimizer.param_groups:
        # params['lr'] /= 10.
        params['lr'] /= 1.5    # NOTE: temporarily hardcoded!

    print(optimizer)


def de_preprocess(tensor):

    return tensor * 0.5 + 0.5


hflip = transforms.Compose([
            de_preprocess,
            transforms.ToPILImage(),
            transforms.functional.hflip,
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])


def hflip_batch(imgs_tensor):
    hfliped_imgs = torch.empty_like(imgs_tensor)
    for i, img_ten in enumerate(imgs_tensor):
        hfliped_imgs[i] = hflip(img_ten)

    return hfliped_imgs


ccrop = transforms.Compose([
    de_preprocess,
    transforms.ToPILImage(),
    transforms.Resize([128, 128]),  # smaller side resized
    transforms.CenterCrop([112, 112]),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


def ccrop_batch(imgs_tensor):
    ccropped_imgs = torch.empty_like(imgs_tensor)
    for i, img_ten in enumerate(imgs_tensor):
        ccropped_imgs[i] = ccrop(img_ten)

    return ccropped_imgs


def gen_plot(fpr, tpr):
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    plt.xlabel("FPR", fontsize = 14)
    plt.ylabel("TPR", fontsize = 14)
    plt.title("ROC Curve", fontsize = 14)
    plot = plt.plot(fpr, tpr, linewidth = 2)
    buf = io.BytesIO()
    plt.savefig(buf, format = 'jpeg')
    buf.seek(0)
    plt.close()

    return buf


def perform_val(multi_gpu, device, embedding_size, batch_size, backbone, carray, issame, nrof_folds = 10, tta = True, dset_name=None, ccrop=True):
    # print('carray:', carray)
    if multi_gpu:
        backbone = backbone.module # unpackage model from DataParallel
        backbone = backbone.to(device)
    else:
        backbone = backbone.to(device)
    backbone.eval() # switch to evaluation mode

    idx = 0
    embeddings = np.zeros([len(carray), embedding_size])

    pbar = tqdm(total=len(carray))
    with torch.no_grad():
        while idx + batch_size <= len(carray):
            # batch = torch.tensor(carray[idx:idx + batch_size][:, [2, 1, 0], :, :])
            batch = torch.tensor(carray[idx:idx + batch_size]).float()
            if batch.shape[-1] == 3:
                batch = rearrange(batch, 'b h w c -> b c h w')
            if tta:
                ccropped = ccrop_batch(batch) if ccrop else batch
                fliped = hflip_batch(ccropped)
                ccropf = backbone(ccropped.to(device))
                cflipf = backbone(fliped.to(device))
                emb_batch = ccropf.cpu() + cflipf.cpu()
                embeddings[idx:idx + batch_size] = l2_norm(emb_batch)
            else:
                ccropped = ccrop_batch(batch) if ccrop else batch
                embeddings[idx:idx + batch_size] = l2_norm(backbone(ccropped.to(device))).cpu()
            idx += batch_size
            pbar.update(batch_size)
        if idx < len(carray):
            batch = torch.tensor(carray[idx:]).float()
            if batch.shape[-1] == 3:
                batch = rearrange(batch, 'b h w c -> b c h w')
            if tta:
                ccropped = ccrop_batch(batch) if ccrop else batch
                fliped = hflip_batch(ccropped)
                ccropf = backbone(ccropped.to(device))
                cflipf = backbone(fliped.to(device))
                emb_batch = ccropf.cpu() + cflipf.cpu()
                embeddings[idx:] = l2_norm(emb_batch)
            else:
                ccropped = ccrop_batch(batch) if ccrop else batch
                embeddings[idx:] = l2_norm(backbone(ccropped.to(device))).cpu()
    
    pbar.close()

    tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds)
    buf = gen_plot(fpr, tpr)
    roc_curve = Image.open(buf)
    roc_curve_tensor = transforms.ToTensor()(roc_curve)

    return accuracy.mean(), best_thresholds.mean(), roc_curve_tensor


def buffer_val(writer, db_name, acc, best_threshold, roc_curve_tensor, epoch, n_samples_passed=None):
    # writer.add_scalar('{}_Accuracy'.format(db_name), acc, epoch)
    # writer.add_scalar('{}_Best_Threshold'.format(db_name), best_threshold, epoch)
    # writer.add_image('{}_ROC_Curve'.format(db_name), roc_curve_tensor, epoch)

    # wandb
    stats = {'{}_Accuracy'.format(db_name): acc,
             '{}_Best_Threshold'.format(db_name): best_threshold,
             'epoch': epoch}
    if n_samples_passed is not None:
        stats['step'] = n_samples_passed
    writer.log(stats) #, step=epoch)
    # writer.log('{}_ROC_Curve'.format(db_name), roc_curve_tensor, step=epoch)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val   = 0
        self.avg   = 0
        self.sum   = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val   = val
        self.sum   += val * n
        self.count += n
        self.avg   = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred    = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        # correct_k = correct[:k].view(-1).float().sum(0)
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res


def collate_fn_ignore_none(batch):
    len_batch = len(batch) # original batch length
    batch = list(filter (lambda x:x is not None, batch)) # filter out all the Nones
    if len_batch > len(batch): # if there are samples missing just use existing members, doesn't work if you reject every sample in a batch
        print('[collate] len_batch', len_batch, 'len(batch)', len(batch))
        diff = len_batch - len(batch)
        for i in range(diff):
            batch = batch + batch[:diff]
    return torch.utils.data.dataloader.default_collate(batch)


def apply_increasing_layer_decay(backbone, first_layer_lr=0.0):
    hooks = []

    n_weights = 0
    for param_name, _ in backbone.named_parameters():
        if param_name.endswith('.weight'):
            n_weights += 1
    
    print('n_weights:', n_weights)
    if n_weights == 0:
        return hooks
    
    cur_weight = 0
    for param_name, param in backbone.named_parameters():
        print('param_name:', param_name)
        if param_name.endswith('.weight'):
            cur_weight += 1
        
        if param_name.endswith('.weight') or param_name.endswith('.bias'):
            cur_lr_ratio = first_layer_lr + cur_weight / float(n_weights) * (1.0 - first_layer_lr)
            cur_hook = param.register_hook(lambda grad: grad * cur_lr_ratio)
            print(f'hook with {cur_lr_ratio} registered')
            hooks.append(cur_hook)
    
    return hooks


def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)
