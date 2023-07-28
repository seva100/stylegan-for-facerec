# -*- coding: utf-8 -*-
"""
Created on 18-5-21 下午5:26

"""
import sys
import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.utils.weight_norm as weight_norm
import torch.nn.functional as F

import pdb

from models.encoders.map2style import GradualStyleBlock


__all__ = ['gac_resnet18', 'gac_resnet34', 'gac_resnet50', 'gac_resnet100', 'gac_resnet152']


def conv3x3(ndemog, in_planes, out_planes, stride=1, adap=False, fuse_epoch=9):
    """3x3 convolution with padding"""
    return AdaConv2d(ndemog, in_planes, out_planes, 3, stride,
                     padding=1, adap=adap, fuse_epoch=fuse_epoch)
    # return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
    #                  padding=1, bias=False)

class IRBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, height=None, width=None, 
        downsample=None, use_se=False, use_att=False, use_spatial_att=False,
        ndemog=4, hard_att_channel=False, hard_att_spatial=False, lowresol_set={},
        adap=False, fuse_epoch=9, att_mock=False):
        super(IRBlock, self).__init__()
        self.bn0 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(ndemog, inplanes, planes, stride, adap=adap, fuse_epoch=fuse_epoch)
        self.bn1 = nn.BatchNorm2d(planes)
        self.prelu1 = nn.PReLU(num_parameters=planes)
        self.conv2 = conv3x3(ndemog, planes, planes, adap=adap, fuse_epoch=fuse_epoch)
        self.bn2 = nn.BatchNorm2d(planes)
        self.prelu2 = nn.PReLU(num_parameters=planes)
        self.downsample = downsample
        self.stride = stride
        self.use_se = use_se
        self.use_att = use_att
        self.att_mock = att_mock
        if self.use_se:
            self.se = SEBlock(planes)
        if self.use_att:
            self.att = AttBlock(planes, height, width, ndemog, use_spatial_att,
                hard_att_channel, hard_att_spatial, lowresol_set, self.att_mock)

    def forward(self, x_dict):
        x = x_dict['x']
        demog_label = x_dict['demog_label']
        epoch = x_dict['epoch']
        attc = None
        atts = None

        residual = x
        out = self.bn0(x)
        out = self.conv1(out, demog_label, epoch)
        out = self.bn1(out)
        out = self.prelu1(out)

        out = self.conv2(out, demog_label, epoch)
        out = self.bn2(out)
        if self.use_se:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.prelu2(out)

        if self.use_att:
            out,attc,atts = self.att(out, demog_label)

        return {'x':out, 'demog_label':demog_label, 'epoch': epoch, 'attc':attc, 'atts':atts}

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.PReLU(),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class AttBlock(nn.Module): # add more options, e.g, hard attention, low resolution attention
    def __init__(self, nchannel, height, width, ndemog=4, use_spatial_att=False,
        hard_att_channel=False, hard_att_spatial=False, lowersol_set={}, att_mock=False):
        super(AttBlock, self).__init__()
        self.ndemog = ndemog

        self.hard_att_channel = hard_att_channel
        self.hard_att_spatial = hard_att_spatial
        
        self.lowersol_mode = lowersol_set['mode']
        lowersol_rate = lowersol_set['rate']

        self.att_channel = nn.parameter.Parameter(torch.Tensor(1, 1, nchannel, 1, 1))
        nn.init.xavier_uniform_(self.att_channel)
        self.att_channel.data = self.att_channel.data.repeat(ndemog,1,1,1,1)

        self.use_spatial_att = use_spatial_att
        if use_spatial_att:
            self.height = int(height)
            self.width = int(width)
            self.att_spatial = nn.parameter.Parameter(torch.Tensor(ndemog, 1, 1, 
                int(height*lowersol_rate), int(width*lowersol_rate)))
            nn.init.xavier_uniform_(self.att_spatial)
        else:
            self.att_spatial = None
        
        self.att_mock = att_mock

    def forward(self, x, demog_label):
        y = x
        demogs = list(set(demog_label.tolist()))
        # demogs = list(range(self.ndemog))

        if self.hard_att_channel:
            att_channel = torch.where(torch.sigmoid(self.att_channel) >= 0.5, 
                torch.ones_like(self.att_channel), torch.zeros_like(self.att_channel))
        else:
            att_channel = torch.sigmoid(self.att_channel)

        if self.use_spatial_att:
            if self.hard_att_spatial:
                att_spatial = torch.where(torch.sigmoid(self.att_spatial) >= 0.5, 
                    torch.ones_like(self.att_spatial), torch.zeros_like(self.att_spatial))
            else:
                att_spatial = torch.sigmoid(self.att_spatial)
            att_spatial = F.interpolate(att_spatial, size=(att_spatial.size(2), 
                self.height,self.width), mode=self.lowersol_mode)
        else:
            att_spatial = None

        if self.use_spatial_att:
            for demog in demogs:
                indices = torch.nonzero((demog_label==demog), as_tuple=False).squeeze()
                if indices.dim() == 0:
                    indices = indices.unsqueeze(0)
                y[indices,:,:,:] = x[indices,:,:,:] *\
                    att_channel.repeat(1, indices.size(0), 1, x.size(2), x.size(3))[demog,:,:,:,:] * \
                    att_spatial.repeat(1, indices.size(0), x.size(1), 1, 1)[demog,:,:,:,:]
        elif not self.att_mock:
            for demog in demogs:
                indices = torch.nonzero((demog_label==demog), as_tuple=False).squeeze()
                if indices.dim() == 0:
                    indices = indices.unsqueeze(0)
                # y[indices,:,:,:] = x[indices,:,:,:] *\
                #     att_channel.repeat(1, indices.size(0), 1, x.size(2), x.size(3))[demog,:,:,:,:]
                x_clone = x[indices,:,:,:].clone()
                attr_channel_clone = att_channel.repeat(1, indices.size(0), 1, x.size(2), x.size(3))[demog,:,:,:,:].clone()
                y[indices,:,:,:] = x_clone * attr_channel_clone

        # print('self.att_mock:', self.att_mock)
        # sys.stdout.flush()
        # if self.att_mock == True, no attention will be applied
        return y, att_channel, att_spatial


class AdaConv2d_old(nn.Module):
    def __init__(self, ndemog, ic, oc, ks, stride, padding=0, adap=True, fuse_epoch=9):
        super(AdaConv2d, self).__init__()
        self.stride = stride
        self.padding = padding
        self.fuse_epoch = fuse_epoch

        self.oc = oc
        self.ic = ic
        self.ks = ks
        self.ndemog = ndemog
        self.adap = adap
        # self.kernel_adap = nn.Parameter(torch.Tensor(ndemog, oc, ic, ks, ks))
        self.kernel_base = nn.Parameter(torch.Tensor(oc, ic, ks, ks))
        # self.kernel_net = nn.Linear(ndemog, oc*ic*ks*ks)
        self.kernel_mask = nn.Parameter(torch.Tensor(1, ic, ks, ks))
        # self.fuse_mark = nn.Parameter(torch.zeros(1))
        self.fuse_mark = nn.Parameter(torch.Tensor(1))

        # self.conv = nn.Conv2d(ic, oc, kernel_size=3, stride=stride,
        #              padding=1, bias=False)
        # self.conv.weight = self.kernel_base

        if adap:
            self.kernel_mask.data = self.kernel_mask.data.repeat(ndemog,1,1,1)

    def forward(self, x, demog_label, epoch):
        demogs = list(range(self.ndemog))

        if self.adap:
            for i,demog in enumerate(demogs):
                # get indices
                indices = torch.nonzero((demog_label==demog), as_tuple=False).squeeze()
                if indices.dim() == 0:
                    indices = indices.unsqueeze(0)

                # k_input = F.one_hot(demog_label, num_classes=self.ndemog)
                
                # get mask
                if epoch >= self.fuse_epoch:
                    if self.fuse_mark[0] == -1:
                        mask = self.kernel_mask[0,:,:,:].unsqueeze(0)
                        # kernel_mask = self.kernel_adap[0,:,:,:,:]
                    else:
                        mask = self.kernel_mask[demog,:,:,:].unsqueeze(0)
                        # kernel_mask = self.kernel_adap[demog,:,:,:,:]
                else:
                    mask = self.kernel_mask[demog,:,:,:].unsqueeze(0)
                    # kernel_mask = self.kernel_adap[demog,:,:,:,:]
                # if epoch >= self.fuse_epoch:
                #     if self.fuse_mark[0] == -1:
                #         kernel_mask = self.kernel_adap[0,:,:,:,:]
                #         # k_input = F.one_hot([0], num_classes=self.ndemog)
                #     else:
                #         kernel_mask = self.kernel_adap[demog,:,:,:,:]
                #         # k_input = F.one_hot(demog_label, num_classes=self.ndemog)
                # else:
                #     kernel_mask = self.kernel_adap[demog,:,:,:,:]
                #     # k_input = F.one_hot(demog_label, num_classes=self.ndemog)
                # # kernel_mask = self.kernel_net(k_input)

                # get output
                if i == 0:
                    temp = F.conv2d(x[indices,:,:,:], self.kernel_base*mask.repeat(self.oc,1,1,1),
                        stride=self.stride, padding=self.padding)
                    # temp = self.conv(x[indices,:,:,:])
                    # initialize output
                    size = [x.size(0)]
                    for i in range(1,temp.dim()):
                        size.append(temp.size(i))
                    output = torch.zeros(size)
                    if x.is_cuda:
                        output = output.cuda()                    
                    output[indices,:,:,:] = temp
                else:
                    output[indices,:,:,:] = F.conv2d(x[indices,:,:,:], 
                        self.kernel_base*mask.repeat(self.oc,1,1,1),
                        stride=self.stride, padding=self.padding)
                    # output[indices,:,:,:] = self.conv(x[indices,:,:,:])
        else:
            print('[adap=False]')
            sys.stdout.flush()
            output = F.conv2d(x, self.kernel_base, stride=self.stride, padding=self.padding)
            # output = self.conv(x)
            # k_input = F.one_hot(torch.tensor([0]), num_classes=self.ndemog)
            # kernel_mask = self.kernel_net(k_input.float())
            # kernel_mask = kernel_mask.view(self.oc, self.ic, self.ks, self.ks)
            # if x.is_cuda:
            #     kernel_mask = kernel_mask.cuda()

        return output


class AdaConv2d(nn.Module):
    def __init__(self, ndemog, ic, oc, ks, stride, padding=0, adap=True, fuse_epoch=9):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.fuse_epoch = fuse_epoch

        self.oc = oc
        self.ic = ic
        self.ks = ks
        self.ndemog = ndemog
        self.adap = adap
        # self.kernel_adap = nn.Parameter(torch.Tensor(ndemog, oc, ic, ks, ks))
        self.kernel_base = nn.Parameter(torch.Tensor(oc, ic, ks, ks))
        # self.kernel_net = nn.Linear(ndemog, oc*ic*ks*ks)
        self.kernel_mask = nn.Parameter(torch.Tensor(1, ic, ks, ks))

        self.fuse_mark = nn.Parameter(torch.Tensor(1))
        self.fuse_mark.data[0] = -1

        # self.conv = nn.Conv2d(ic, oc, kernel_size=3, stride=stride,
        #              padding=1, bias=False)
        # self.conv.weight = self.kernel_base

        # if adap:
        self.kernel_mask.data = self.kernel_mask.data.repeat(ndemog,1,1,1)

    def forward(self, x, demog_label, epoch):
        demogs = list(range(self.ndemog))

        if self.adap:
            
            # ----------------------------------------------------------
            # version 1 -- same speed (2.5h)
            # oh, ow = conv_output_shape((x.shape[2], x.shape[3]), self.ks, self.stride, self.padding)
            # output = torch.zeros((x.size(0), self.oc, oh, ow)).to(x.device)

            # kernel_comb = (self.kernel_base.unsqueeze(1) # (oc, 1, ic, ks, ks)
            #                * self.kernel_mask.unsqueeze(0).repeat(self.oc, 1, 1, 1, 1)  # (oc, ndemog, ic, ks, ks)
            # )  # (oc, ndemog, ic, ks, ks)

            # for i,demog in enumerate(demogs):
            #     # get indices
            #     indices = torch.nonzero((demog_label==demog), as_tuple=False).squeeze()
            #     if indices.dim() == 0:
            #         indices = indices.unsqueeze(0)
            #     comb = kernel_comb[:, i]
            #     output[indices,:,:,:] = F.conv2d(x[indices,:,:,:], 
            #         comb,
            #         stride=self.stride, padding=self.padding)

            # oh, ow = conv_output_shape((x.shape[2], x.shape[3]), self.ks, self.stride, self.padding)
            # ----------------------------------------------------------

            # ----------------------------------------------------------
            # version 2 -- much faster (1h)
            kernel_comb = (self.kernel_base.unsqueeze(1) # (oc, 1, ic, ks, ks)
                           * self.kernel_mask.unsqueeze(0).repeat(self.oc, 1, 1, 1, 1)  # (oc, ndemog, ic, ks, ks)
            )  # (oc, ndemog, ic, ks, ks)

            output = F.conv2d(x, 
                              kernel_comb[:, 0],
                              stride=self.stride, padding=self.padding)

            for i in range(self.ndemog):
                # get indices
                if i >= 1 and torch.any(demog_label == i):
                    # indices = torch.nonzero((demog_label==demog), as_tuple=False).squeeze()
                    # if indices.dim() == 0:
                    #     indices = indices.unsqueeze(0)
                    if epoch >= self.fuse_epoch:
                        if self.fuse_mark[0] == -1:
                            comb = kernel_comb[:, 0]
                        else:
                            comb = kernel_comb[:, i]
                    # output[indices,:,:,:] = F.conv2d(x[indices,:,:,:], 
                    #     comb,
                    #     stride=self.stride, padding=self.padding)
                    output[demog_label == i] = F.conv2d(x[demog_label == i].contiguous(), 
                        comb, stride=self.stride, padding=self.padding).contiguous()
                        # output[indices,:,:,:] = self.conv(x[indices,:,:,:])

            # # output[demog_label == 0] = F.conv2d(x[demog_label == 0], 
            # #     kernel_comb[:, 0], stride=self.stride, padding=self.padding)
            # output[demog_label == 1] = F.conv2d(x[demog_label == 1], 
            #     kernel_comb[:, 1], stride=self.stride, padding=self.padding)
            # output[demog_label == 2] = F.conv2d(x[demog_label == 2], 
            #     kernel_comb[:, 2], stride=self.stride, padding=self.padding)
            # output[demog_label == 3] = F.conv2d(x[demog_label == 3], 
            #     kernel_comb[:, 3], stride=self.stride, padding=self.padding)

            # ----------------------------------------------------------

            # ----------------------------------------------------------
            # version 3 -- 1.05h
            # kernel_comb = (self.kernel_base.unsqueeze(1) # (oc, 1, ic, ks, ks)
            #                * self.kernel_mask.unsqueeze(0).repeat(self.oc, 1, 1, 1, 1)  # (oc, ndemog, ic, ks, ks)
            # )  # (oc, ndemog, ic, ks, ks)

            # output = []
            # perm = []

            # for i,demog in enumerate(demogs):
            #     # get indices
            #     indices = torch.nonzero((demog_label==demog), as_tuple=False).squeeze()
            #     if indices.dim() == 0:
            #         indices = indices.unsqueeze(0)
            #     perm.append(indices)
            #     comb = kernel_comb[:, i]
            #     output_partial = F.conv2d(x[indices,:,:,:], 
            #         comb,
            #         stride=self.stride, padding=self.padding)
            #         # output[indices,:,:,:] = self.conv(x[indices,:,:,:])
            #     output.append(output_partial)
            
            # output = torch.cat(output, dim=0)
            # perm = torch.cat(perm, dim=0)
            # # print('perm:', perm)
            # # print('perm.size(0):', perm.size(0))
            # inv_perm = inverse_permutation(perm)
            # # print('inv_perm', inv_perm)
            # output = output[inv_perm]

            # ----------------------------------------------------------
        else:
            # print('no adaconv [adap=False]')
            # sys.stdout.flush()
            output = F.conv2d(x, self.kernel_base, stride=self.stride, padding=self.padding)
        
        # # mock:
        # output = F.conv2d(x, self.kernel_base, stride=self.stride, padding=self.padding)
            
        return output


class ResNetFace(nn.Module):
    def __init__(self, block, layers, **kwargs,
        ):
        # use_se=False, use_spatial_att=False, ndemog=4, nclasses=2,
        self.inplanes = 64
        self.use_se = kwargs['use_se']
        self.use_spatial_att = kwargs['use_spatial_att']
        self.ndemog = kwargs['ndemog']
        self.hard_att_channel = kwargs['hard_att_channel']
        self.hard_att_spatial = kwargs['hard_att_spatial']
        self.lowresol_set = kwargs['lowresol_set']
        self.fuse_epoch = kwargs['fuse_epoch']
        self.n_styles = kwargs['n_styles']
        self.adap = kwargs['adap']
        self.att_mock = kwargs['att_mock']

        super(ResNetFace, self).__init__()
        # self.attinput = AttBlock(3, 112, 112, self.ndemog, self.use_spatial_att,
        #     self.hard_att_channel, self.hard_att_spatial, self.lowresol_set)
        self.attinput = AttBlock(6, 112, 112, self.ndemog, self.use_spatial_att,
            self.hard_att_channel, self.hard_att_spatial, self.lowresol_set, self.att_mock)

        # self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.conv1 = nn.Conv2d(6, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU(64)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.attconv1 = AttBlock(64, 56, 56, self.ndemog, self.use_spatial_att,
            self.hard_att_channel, self.hard_att_spatial, self.lowresol_set, self.att_mock)
        
        self.layer1 = self._make_layer(block, 64, layers[0], height=56, width=56)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, height=28, width=28)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, height=14, width=14)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.bn4 = nn.BatchNorm2d(512)
        self.attbn4 = AttBlock(512, 7, 7, self.ndemog, self.use_spatial_att,
            self.hard_att_channel, self.hard_att_spatial, self.lowresol_set, self.att_mock)
        
        # self.dropout = nn.Dropout(p=0.4)
        # self.fc5 = nn.Linear(512 * 7 * 7, 512)
        # self.bn5 = nn.BatchNorm1d(512)

        # pSp output layers
        self.styles = nn.ModuleList()
        self.style_count = self.n_styles
        for i in range(self.style_count):
            style = GradualStyleBlock(512, 512, 16)
            self.styles.append(style)

        for m in self.modules():
            if isinstance(m, AdaConv2d):
                nn.init.xavier_normal_(m.kernel_base)
                nn.init.xavier_normal_(m.kernel_mask)
                nn.init.constant_(m.fuse_mark.data, val=-1)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, height=None, width=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, height, width,
            downsample, self.use_se, False, self.use_spatial_att,
            self.ndemog, self.hard_att_channel, self.hard_att_spatial, self.lowresol_set, 
            self.adap, self.fuse_epoch, self.att_mock))

        if height != None and width != None:
            use_att = True
        else:
            use_att = False

        self.inplanes = planes
        for i in range(1, blocks):
            if i == blocks-1:
                layers.append(block(self.inplanes, planes, 1, height, width,
                    None, self.use_se, use_att, self.use_spatial_att,
                    self.ndemog, self.hard_att_channel, self.hard_att_spatial, self.lowresol_set, 
                    self.adap, self.fuse_epoch, self.att_mock))
            else:
                layers.append(block(self.inplanes, planes, 1, height, width, 
                    None, self.use_se, False, self.use_spatial_att,
                    self.ndemog, self.hard_att_channel, self.hard_att_spatial, self.lowresol_set, 
                    self.adap, self.fuse_epoch, self.att_mock))
        
        return nn.Sequential(*layers)

    # def forward(self, inputs, epoch):
    def forward(self, x, demog_label, epoch):
        # x = inputs[0]
        # demog_label = inputs[1]

        x,attc1,atts1 = self.attinput(x,demog_label)
        
        x = self.conv1(x) # 3 x 112 x 112
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.maxpool(x) # 64 x 56 x 56
        x,attc2,atts2 = self.attconv1(x,demog_label)

        x_dict = self.layer1({'x':x, 'demog_label':demog_label, 'epoch':epoch}) # 64 x 56 x 56
        attc3 = x_dict['attc']
        atts3 = x_dict['atts']

        x_dict = self.layer2(x_dict) # 128 x 28 x 28
        attc4 = x_dict['attc']
        atts4 = x_dict['atts']

        x_dict = self.layer3(x_dict) # 256 x 14 x 14
        attc5 = x_dict['attc']
        atts5 = x_dict['atts']

        x_dict = self.layer4(x_dict) # 512 x 7 x 7

        x = x_dict['x']
        x = self.bn4(x)
        x,attc6,atts6 = self.attbn4(x, demog_label)
        
        # x = self.dropout(x)
        # x = x.view(x.size(0), -1) # 1 x 25088
        # x = self.fc5(x) # 1 x 512
        # x = self.bn5(x)

        # attc = [attc1, attc2, attc3, attc4, attc5, attc6]
        # atts = [atts1, atts2, atts3, atts4, atts5, atts6]

        # return x,{'attc':attc,'atts':atts}

        latents = []
        for j in range(self.style_count):
            latents.append(self.styles[j](x))
        out = torch.stack(latents, dim=1)

        return out

def gac_resnet18(use_se=False, **kwargs):
    model = ResNetFace(IRBlock, [2, 2, 2, 2], use_se=use_se, **kwargs)
    return model

def gac_resnet34(use_se=False, **kwargs):
    model = ResNetFace(IRBlock, [3, 4, 6, 3], use_se=use_se, **kwargs)
    return model

# def gac_resnet50(use_se=False, **kwargs):
def gac_resnet50(use_se=False, n_styles=None):

    use_spatial_att = False
    fuse_epoch = 0
    # nclasses = 28000
    ndemog = 4
    kwargs = {"use_spatial_att":use_spatial_att, "ndemog":ndemog,\
        "hard_att_channel":False, "hard_att_spatial":False, \
        "lowresol_set":{'rate':1.0, 'mode':'nearest'},\
        "fuse_epoch":fuse_epoch, 'n_styles': n_styles,
        "adap": True, "att_mock": False}
    
    model = ResNetFace(IRBlock, [3, 4, 14, 3], use_se=use_se, **kwargs)
    return model

def gac_resnet100(use_se=False, **kwargs):
    model = ResNetFace(IRBlock, [3, 13, 30, 3], use_se=use_se, **kwargs)
    return model

def gac_resnet152(use_se=False, **kwargs):
    model = ResNetFace(IRBlock, [3, 8, 36, 3], use_se=use_se, **kwargs)
    return model
