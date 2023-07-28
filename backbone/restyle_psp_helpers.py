import sys
from collections import namedtuple
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Conv2d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout, MaxPool2d, AdaptiveAvgPool2d, Sequential, Module, ModuleList
import torch.nn.functional as F

"""
ArcFace implementation from [TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch)
"""


class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    """ A named tuple describing a ResNet block. """


def get_block(in_channel, depth, num_units, stride=2):
    return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]


def get_blocks(num_layers):
    if num_layers == 34:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=6),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    else:
        raise ValueError("Invalid number of layers: {}. Must be one of [34, 50, 100, 152]".format(num_layers))
    return blocks


class SEModule(Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


def mlist_forward(mlist, x, race=None):
    out = x
    for m in mlist:
        if not isinstance(m, Conv2dExtended) and not isinstance(m, AdaConv2d_faster):
            out = m(out)
        else:
            # print('[mlist_forward] race:', race)
            out = m(out, race)
    return out


class bottleneck_IR(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth)
            )
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False), PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False), BatchNorm2d(depth)
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class bottleneck_IR_SE(Module):
    def __init__(self, in_channel, depth, stride, dropout=None):
        super(bottleneck_IR_SE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            if not dropout:
                # self.shortcut_layer = Sequential(
                # 	Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                # 	BatchNorm2d(depth)
                # )
                self.shortcut_layer = ModuleList([
                    Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                    BatchNorm2d(depth)
                ])
            else:
                # self.shortcut_layer = Sequential(
                # 	Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                # 	Dropout(p=dropout),
                # 	BatchNorm2d(depth)
                # )
                self.shortcut_layer = ModuleList([
                    Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                    Dropout(p=dropout),
                    BatchNorm2d(depth)
                ])

        if not dropout:
            # self.res_layer = Sequential(
            # 	BatchNorm2d(in_channel),
            # 	Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            # 	PReLU(depth),
            # 	Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            # 	BatchNorm2d(depth),
            # 	SEModule(depth, 16)
            # )
            self.res_layer = ModuleList([
                BatchNorm2d(in_channel),
                Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
                PReLU(depth),
                Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
                BatchNorm2d(depth),
                SEModule(depth, 16)
            ])
        else:
            print('[bottleneck_IR_SE] adding dropout layer')
            # self.res_layer = Sequential(
            # 	BatchNorm2d(in_channel),
            # 	Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            # 	Dropout(p=dropout),
            # 	PReLU(depth),
            # 	Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            # 	Dropout(p=dropout),
            # 	BatchNorm2d(depth),
            # 	SEModule(depth, 16)
            # )
            self.res_layer = ModuleList([
                BatchNorm2d(in_channel),
                Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
                Dropout(p=dropout),
                PReLU(depth),
                Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
                Dropout(p=dropout),
                BatchNorm2d(depth),
                SEModule(depth, 16)
            ])
        
        self.in_channel = in_channel
        self.depth = depth
        self.use_att = False

    def forward(self, x, race=None):
        if not isinstance(self.shortcut_layer, ModuleList):
            shortcut = self.shortcut_layer(x)
        else:
            shortcut = mlist_forward(self.shortcut_layer, x, race=race)
        # res = self.res_layer(x)
        res = mlist_forward(self.res_layer, x, race=race)
        if self.use_att:
            res = self.att(res, race)
        return res + shortcut
    
    def add_dropout(self, p):
        # print('[before] shortcut layer:', self.shortcut_layer)
        # print('[before] res layer:', self.res_layer)

        if isinstance(self.shortcut_layer, ModuleList):
            self.shortcut_layer.insert(1, Dropout(p=p))
        if isinstance(self.res_layer, ModuleList):
            self.res_layer.insert(2, Dropout(p=p))
            self.res_layer.insert(5, Dropout(p=p))

        # print('[after] shortcut layer:', self.shortcut_layer)
        # print('[after] res layer:', self.res_layer)
    
    def add_attblock(self, init_strategy='ones'):
        self.use_att = True
        self.att = AttBlock(self.depth, init_strategy=init_strategy)


class Conv2dExtended(Conv2d):
    def __init__(self, n_demog, *args, **kwargs):
        # super().__init__(*args, **kwargs)
        self.conv = Conv2d(*args, **kwargs)
        self.n_demog = n_demog
    
    def forward(self, x, races):
        demog_feat_map = torch.zeros((x.size(0), self.n_demog, 1, 1)).to(x.device)
        b_inds = torch.arange(x.size(0)).long().to(x.device)
        demog_feat_map[b_inds, races] = 1
        # print('demog_feat_map:', demog_feat_map)
        demog_feat_map = demog_feat_map.repeat(1, 1, x.size(2), x.size(3))
        
        x_cat = torch.cat([x, demog_feat_map], dim=1)

        out = self.conv(x_cat)
        return out


class AdaConv2d_faster(nn.Module):
    def __init__(self, ndemog, ic, oc, ks, stride, padding=0, adap=True):
        super(AdaConv2d_faster, self).__init__()
        self.stride = stride
        self.padding = padding

        self.oc = oc
        self.ic = ic
        self.ks = ks
        self.ndemog = ndemog
        self.adap = adap
        # self.kernel_adap = nn.Parameter(torch.Tensor(ndemog, oc, ic, ks, ks))
        self.kernel_base = nn.Parameter(torch.Tensor(oc, ic, ks, ks))
        # self.kernel_net = nn.Linear(ndemog, oc*ic*ks*ks)
        self.kernel_mask = nn.Parameter(torch.Tensor(1, ic, ks, ks))

        # self.conv = nn.Conv2d(ic, oc, kernel_size=3, stride=stride,
        #              padding=1, bias=False)
        # self.conv.weight = self.kernel_base

        if adap:
            self.kernel_mask.data = self.kernel_mask.data.repeat(ndemog,1,1,1)
        
        # self.kernel_base.data.normal_(0, np.sqrt(2. / (oc * ks * ks)))
        # self.kernel_mask.data.normal_(0, np.sqrt(2. / (self.kernel_mask.size(0) * ks * ks)))

        nn.init.xavier_normal_(self.kernel_base)
        nn.init.xavier_normal_(self.kernel_mask)

    def forward(self, x, demog_label):
        # demogs = list(range(self.ndemog))

        # print('demog_label[:10]:', demog_label[:10])

        if self.adap:
            # ----------------------------------------------------------
            # version 2 -- much faster (1h)
            kernel_comb = (self.kernel_base.unsqueeze(1) # (oc, 1, ic, ks, ks)
                           * self.kernel_mask.unsqueeze(0).repeat(self.oc, 1, 1, 1, 1)  # (oc, ndemog, ic, ks, ks)
            )  # (oc, ndemog, ic, ks, ks)

            # print('[Adaconv] races[:100]:', demog_label[:100])

            output = F.conv2d(x, 
                              kernel_comb[:, 0],
                              stride=self.stride, padding=self.padding)

            for i in range(self.ndemog):
                # get indices
                if i > 1 and torch.any(demog_label == i):
                    # indices = torch.nonzero((demog_label==demog), as_tuple=False).squeeze()
                    # if indices.dim() == 0:
                    #     indices = indices.unsqueeze(0)
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
        else:
            output = F.conv2d(x, self.kernel_base, stride=self.stride, padding=self.padding)
        
        # # mock:
        # output = F.conv2d(x, self.kernel_base, stride=self.stride, padding=self.padding)

        # print('x.shape:', x.shape)
        # print('output.shape:', output.shape)
        # print('x beginning:', x[:10, :10, :10, :10])
        # print('output beginning:', output[:10, :10, :10, :10])
            
        return output


class AttBlock(nn.Module):
    def __init__(self, nchannel, ndemog=4, init_strategy='xavier', att_mock=False):
        super().__init__()
        self.ndemog = ndemog

        self.att_channel = nn.parameter.Parameter(torch.Tensor(1, 1, nchannel, 1, 1))
        
        if init_strategy == 'xavier':
            nn.init.xavier_uniform_(self.att_channel)
        elif init_strategy == 'ones':
            self.att_channel.data.fill_(0)
        
        self.att_channel.data = self.att_channel.data.repeat(ndemog,1,1,1,1)

        self.init_strategy = init_strategy
        self.att_mock = att_mock

    def forward(self, x, demog_label):
        y = x
        # demogs = list(set(demog_label.tolist()))
        # demogs = list(range(self.ndemog))

        att_channel = torch.sigmoid(self.att_channel)
        if self.init_strategy == 'ones':
            # multiplying by 2 to make sigmoid(0) * 2 equal to 1
            att_channel *= 2

        if not self.att_mock:
            
            # old version
            # for demog in demogs:
            #     indices = torch.nonzero((demog_label==demog), as_tuple=False).squeeze()
            #     if indices.dim() == 0:
            #         indices = indices.unsqueeze(0)
            #     # y[indices,:,:,:] = x[indices,:,:,:] *\
            #     #     att_channel.repeat(1, indices.size(0), 1, x.size(2), x.size(3))[demog,:,:,:,:]
            #     x_clone = x[indices,:,:,:].clone()
            #     attr_channel_clone = att_channel.repeat(1, indices.size(0), 1, x.size(2), x.size(3))[demog,:,:,:,:].clone()
            #     y[indices,:,:,:] = x_clone * attr_channel_clone

            # if verbose:
            #     print('att_channel[:, :, 0]:', att_channel[:, :, 0], 
            #         '\n', 'demog_label[:20]', demog_label[:20],
            #         '\n', 'att_channel[demog_label][:20, :, 0]:', att_channel[demog_label][:20, :, 0])
            #     sys.stdout.flush()
            att_channel_resampled = att_channel[demog_label].squeeze(1)
            y *= att_channel_resampled

        # print('self.att_mock:', self.att_mock)
        # sys.stdout.flush()
        # if self.att_mock == True, no attention will be applied
        return y
