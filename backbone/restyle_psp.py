from copy import deepcopy
import pstats
# from cv2 import TonemapDrago
import imageio
from einops import rearrange
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Conv2d, BatchNorm2d, PReLU, Sequential, Module
from torchvision.models.resnet import resnet34

# from models.encoders.helpers import get_blocks, bottleneck_IR, bottleneck_IR_SE
# from models.encoders.map2style import GradualStyleBlock
# from backbone.stylegan2.model import EqualLinear, Generator
from backbone.restyle_psp_helpers import AttBlock, get_blocks, bottleneck_IR, bottleneck_IR_SE
from backbone.restyle_psp_helpers import Conv2dExtended, AdaConv2d_faster
from backbone.stylegan2_ada.generator import Generator as GeneratorAda
from util.utils import _initialize_weights
# from util.resize_right import resize_right
# from util.resize_right import interp_methods


class GradualStyleBlock(Module):
    def __init__(self, in_c, out_c, spatial):
        super(GradualStyleBlock, self).__init__()

        from backbone.stylegan2.model import EqualLinear
        
        self.out_c = out_c
        self.spatial = spatial
        num_pools = int(np.log2(spatial))
        modules = []
        modules += [Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU()]
        for i in range(num_pools - 1):
            modules += [
                Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU()
            ]
        self.convs = nn.Sequential(*modules)
        self.linear = EqualLinear(out_c, out_c, lr_mul=1)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.out_c)
        x = self.linear(x)
        return x


class BackboneEncoder(Module):
    """
    The simpler backbone architecture used by ReStyle where all style vectors are extracted from the final 16x16 feature
    map of the encoder. This classes uses the simplified architecture applied over an ResNet IRSE-50 backbone.
    Note this class is designed to be used for the human facial domain.
    """
    def __init__(self, num_layers, mode='ir', n_styles=18, opts=None):
        super(BackboneEncoder, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE

        self.input_layer = Sequential(Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

        self.styles = nn.ModuleList()
        self.style_count = n_styles
        for i in range(self.style_count):
            style = GradualStyleBlock(512, 512, 16)
            self.styles.append(style)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        latents = []
        for j in range(self.style_count):
            latents.append(self.styles[j](x))
        out = torch.stack(latents, dim=1)
        return out


class PSPOutputLayer(Module):
    def __init__(self, in_c, out_c, spatial, n_styles=18):
        super().__init__()
        self.styles = nn.ModuleList()
        self.style_count = n_styles
        for i in range(self.style_count):
            # style = GradualStyleBlock(512, 512, 16)
            # style = GradualStyleBlock(512, 512, 7)
            style = GradualStyleBlock(in_c, out_c, spatial)
            self.styles.append(style)
    
    def forward(self, x):
        latents = []
        for j in range(self.style_count):
            latents.append(self.styles[j](x))
            # print('[encoder] latents[-1]:', latents[-1].shape)
        
        out = torch.stack(latents, dim=1)
        # print('[encoder] out:', out.shape)
        
        return out


class BackboneEncoderDiffHead(Module):
    """
    The simpler backbone architecture used by ReStyle where all style vectors are extracted from the final 16x16 feature
    map of the encoder. This classes uses the simplified architecture applied over an ResNet IRSE-50 backbone.
    Note this class is designed to be used for the human facial domain.
    """
    def __init__(self, num_layers, mode='ir', n_styles=18, opts=None, emb_size=512, input_size=256, double_in_channels=False, output_layer_type='facerec', include_dropout=None):
        super().__init__()
        assert num_layers in [34, 50, 100, 152], 'num_layers should be 34, 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        print(f'Initializing backbone encoder with {num_layers} layers')
        print('num_layers:', num_layers)
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE

        # self.input_layer = Sequential(Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
        self.input_layer = Sequential(Conv2d(6, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        self.input_layer_att = nn.ModuleList([])
        modules = []
        doubling_factor = 2 if double_in_channels else 1
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel * doubling_factor,
                                           bottleneck.depth,
                                           bottleneck.stride,
                                           dropout=include_dropout))
        self.body = nn.ModuleList(modules)

        # self.styles = nn.ModuleList()
        # self.style_count = n_styles
        # for i in range(self.style_count):
        #     style = GradualStyleBlock(512, 512, 16)
        #     self.styles.append(style)

        self.input_size = input_size
        if input_size == 400:
            end_sp_size = 25
        elif input_size == 256:
            end_sp_size = 16
        elif input_size == 200:
            end_sp_size = 13
        elif input_size == 112:
            end_sp_size = 7

        self.output_layer_type = output_layer_type
        if output_layer_type == 'facerec':
            self.output_layer = Sequential(nn.BatchNorm2d(512),
                                           nn.Dropout(),
                                           nn.Flatten(),
                                           nn.Linear(512 * end_sp_size * end_sp_size, emb_size),    # 7x7 for 112x112 input, or 16x16 for 256x256 input
                                           nn.BatchNorm1d(emb_size))
        elif output_layer_type == 'pSp':
            self.output_layer = PSPOutputLayer(512, 512, 9, n_styles) 
        elif output_layer_type == 'both':
            self.output_layer_facerec = Sequential(nn.BatchNorm2d(512),
                                                   nn.Dropout(),
                                                   nn.Flatten(),
                                                   nn.Linear(512 * end_sp_size * end_sp_size, emb_size),    # 7x7 for 112x112 input, or 16x16 for 256x256 input
                                                   nn.BatchNorm1d(emb_size))
            self.output_layer_psp = PSPOutputLayer(512, 512, 9, n_styles)
        
        self.use_att = False
    
    def add_dropouts(self, include_dropout=None):
        if include_dropout:
            for name, m in self.body.named_modules():
                if isinstance(m, bottleneck_IR_SE):
                    print('adding dropout to the layer', name)
                    m.add_dropout(include_dropout)

    def forward(self, x, *args, races=None, **kwargs):
        if x.size(2) != self.input_size:
            # x = resize_right.resize(x, self.input_size, interp_method=interp_methods.linear)
            print('[interpolating ', x.size(2), ' to ', self.input_size, ']')
            x = F.interpolate(x, self.input_size, mode='bilinear')
        
        x = self.input_layer(x)

        if self.use_att:
            for module in self.input_layer_att:
                x = module(x, races)
        
        # x = self.body(x)
        for module in self.body:
            x = module(x, races)

        if self.output_layer_type in ('facerec', 'pSp'):
            x = self.output_layer(x)
        elif self.output_layer_type == 'both':
            x_facerec = self.output_layer_facerec(x)
            x_pSp = self.output_layer_psp(x)
            x = {'facerec': x_facerec, 'pSp': x_pSp}

        return x


class ResNetBackboneEncoder(Module):
    """
    The simpler backbone architecture used by ReStyle where all style vectors are extracted from the final 16x16 feature
    map of the encoder. This classes uses the simplified architecture applied over an ResNet34 backbone.
    """
    def __init__(self, n_styles=18, opts=None):
        super(ResNetBackboneEncoder, self).__init__()

        self.conv1 = nn.Conv2d(opts.input_nc, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BatchNorm2d(64)
        self.relu = PReLU(64)

        resnet_basenet = resnet34(pretrained=True)
        blocks = [
            resnet_basenet.layer1,
            resnet_basenet.layer2,
            resnet_basenet.layer3,
            resnet_basenet.layer4
        ]
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(bottleneck)
        self.body = Sequential(*modules)

        self.styles = nn.ModuleList()
        self.style_count = n_styles
        for i in range(self.style_count):
            style = GradualStyleBlock(512, 512, 16)
            self.styles.append(style)

    def forward(self, x, *args, **kwargs):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.body(x)
        latents = []
        for j in range(self.style_count):
            latents.append(self.styles[j](x))
        out = torch.stack(latents, dim=1)
        return out


class ResNetBackboneEncoderDiffHead(Module):
    """
    The simpler backbone architecture used by ReStyle where all style vectors are extracted from the final 16x16 feature
    map of the encoder. This classes uses the simplified architecture applied over an ResNet34 backbone.
    """
    def __init__(self, n_styles=18, emb_size=512, opts=None):
        super().__init__()

        self.conv1 = nn.Conv2d(opts.input_nc, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BatchNorm2d(64)
        self.relu = PReLU(64)

        resnet_basenet = resnet34(pretrained=True)
        blocks = [
            resnet_basenet.layer1,
            resnet_basenet.layer2,
            resnet_basenet.layer3,
            resnet_basenet.layer4
        ]
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(bottleneck)
        self.body = Sequential(*modules)

        # self.styles = nn.ModuleList()
        # self.style_count = n_styles
        # for i in range(self.style_count):
        #     style = GradualStyleBlock(512, 512, 16)
        #     self.styles.append(style)

        self.output_layer = Sequential(nn.BatchNorm2d(512),
                                       nn.Dropout(),
                                       nn.Flatten(),
                                       nn.Linear(512 * 7 * 7, emb_size),
                                       nn.BatchNorm1d(emb_size))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.body(x)

        # latents = []
        # for j in range(self.style_count):
        #     latents.append(self.styles[j](x))
        # out = torch.stack(latents, dim=1)

        x = self.output_layer(x)

        return x


# specify the encoder types for pSp and e4e - this is mainly used for the inference scripts
ENCODER_TYPES = { 
    'pSp': ['GradualStyleEncoder', 'ResNetGradualStyleEncoder', 'BackboneEncoder', 'ResNetBackboneEncoder'],
    'e4e': ['ProgressiveBackboneEncoder', 'ResNetProgressiveBackboneEncoder']
}

RESNET_MAPPING = {
    'layer1.0': 'body.0',
    'layer1.1': 'body.1',
    'layer1.2': 'body.2',
    'layer2.0': 'body.3',
    'layer2.1': 'body.4',
    'layer2.2': 'body.5',
    'layer2.3': 'body.6',
    'layer3.0': 'body.7',
    'layer3.1': 'body.8',
    'layer3.2': 'body.9',
    'layer3.3': 'body.10',
    'layer3.4': 'body.11',
    'layer3.5': 'body.12',
    'layer4.0': 'body.13',
    'layer4.1': 'body.14',
    'layer4.2': 'body.15',
}

model_paths = {
	'ir_se50': 'pretrained_models/model_ir_se50.pth',
	'resnet34': 'pretrained_models/resnet34-333f7ec4.pth',
	'stylegan_ffhq': 'pretrained_models/stylegan2-ffhq-config-f.pt',
	'stylegan_cars': 'pretrained_models/stylegan2-car-config-f.pt',
	'stylegan_church': 'pretrained_models/stylegan2-church-config-f.pt',
	'stylegan_horse': 'pretrained_models/stylegan2-horse-config-f.pt',
	'stylegan_ada_wild': 'pretrained_models/afhqwild.pt',
	'stylegan_toonify': 'pretrained_models/ffhq_cartoon_blended.pt',
	'shape_predictor': 'pretrained_models/shape_predictor_68_face_landmarks.dat',
	'circular_face': 'pretrained_models/CurricularFace_Backbone.pth',
	'mtcnn_pnet': 'pretrained_models/mtcnn/pnet.npy',
	'mtcnn_rnet': 'pretrained_models/mtcnn/rnet.npy',
	'mtcnn_onet': 'pretrained_models/mtcnn/onet.npy',
	'moco': 'pretrained_models/moco_v2_800ep_pretrain.pth.tar'
}


class pSp(nn.Module):

    # def __init__(self, opts):
    def __init__(self, size=256, encoder_type='BackboneEncoder', checkpoint_path=None, 
                 avg_image=None, num_diff_blocks=1, 
                 include_dropout=None, include_attblocks=None, attblock_init_strategy='ones',
                 decoder_checkpoint_path=None,
                 subbatch_mode='random',
                 stylegan_subbatch_size=None):
        super(pSp, self).__init__()
        # self.set_opts(opts)
        # self.n_styles = int(math.log(self.opts.output_size, 2)) * 2 - 2
        # self.n_styles = int(math.log(output_size, 2)) * 2 - 2
        # Define architecture
        self.size = size
        self.encoder_type = encoder_type
        self.num_diff_blocks = num_diff_blocks
        self.include_dropout = include_dropout
        self.subbatch_mode = subbatch_mode
        self.stylegan_subbatch_size = stylegan_subbatch_size

        self.encoder = self.set_encoder(encoder_type)

        if avg_image is None:
            self.avg_image = None
        else:
            avg_image = imageio.imread(avg_image)
            avg_image = torch.tensor(avg_image).to('cuda:0')
            avg_image = rearrange(avg_image, 'h w c -> c h w')
            avg_image = avg_image / 255.0
            avg_image = (avg_image - 0.5) / 0.5
            self.avg_image = avg_image.to('cuda').float().detach()

        # self.decoder = Generator(output_size, 512, 8, channel_multiplier=2)
        # self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        # Load weights if needed
        
        _initialize_weights(self.encoder)

        if checkpoint_path is not None:
            print('[pSp] Loading weights...')
            self.load_weights(checkpoint_path)

        if include_dropout:
            print('[include_dropout]')
            if not hasattr(self.encoder, 'add_dropouts'):
                raise Exception('encoder contains no .add_dropouts(...) method')
            self.encoder.add_dropouts(include_dropout)

    def set_encoder(self, encoder_type):
        if encoder_type == 'BackboneEncoder':
            encoder = BackboneEncoderDiffHead(50, 'ir_se', input_size=self.size)
        elif encoder_type == 'BackboneEncoder34':
            encoder = BackboneEncoderDiffHead(34, 'ir_se', input_size=self.size)
        elif encoder_type == 'BackboneEncoder100':
            encoder = BackboneEncoderDiffHead(100, 'ir_se', input_size=self.size)
        else:
            raise Exception(f'{encoder_type} is not a valid encoders')
        # we include dropout later to support transferring from a checkpoint without dropout
        return encoder

    def load_weights(self, checkpoint_path):
        # if checkpoint_path is not None:
        if checkpoint_path is None:
            print('NOT loading weights for encoder (checkpoint_path is None)')
            return
        
        print(f'Loading ReStyle pSp from checkpoint: {checkpoint_path}')
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        # print('keys in the ckpt encoder:', 
            #   [name for name in ckpt['state_dict'].keys()])
        ckpt_encoder = self.__get_keys(ckpt, 'encoder')
        # ckpt_encoder = self.__filter_keys(ckpt_encoder, 'styles')
        # print('keys in the ckpt encoder:', 
            #   [name for name in ckpt_encoder.keys()])
        
        # self.encoder.load_state_dict(ckpt_encoder, strict=False)
        print('Loading encoder weights...')
        self.encoder.input_layer.load_state_dict(self.__get_keys(ckpt_encoder, 'input_layer'), strict=True)
        self.encoder.body.load_state_dict(self.__get_keys(ckpt_encoder, 'body'), strict=True)

    def forward(self, x, races=None, is_generated=None):  
        if x.size(2) != self.size:
            # x = resize_right.resize(x, self.input_size, interp_method=interp_methods.linear)
            print('[interpolating ', x.size(2), ' to ', self.size, ']')
            x = F.interpolate(x, self.size, mode='bilinear')
        
        if self.avg_image is not None:      
            avg_image_for_batch = self.avg_image.unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
            x_input = torch.cat([x, avg_image_for_batch.to(x.device)], dim=1)
        else:
            x_input = x
        codes = self.encoder(x_input, races=races, is_generated=is_generated)
        
        return codes

    # def set_opts(self, opts):
    #     self.opts = opts

    def __load_latent_avg(self, ckpt, repeat=None):
        if 'latent_avg' in ckpt:
            # self.latent_avg = ckpt['latent_avg'].to(self.opts.device)
            self.latent_avg = ckpt['latent_avg'].to('cuda:0')
            if repeat is not None:
                self.latent_avg = self.latent_avg.repeat(repeat, 1)
        else:
            self.latent_avg = None

    @staticmethod
    def __get_keys(d, name):
        if 'state_dict' in d:
            d = d['state_dict']
        d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
        return d_filt
    
    @staticmethod
    def __filter_keys(d, substring):
        if 'state_dict' in d:
            d = d['state_dict']
        d_filt = {k: v for k, v in d.items() if substring not in k}
        return d_filt
