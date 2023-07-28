import os
from typing import Optional
from glob import glob

import imageio
import numpy as np
from PIL import Image
from turbojpeg import TurboJPEG
import pandas as pd
from einops import rearrange
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


class FacesDataset(torch.utils.data.Dataset):

    class2race = {
        'African': 0,
        'Asian': 1,
        'Caucasian': 2,
        'Indian': 3
    }

    race2class = ['African', 'Asian', 'Caucasian', 'Indian']

    def __init__(self, root, transform=None, jpeg_loader=None, loss_weights_file=None, return_onehot=False, id2race_file=None):
        """
        The dataset must have the following structure:
        
        <root>/<identity_code>/<filename.jpg>
        """
        super().__init__()

        self.root = root
        self.transform = transform
        self.filenames = list(sorted(glob(os.path.join(root, '*', '*.jpg'))))
        print('Checking loaded data.')
        print('# filenames:', len(self.filenames))
        print('filenames[:5]', self.filenames[:5])

        self.id_list = [fn.split(os.sep)[-2]
                        for fn in self.filenames]
        self.id_list = [fn if '^' not in fn else fn[fn.rfind('^') + 1:]
                        for fn in self.id_list]     # handling case when IDs are given as e.g. "Caucasian^m49.r8743" (removing ethnicity part)
        self.id_list = list(sorted(set(self.id_list)))
        print('self.id_list[:5]:', self.id_list[:5])

        self.id2race = None
        if id2race_file is not None:
            self.id2race = open(id2race_file).read().splitlines()
            self.id2race = {el.split(' ')[0]: el.split(' ')[1]
                            for el in self.id2race}

        self.classes = self.id_list
        self.id2label = {identity: label for label, identity in enumerate(self.id_list)}
        self.n_identities = len(self.id_list)
        print('# identities:', self.n_identities)

        self.orig_n_samples = len(self.filenames)

        self.dims = (112, 112, 3)
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        fn = self.filenames[idx]
        identity, sample_name = fn.split(os.sep)[-2:]
        
        if identity.startswith('African') or identity.startswith('Asian') or identity.startswith('Caucasian') or identity.startswith('Indian'):
            identity = identity[identity.rfind('^') + 1:]    # "Caucasian^m49.r8743" -> "m49.r8743"
        
        sample_name = os.path.splitext(sample_name)[0]
        
        try:
            img = Image.open(fn)
        except:    # broken file
            print('[Image Jpeg loading error]')
            return None
        
        label = self.id2label[identity]
        
        try:
            if self.transform:
                img = self.transform(img)
        except:
            print('[Error during transforming image]')
            return None
        return (img, label)
