import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from loss.lpips.lpips import LPIPS

# Support: ['FocalLoss']

class FocalLoss(nn.Module):
    def __init__(self, gamma = 2, eps = 1e-7, use_weights=False):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.use_weights = use_weights
        self.ce = nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean(), None

