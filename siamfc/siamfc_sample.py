import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import cv2
import sys
import os
from collections import namedtuple
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from got10k.trackers import Tracker
from torch.utils.tensorboard import SummaryWriter

from . import ops
from .backbones import AlexNetV2
from .heads import SiamFC
from .losses import BalancedLoss,FocalLoss,GHMCLoss,OHNMLoss
from .datasets import Pair
from .transforms import SiamFCTransforms

__all__ = ['TrackerSiamFC']

class Net(nn.Module):
    
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone= backbone
        self.head = head
    
    def forward(self, z, x):
        z = self.backbone(z)
        x = self.backbone(x)
        return self.head(z, x)

class TrackerSiamFC(Tracker):
    def __init__(self, net_path=None, **kwargs):
        super.__init__('SiamFC',True)
        self.cfg = self.parse_args(**kwargs)
        
        # setup GPU device if available
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')
        
        #setup model
        self.net = Net(
            backbone=AlexNetV2(),
            head=SiamFC(self.cfg.out_scale))
        ops.init_weights(self.net)
        
        # load checkpoint if provided
        if net_path is not None:
            self.net.load_state_dict(torch.load(
                net_path, map_location=lambda storage, loc: storage))
        
        self.net = self.net.to(self.device)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        