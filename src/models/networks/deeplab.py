import torch.nn as nn
import torchvision
import torch
from skimage import morphology as morph
import numpy as np
from torch import optim
import torch.nn.functional as F

import torch
import torch.nn.functional as F
import numpy as np
from skimage.morphology import watershed
from skimage.segmentation import find_boundaries
from scipy import ndimage
from torchvision import models

class DeepLab(nn.Module):
    def __init__(self,  **kwargs):
        super().__init__()
        self.model = models.segmentation.deeplabv3_resnet50(pretrained_backbone=False)
        # base.forward = lambda x: base.forward(x)['out']
        # # FREEZE BATCH NORMS
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad = False
                m.bias.requires_grad = False
                # m.reset_parameters()
                m.eval()
                # with torch.no_grad():
                #     m.weight.fill_(1.0)
                #     m.bias.zero_()


    

    def forward(self, x):
        return self.model(x)['out']