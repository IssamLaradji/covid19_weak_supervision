import torch.nn as nn
import torch
import math

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.x64 = nn.Conv2d(1,64,kernel_size=1, padding = 0, bias = False)
        self.x128 = nn.Conv2d(1,128,kernel_size=1, padding = 0, bias = False)
        self.x256 = nn.Conv2d(1,256,kernel_size=1, padding = 0, bias = False)
        self.x64.weight = torch.nn.Parameter(torch.ones((64,1,1,1)))
        self.x128.weight = torch.nn.Parameter(torch.ones((128,1,1,1)))
        self.x256.weight = torch.nn.Parameter(torch.ones((256,1,1,1)))
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding = 1, bias=True)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding = 1, bias=True)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding = 1, bias=True)
        params = torch.load('/mnt/projects/vision_prototypes/pau/covid/vgg19-dcbb9e9d.pth')
        self.conv1_2.weight = torch.nn.Parameter(params['features.2.weight'])
        self.conv1_2.bias = torch.nn.Parameter(params['features.2.bias'])
        self.conv2_2.weight = torch.nn.Parameter(params['features.7.weight'])
        self.conv2_2.bias = torch.nn.Parameter(params['features.7.bias'])
        self.conv3_4.weight = torch.nn.Parameter(params['features.16.weight'])
        self.conv3_4.bias = torch.nn.Parameter(params['features.16.bias'])
        #{k: v for k, v in pretrained_dict.items() if k in model_dict}        
    def forward(self, x): 
        x64 = self.x64(x)
        x64 = self.conv1_2(x64)
        x128 = self.x128(x)
        x128 = self.conv2_2(x128)
        x256 = self.x256(x)
        x256 = self.conv3_4(x256)
        x_vgg = torch.cat([x64, x128, x256], dim = 1)
        return x_vgg