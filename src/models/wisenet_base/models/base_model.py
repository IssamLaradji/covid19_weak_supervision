import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch.autograd import Variable
import numpy as np 
from .. import misc as ms
from skimage import morphology as morph
from torch.autograd import Function
import torchvision
# from core import proposals as prp
# from core import predict_methods as pm 
# from core import score_functions as sf 
from .. import ann_utils as au
class BaseModel(nn.Module):
    def __init__(self, train_set, **model_options):
        super().__init__()
        self.options = model_options
        # self.predict_dict = ms.get_functions(pm)

        if hasattr(train_set, "n_classes"):
            self.n_classes = train_set.n_classes
        else:
            self.n_classes = train_set["n_classes"]  

        if hasattr(train_set, "ignore_index"):
            self.ignore_index = train_set.ignore_index
        else:
            self.ignore_index = -100

        self.blob_mode = None
        self.trained_batch_names = set()

    def sanity_checks(self, batch):
        if batch["split"][0] != "train":
            assert batch["name"][0] not in self.trained_batch_names 
        
    @torch.no_grad()
    def predict(self, batch, predict_method="probs"):
        self.sanity_checks(batch)
        self.eval()
        # ms.reload(pm)
        # self.predict_dict = ms.get_functions(pm)
        if predict_method == "counts":
            probs = F.softmax(self(batch["images"].cuda()),dim=1).data
            blob_dict = au.probs2blobs(probs)

            return {"counts":blob_dict["counts"]}

        elif predict_method == "probs":
            probs = F.softmax(self(batch["images"].cuda()),dim=1).data
            return {"probs":probs}

        elif predict_method == "points":
            probs = F.softmax(self(batch["images"].cuda()),dim=1).data
            blob_dict = au.probs2blobs(probs)

            return {"points":blob_dict["points"], 
                    "pointList":blob_dict["pointList"],
                    "probs":probs}
            

        elif predict_method == "blobs":
            probs = F.softmax(self(batch["images"].cuda()),dim=1).data
            blob_dict = au.probs2blobs(probs)
            
            return blob_dict

        else:
            print("Used predict method {}".format(predict_method))
            return self.predict_dict[predict_method](self, batch)

    @torch.no_grad()
    def get_embedding_blobs(self, O, fg_bg_seeds):
        n, c, h, w = O.shape
        # seeds = torch.cat([fg_seeds, bg_seeds], 2)[:,:,None]
        fA = O.view(1,c,-1)
        fS = O[:,:, fg_bg_seeds["yList"], fg_bg_seeds["xList"]]

        n_pixels = h*w
        blobs = torch.zeros(h*w)
        sim = torch.zeros(h*w)
        n_seeds =  fS.shape[-1]

        maximum = 5000000
        n_loops = int(np.ceil((n_pixels * n_seeds) / maximum))
        
        for (s,e) in get_batches(n_pixels, size=n_pixels//n_loops):
            # s,e = map(int, (s,e))
            diff = pairwise_sum(fS[:,:,None], fA[:,:,s:e,None]) 
            sim[s:e] = diff
            blobs[s:e] = diff.max(2)[1] + 1 
        
        bg_min_index = np.where(np.array(fg_bg_seeds["categoryList"])==0)[0].min()
        assert len(fg_bg_seeds["yList"])//2 == bg_min_index
        blobs[blobs > int(bg_min_index)] = 0
        blobs = blobs.squeeze().reshape(h,w).long()

        categoryDict = {}
        for i, category_id in enumerate(fg_bg_seeds["categoryList"]):
            if category_id == 0:
                 continue

            categoryDict[i+1] = category_id 

        return {"blobs":ms.t2n(blobs), "categoryDict":categoryDict, "sim":sim}
        
def get_batches(n_pixels, size=500000):
    batches = []
    for i in range(0, n_pixels, size):
        batches +=[(i, i+size)]
    return batches


class Upsampler(nn.Module):
    def __init__(self, expansion_rate, n_output):
        super().__init__()
        
        self.score_32s = nn.Conv2d(512 *  4,
                                   n_output,
                                   kernel_size=1)
        
        self.score_16s = nn.Conv2d(256 *  4,
                                   n_output,
                                   kernel_size=1)
        
        self.score_8s = nn.Conv2d(128 *  4,
                                   n_output,
                                   kernel_size=1)
    
    def upsample(self, x_input, x_8s, x_16s, x_32s):
        input_spatial_dim = x_input.size()[2:]
        
        logits_8s = self.score_8s(x_8s)
        logits_16s = self.score_16s(x_16s)
        logits_32s = self.score_32s(x_32s)
        
        logits_16s_spatial_dim = logits_16s.size()[2:]
        logits_8s_spatial_dim = logits_8s.size()[2:]
        
                
        logits_16s += nn.functional.interpolate(logits_32s,
                                        size=logits_16s_spatial_dim,
                                        mode="bilinear",
                                        align_corners=True)
        
        logits_8s += nn.functional.interpolate(logits_16s,
                                        size=logits_8s_spatial_dim,
                                        mode="bilinear",
                                        align_corners=True)
        
        logits_upsampled = nn.functional.interpolate(logits_8s,
                                       size=input_spatial_dim,
                                        mode="bilinear",
                                        align_corners=True)
        return logits_upsampled


class FeatureExtracterStrong(nn.Module):
    def __init__(self):
        super().__init__()
        
        resnet50_32s = torchvision.models.resnet50(pretrained=True)
        resnet50_32s.fc = nn.Sequential()
        self.resnet50_32s = resnet50_32s
        self.expansion_rate = resnet50_32s.layer1[0].expansion

        # # FREEZE BATCH NORMS
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    
    def extract_features(self, x_input):
        self.resnet50_32s.eval()
        x = self.resnet50_32s.conv1(x_input)
        x = self.resnet50_32s.bn1(x)
        x = self.resnet50_32s.relu(x)
        x = self.resnet50_32s.maxpool(x)

        x_4s = self.resnet50_32s.layer1(x)
        
        x_8s = self.resnet50_32s.layer2(x_4s)
        x_16s = self.resnet50_32s.layer3(x_8s)
        x_32s = self.resnet50_32s.layer4(x_16s)

        return x_4s, x_8s, x_16s, x_32s

class FeatureExtracterStrong101(nn.Module):
    def __init__(self):
        super().__init__()
        
        resnet50_32s = torchvision.models.resnet101(pretrained=True)
        resnet50_32s.fc = nn.Sequential()
        self.resnet50_32s = resnet50_32s
        self.expansion_rate = resnet50_32s.layer1[0].expansion

        # # FREEZE BATCH NORMS
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    
    def extract_features(self, x_input):
        self.resnet50_32s.eval()
        x = self.resnet50_32s.conv1(x_input)
        x = self.resnet50_32s.bn1(x)
        x = self.resnet50_32s.relu(x)
        x = self.resnet50_32s.maxpool(x)

        x_4s = self.resnet50_32s.layer1(x)
        
        x_8s = self.resnet50_32s.layer2(x_4s)
        x_16s = self.resnet50_32s.layer3(x_8s)
        x_32s = self.resnet50_32s.layer4(x_16s)

        return x_4s, x_8s, x_16s, x_32s

class UpsamplerStrong(nn.Module):
    def __init__(self, expansion_rate, n_output):
        super().__init__()
        self.out_channels = depth = 256
        self.P6 = nn.MaxPool2d(kernel_size=1, stride=2)
        self.P5_conv1 = nn.Conv2d(2048, self.out_channels, kernel_size=1, stride=1)
        self.P5_conv2 = nn.Sequential(
            SamePad2dStrong(kernel_size=3, stride=1),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1),
        )
        self.P4_conv1 =  nn.Conv2d(1024, self.out_channels, kernel_size=1, stride=1)
        self.P4_conv2 = nn.Sequential(
            SamePad2dStrong(kernel_size=3, stride=1),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1),
        )
        self.P3_conv1 = nn.Conv2d(512, self.out_channels, kernel_size=1, stride=1)
        self.P3_conv2 = nn.Sequential(
            SamePad2dStrong(kernel_size=3, stride=1),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1),
        )
        self.P2_conv1 = nn.Conv2d(256, self.out_channels, kernel_size=1, stride=1)
        self.P2_conv2 = nn.Sequential(
            SamePad2dStrong(kernel_size=3, stride=1),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1),
        )

        # self.mask = StrongMask(depth=depth, pool_size=14, num_classes=n_output)
        self.out = nn.Conv2d(256, n_output, kernel_size=1, stride=1)
    
    def upsample(self, x_input, x_4s, x_8s, x_16s, x_32s):
        input_spatial_dim = x_input.size()[2:]
        
        c2_out = x_4s
        c3_out = x_8s
        c4_out = x_16s
        
        p5_out = self.P5_conv1(x_32s)

        c4_conv = self.P4_conv1(c4_out)
        p4_out = c4_conv + F.interpolate(p5_out, size=c4_conv.size()[-2:], mode="nearest")
        # p4_out = c4_conv + F.interpolate(p5_out, scale_factor=2, mode="nearest")

        c3_conv = self.P3_conv1(c3_out)
        p3_out = c3_conv +  F.interpolate(p4_out, size=c3_conv.size()[-2:], mode="nearest")
        # p3_out = c3_conv +  F.interpolate(p4_out, scale_factor=2, mode="nearest")

        c2_conv = self.P2_conv1(c2_out)
        p2_out = c2_conv + F.interpolate(p3_out, size=c2_conv.size()[-2:], mode="nearest")
        # p2_out = c2_conv + F.interpolate(p3_out, scale_factor=2, mode="nearest")

        p5_out = self.P5_conv2(p5_out)
        p4_out = self.P4_conv2(p4_out)
        p3_out = self.P3_conv2(p3_out)
        p2_out = self.P2_conv2(p2_out)

        # P6 is used for the 5th anchor scale in RPN. Generated by
        # subsampling from P5 with stride of 2.
        # p6_out = self.P6(p5_out)
        out = self.out(p2_out)
        out = nn.functional.interpolate(out,
                                       size=input_spatial_dim,
                                        mode="nearest")
        return out
        # return [p2_out, p3_out, p4_out, p5_out, p6_out]

      


class StrongMask(nn.Module):
    def __init__(self, depth=256, pool_size=14, num_classes=21):
        super(StrongMask, self).__init__()
        self.depth = depth
        self.pool_size = pool_size
        self.num_classes = num_classes
        self.padding = SamePad2dStrong(kernel_size=3, stride=1)
        self.conv1 = nn.Conv2d(self.depth, 256, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.deconv = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(self.padding(x))
        x = self.relu(x)
        x = self.conv2(self.padding(x))
        x = self.relu(x)
        x = self.conv3(self.padding(x))
        x = self.relu(x)
        x = self.conv4(self.padding(x))
        x = self.relu(x)
        x = self.deconv(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.sigmoid(x)

        return x

import math
class SamePad2dStrong(nn.Module):
    """Mimics tensorflow's 'SAME' padding.
    """

    def __init__(self, kernel_size, stride):
        super(SamePad2dStrong, self).__init__()
        self.kernel_size = torch.nn.modules.utils._pair(kernel_size)
        self.stride = torch.nn.modules.utils._pair(stride)

    def forward(self, input):
        in_width = input.size()[2]
        in_height = input.size()[3]
        out_width = math.ceil(float(in_width) / float(self.stride[0]))
        out_height = math.ceil(float(in_height) / float(self.stride[1]))
        pad_along_width = ((out_width - 1) * self.stride[0] +
                           self.kernel_size[0] - in_width)
        pad_along_height = ((out_height - 1) * self.stride[1] +
                            self.kernel_size[1] - in_height)
        pad_left = math.floor(pad_along_width / 2)
        pad_top = math.floor(pad_along_height / 2)
        pad_right = pad_along_width - pad_left
        pad_bottom = pad_along_height - pad_top
        return F.pad(input, (pad_left, pad_right, pad_top, pad_bottom), 'constant', 0)

    def __repr__(self):
        return self.__class__.__name__
def pairwise_sum(fi, fj):
    diff = (fi - fj).pow(2).sum(1).clamp(min=0, max=50)
    # print(diff.max().item())
    return  (2./ (1+torch.exp(diff))).clamp(min=1e-6, max=(1.-1e-6))
    





def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()


def conv3x3(in_planes, out_planes, stride=1, padding=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=(3,3), stride=(stride,stride),
                     padding=(padding,padding))

def conv1x1(in_planes, out_planes, stride=1):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0)





class FeatureExtracter(nn.Module):
    def __init__(self):
        super().__init__()
        
        resnet50_32s = torchvision.models.resnet50(pretrained=True)
        resnet50_32s.fc = nn.Sequential()
        self.resnet50_32s = resnet50_32s
        self.expansion_rate = resnet50_32s.layer1[0].expansion
        self.resnet50_32s.fc = nn.Sequential()
        # # FREEZE BATCH NORMS
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    
    def extract_features(self, x_input):
        self.resnet50_32s.eval()
        x = self.resnet50_32s.conv1(x_input)
        x = self.resnet50_32s.bn1(x)
        x = self.resnet50_32s.relu(x)
        x = self.resnet50_32s.maxpool(x)

        x = self.resnet50_32s.layer1(x)
        
        x_8s = self.resnet50_32s.layer2(x)
        x_16s = self.resnet50_32s.layer3(x_8s)
        x_32s = self.resnet50_32s.layer4(x_16s)

        return x_8s, x_16s, x_32s


class FeatureExtracterPyramid(FeatureExtracter):
    def __init__(self):
        super().__init__()
        self.resnet_extract_features = super().extract_features

    def extract_features(self, x_input):
        x_8s, x_16s, x_32s = self.resnet_extract_features(x_input)
        
        for s in [0.25, 0.5, 2.0]:
           x_8s_scaled, x_16s_scaled, x_32s_scaled = self.resnet_extract_features(F.interpolate(input=x_input, scale_factor=s, mode='nearest'))
           _,_,h, w = x_8s.shape
           x_8s  = x_8s + F.interpolate(x_8s_scaled, size=(h,w), mode='nearest')
           
           _,_,h, w = x_16s.shape
           x_16s = x_16s + F.interpolate(x_16s_scaled, size=(h,w), mode='nearest')
           
           _,_,h, w = x_32s.shape
           x_32s = x_32s + F.interpolate(x_32s_scaled, size=(h,w), mode='nearest')

        return x_8s, x_16s, x_32s




class FeatureExtracterDilated(FeatureExtracter):
    def __init__(self):
        super().__init__()
        from models.dilated import resnet50
        # resnet50_32s = torchvision.models.resnet50(pretrained=True)
        resnet50_32s = resnet50(pretrained=True, dilated=True)
        resnet50_32s.fc = nn.Sequential()
        self.resnet50_32s = resnet50_32s
        self.expansion_rate = resnet50_32s.layer1[0].expansion
        self.resnet50_32s.fc = nn.Sequential()
        # # FREEZE BATCH NORMS
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad = False
                m.bias.requires_grad = False
