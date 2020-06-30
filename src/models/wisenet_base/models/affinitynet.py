import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch
import torch.nn as nn
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch.autograd import Variable
import numpy as np 
import misc as ms
from skimage import morphology as morph
from torch.autograd import Function
from losses import saliency
from core import blobs_utils as bu
# from core import proposals as prp
from . import base_model as bm
from addons.pycocotools import mask as maskUtils
# from core import score_functions as sf
import ann_utils as au


import torch
from torch import nn
import numpy as np

import torch.nn.functional as F


import torch
import torchvision
import argparse
import importlib
import numpy as np

from torch.utils.data import DataLoader
import scipy.misc
import torch.nn.functional as F
import os.path

import torch
import torch.nn as nn
import torch.sparse as sparse
import torch.nn.functional as F

from models import lcfcn

class WiseAffinity(bm.BaseModel):
    def __init__(self, train_set, **model_options):
        super().__init__(train_set, **model_options)

        self.aff =  AffinityHead()
        self.aff.load_state_dict(torch.load("/mnt/datasets/public/issam/res38_aff.pth"))
        self.beta = 8.0
        self.logt = 8
        self.alpha = 16.0

        # self.cam = CAMHead()
        self.lcfcn = lcfcn.LCFCN_BO(train_set)
        path = "/mnt/projects/counting/Saves/main//dataset:Pascal2012_model:LCFCN_BO_metric:MAE_loss:lcfcnLoss_config:wtp//State_Dicts/best_model.pth"
        self.lcfcn.load_state_dict(torch.load(path))
        cropsize = 448
        radius = 5
        self.extract_aff_labels = ExtractAffinityLabelInRadius(cropsize=cropsize//8, radius=radius)

        # # FREEZE BATCH NORMS
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    @torch.no_grad()
    def predict(self, batch, predict_method="counts"):
        self.eval()
        img = batch["images"]
        
        padded_size = (int(np.ceil(img.shape[2]/8)*8), int(np.ceil(img.shape[3]/8)*8))
        p2d = (0, padded_size[1] - img.shape[3], 0, padded_size[0] - img.shape[2])
        img = F.pad(img, p2d)

        dheight = int(np.ceil(img.shape[2]/8))
        dwidth = int(np.ceil(img.shape[3]/8))
        n, c, h, w = img.shape

        lcfcn_pointList = au.mask2pointList(batch["points"])["pointList"]
        counts = np.zeros(self.n_classes-1)
        if len(lcfcn_pointList) == 0:
            return {"blobs": np.zeros((h,w), int), "annList":[], "counts":counts}


        propDict = au.pointList2propDict(lcfcn_pointList, batch, 
                                        proposal_type="sharp",
                                             thresh=0.5)

        aff_mat = torch.pow(self.aff.forward(img.cuda(), True), self.beta)
        trans_mat = aff_mat / torch.sum(aff_mat, dim=0, keepdim=True)

        for _ in range(self.logt):
            trans_mat = torch.matmul(trans_mat, trans_mat)

        import ipdb; ipdb.set_trace()  # breakpoint ac0c04d2 //

        for prop in propDict["propDict"]:
            mask = prop["annList"][0]["mask"]
            mask = torch.FloatTensor(mask)[None]
            mask = F.pad(mask, p2d)
            mask_arr =  F.avg_pool2d(mask, 8, 8)

            mask_vec = mask_arr.view(1, -1)

            mask_rw = torch.matmul(mask_vec.cuda(), trans_mat)
            mask_rw = mask_rw.view(1, dheight, dwidth)
            mask_rw = torch.nn.Upsample((img.shape[2], img.shape[3]), mode='bilinear')(mask_rw[None])
        
        import ipdb; ipdb.set_trace()  # breakpoint 89e7f819 //

        cam_rw = torch.nn.Upsample((img.shape[2], img.shape[3]), mode='bilinear')(cam_rw)
        _, cam_rw_pred = torch.max(cam_rw, 1)

        res = np.uint8(cam_rw_pred.cpu().data[0])[:h, :w]

        if predict_method == "annList":
            pass
        else:
            return img, res 
        # scipy.misc.imsave(os.path.join(args.out_rw, name + '.png'), res)


    @torch.no_grad()
    def visualize(self, batch, predict_method="counts"):
        img, res  = self.predict(batch)
        ms.images(img, res, denorm=1)

class AffinityNetBasic(bm.BaseModel):
    def __init__(self, train_set, **model_options):
        super().__init__(train_set, **model_options)

        self.aff =  AffinityHead()
        self.aff.load_state_dict(torch.load("/mnt/datasets/public/issam/res38_aff.pth"))
        self.beta = 8.0
        self.logt = 8
        self.alpha = 16.0

        # self.cam = CAMHead()
        self.lcfcn = lcfcn.LCFCN(train_set)
        path = "/mnt/projects/counting/Saves/main//dataset:Pascal2012_model:LCFCN_BO_metric:MAE_loss:lcfcnLoss_config:wtp//State_Dicts/best_model.pth"
        self.lcfcn.load_state_dict(torch.load(path))
        cropsize = 448
        radius = 5
        self.extract_aff_labels = ExtractAffinityLabelInRadius(cropsize=cropsize//8, radius=radius)

        # # FREEZE BATCH NORMS
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad = False
                m.bias.requires_grad = False



    @torch.no_grad()
    def predict(self, batch, predict_method="counts"):
        self.eval()
        img = batch["images"]
        
        padded_size = (int(np.ceil(img.shape[2]/8)*8), int(np.ceil(img.shape[3]/8)*8))

        p2d = (0, padded_size[1] - img.shape[3], 0, padded_size[0] - img.shape[2])
        img = F.pad(img, p2d)

        dheight = int(np.ceil(img.shape[2]/8))
        dwidth = int(np.ceil(img.shape[3]/8))
        n, c, h, w = img.shape

        lcfcn_pointList = au.mask2pointList(batch["points"])["pointList"]
        counts = np.zeros(self.n_classes-1)
        if len(lcfcn_pointList) == 0:
            return {"blobs": np.zeros((h,w), int), "annList":[], "counts":counts}


        propDict = au.pointList2propDict(lcfcn_pointList, batch, 
                                        proposal_type="sharp",
                                             thresh=0.5)

        aff_mat = torch.pow(self.aff.forward(img.cuda(), True), self.beta)
        trans_mat = aff_mat / torch.sum(aff_mat, dim=0, keepdim=True)

        for _ in range(self.logt):
            trans_mat = torch.matmul(trans_mat, trans_mat)

        import ipdb; ipdb.set_trace()  # breakpoint ac0c04d2 //

        for prop in propDict["propDict"]:
            mask = prop["annList"][0]["mask"]
            mask = torch.FloatTensor(mask)[None]
            mask = F.pad(mask, p2d)
            mask_arr =  F.avg_pool2d(mask, 8, 8)

            mask_vec = mask_arr.view(1, -1)

            mask_rw = torch.matmul(mask_vec.cuda(), trans_mat)
            mask_rw = mask_rw.view(1, dheight, dwidth)
            mask_rw = torch.nn.Upsample((img.shape[2], img.shape[3]), mode='bilinear')(mask_rw[None])
        
        import ipdb; ipdb.set_trace()  # breakpoint 89e7f819 //

        cam_rw = torch.nn.Upsample((img.shape[2], img.shape[3]), mode='bilinear')(cam_rw)
        _, cam_rw_pred = torch.max(cam_rw, 1)

        res = np.uint8(cam_rw_pred.cpu().data[0])[:h, :w]

        if predict_method == "annList":
            pass
        else:
            return img, res 
        # scipy.misc.imsave(os.path.join(args.out_rw, name + '.png'), res)


    @torch.no_grad()
    def visualize(self, batch, predict_method="counts"):
        img, res  = self.predict(batch)
        ms.images(img, res, denorm=1)

class AffinityNet(bm.BaseModel):
    def __init__(self, train_set, **model_options):
        super().__init__(train_set, **model_options)

        self.aff =  AffinityHead()
        self.aff.load_state_dict(torch.load("/mnt/datasets/public/issam/res38_aff.pth"))
        self.beta = 8.0
        self.logt = 8
        self.alpha = 16.0

        self.cam = CAMHead()
        self.cam.load_state_dict(torch.load("/mnt/datasets/public/issam/res38_cls.pth"))
        self.lcfcn = lcfcn.LCFCN(train_set)
        path = "/mnt/projects/counting/Saves/main//dataset:Pascal2012_model:LCFCN_BO_metric:MAE_loss:lcfcnLoss_config:wtp//State_Dicts/best_model.pth"
        self.lcfcn.load_state_dict(torch.load(path))
        self.extract_aff_labels = ExtractAffinityLabelInRadius()

        # # FREEZE BATCH NORMS
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad = False
                m.bias.requires_grad = False



    @torch.no_grad()
    def predict(self, batch, predict_method="counts"):
        self.eval()
        img = batch["images"]
        
        padded_size = (int(np.ceil(img.shape[2]/8)*8), int(np.ceil(img.shape[3]/8)*8))

        p2d = (0, padded_size[1] - img.shape[3], 0, padded_size[0] - img.shape[2])
        img = F.pad(img, p2d)

        dheight = int(np.ceil(img.shape[2]/8))
        dwidth = int(np.ceil(img.shape[3]/8))
        n, c, h, w = img.shape

        label = (batch["counts"]>0)
        ###### CAM
        # cam = np.load(os.path.join(args.cam_dir, name + '.npy')).item()
        
        cam = self.cam.forward_cam(img.cuda())
        cam = F.interpolate(cam, (h,w), mode='bilinear', align_corners=False)[0]
        #  ms.images(ms.gray2cmap(cam[18]))
        cam = cam.cpu().numpy() * label.clone().view(20, 1, 1).numpy()
        sum_cam =cam
        norm_cam = sum_cam / (np.max(sum_cam, (1, 2), keepdims=True) + 1e-5)

        cam_dict = {}

        for i in range(20):
            if label.squeeze()[i].item() > 1e-5:
                cam_dict[i] = norm_cam[i]

        # bg_score = [np.ones_like(norm_cam[0])*0.2]
        # pred_cam = np.argmax(np.concatenate((bg_score, norm_cam)), 0)

        # ms.images(img, pred_cam, denorm=1)
        ######
        cam_full_arr = np.zeros((21, h, w), np.float32)

        for k, v in cam_dict.items():
            cam_full_arr[k+1] = v

        cam_full_arr[0] = (1 - np.max(cam_full_arr[1:], (0), keepdims=False))**self.alpha
        cam_full_arr = np.pad(cam_full_arr, ((0, 0), (0, p2d[3]), (0, p2d[1])), mode='constant')

        aff_mat = torch.pow(self.aff.forward(img.cuda(), True), self.beta)
        trans_mat = aff_mat / torch.sum(aff_mat, dim=0, keepdim=True)
        for _ in range(self.logt):
            trans_mat = torch.matmul(trans_mat, trans_mat)

        cam_full_arr = torch.from_numpy(cam_full_arr)
        cam_full_arr = F.avg_pool2d(cam_full_arr, 8, 8)

        cam_vec = cam_full_arr.view(21, -1)

        cam_rw = torch.matmul(cam_vec.cuda(), trans_mat)
        cam_rw = cam_rw.view(1, 21, dheight, dwidth)

        cam_rw = torch.nn.Upsample((img.shape[2], img.shape[3]), mode='bilinear')(cam_rw)
        _, cam_rw_pred = torch.max(cam_rw, 1)
        # ms.images(cam_rw)
        import ipdb; ipdb.set_trace()  # breakpoint 723d8a42 //
        cam_full_rw = torch.nn.Upsample((img.shape[2], img.shape[3]), mode='bilinear')(cam_full_arr[None])
        # ms.images(img, cam_full_rw.max(1)[1], denorm=1)
        # ms.images(img, cam_rw.max(1)[1], denorm=1)
        res = np.uint8(cam_rw_pred.cpu().data[0])[:h, :w]

        if predict_method == "annList":
            pass
        else:
            return img, res 
        # scipy.misc.imsave(os.path.join(args.out_rw, name + '.png'), res)


    @torch.no_grad()
    def visualize(self, batch, predict_method="counts"):
        img, res  = self.predict(batch)
        ms.images(img, res, denorm=1)
# import network.resnet38d
# from tool import pyutils
# class CAMHead(nn.Module):
#     def __init__(self, num_classes):
#         super().__init__()
#         self.fc8 = nn.Conv2d(2048, num_classes-1, 1, bias=False)
#         torch.nn.init.xavier_uniform_(self.fc8.weight)


#         # # FREEZE BATCH NORMS
#         for m in self.modules():
#             if isinstance(m, nn.BatchNorm2d):
#                 m.weight.requires_grad = False
#                 m.bias.requires_grad = False
                

#     def forward(self, x):
#         x = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=0)

#         x = self.fc8(x)
#         x = x.view(x.size(0), -1)

#         return x

#     def forward_cam(self, x):
#         x = F.conv2d(x, self.fc8.weight)
#         x = F.relu(x)

#         return x




class CAM(bm.BaseModel):
    def __init__(self, train_set, **model_options):
        super().__init__(train_set, **model_options)

        self.feature_extracter = bm.FeatureExtracter()
        self.cam = CAMHead(train_set.n_classes)

        # # FREEZE BATCH NORMS
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    def forward(self, x):
        x_8s, x_16s, x = self.feature_extracter.extract_features(x)
        x = self.cam(x)
        return x

    def forward_cam(self, x):
        x_8s, x_16s, x = self.feature_extracter.extract_features(x)
        x = self.cam.forward_cam(x)

        return x

    @torch.no_grad()
    def visualize(self, batch, cam_index=None):
        cam = ms.resizeTo(self.forward_cam(batch["images"].cuda()),batch["images"])
        preds = self.predict(batch, "counts")
        print(preds)
        if cam_index is None:
            cam_index = preds["indices"][0]
        image_points = ms.get_image(batch["images"], (batch["points"]==(cam_index+1)).long(), denorm=1, enlarge=1)
        ms.images(image_points[0],  ms.gray2cmap(ms.t2n(cam[:,cam_index])))


    @torch.no_grad()
    def predict(self, batch, predict_method="counts"):
        self.sanity_checks(batch)

        self.eval()
        # ms.reload(pm)
        # self.predict_dict = ms.get_functions(pm)
        if predict_method == "counts":
            probs = torch.sigmoid(self(batch["images"].cuda())).data
            counts = probs>0.5
            return {"counts":counts, "indices":np.where(counts!=0)[1].tolist()}

        elif predict_method == "probs":
            probs = F.softmax(self(batch["images"].cuda()),dim=1).data
            return {"probs":probs}

        elif predict_method == "points":
            probs = F.softmax(self(batch["images"].cuda()),dim=1).data
            blob_dict = bu.probs2blobs(probs)

            return {"points":blob_dict["points"], 
                    "pointList":blob_dict["pointList"],
                    "probs":probs}
            

        elif predict_method == "blobs":
            probs = F.softmax(self(batch["images"].cuda()),dim=1).data
            blob_dict = bu.probs2blobs(probs)
            
            return blob_dict

        else:
            print("Used predict method {}".format(predict_method))
            return self.predict_dict[predict_method](self, batch)







class Hybrid(bm.BaseModel):
    
    def __init__(self, train_set, **model_options):
        super().__init__(train_set, **model_options)

        self.feature_extracter = bm.FeatureExtracter()

        
        self.affinity = AffinityHead(train_set.n_classes)
        self.cam = CAMHead(train_set.n_classes)
        # # FREEZE BATCH NORMS
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad = False
                m.bias.requires_grad = False

        # self.affinity = AffinityNet(train_set, **model_options)
                

    def forward(self, x):
        x_8s, x_16s, x = self.feature_extracter.extract_features(x)
        # x = super().forward(x)
        # x = self.dropout7(x)
        x = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=0)

        x = self.fc8(x)
        x = x.view(x.size(0), -1)
        import ipdb; ipdb.set_trace()  # breakpoint 3e137813 //

        return x

    def forward_cam(self, x):
        x_8s, x_16s, x = self.feature_extracter.extract_features(x)

        x = F.conv2d(x, self.fc8.weight)
        x = F.relu(x)

        return x
#     def get_parameter_groups(self):
#         groups = ([], [], [], [])

#         for m in self.modules():

#             if (isinstance(m, nn.Conv2d) or isinstance(m, nn.modules.normalization.GroupNorm)):

#                 if m.weight.requires_grad:
#                     if m in self.from_scratch_layers:
#                         groups[2].append(m.weight)
#                     else:
#                         groups[0].append(m.weight)

#                 if m.bias is not None and m.bias.requires_grad:

#                     if m in self.from_scratch_layers:
#                         groups[3].append(m.bias)
#                     else:
#                         groups[1].append(m.bias)

#         return groups




def get_indices_of_pairs(radius, size):

    search_dist = []

    for x in range(1, radius):
        search_dist.append((0, x))

    for y in range(1, radius):
        for x in range(-radius + 1, radius):
            if x * x + y * y < radius * radius:
                search_dist.append((y, x))

    radius_floor = radius - 1

    full_indices = np.reshape(np.arange(0, size[0]*size[1], dtype=np.int64),
                                   (size[0], size[1]))

    cropped_height = size[0] - radius_floor
    cropped_width = size[1] - 2 * radius_floor

    indices_from = np.reshape(full_indices[:-radius_floor, radius_floor:-radius_floor],
                              [-1])

    indices_to_list = []

    for dy, dx in search_dist:
        indices_to = full_indices[dy:dy + cropped_height,
                     radius_floor + dx:radius_floor + dx + cropped_width]
        indices_to = np.reshape(indices_to, [-1])

        indices_to_list.append(indices_to)

    concat_indices_to = np.concatenate(indices_to_list, axis=0)

    return indices_from, concat_indices_to


class ExtractAffinityLabelInRadius():

    def __init__(self, cropsize, radius=5):
        self.radius = radius

        self.search_dist = []

        for x in range(1, radius):
            self.search_dist.append((0, x))

        for y in range(1, radius):
            for x in range(-radius+1, radius):
                if x*x + y*y < radius*radius:
                    self.search_dist.append((y, x))

        self.radius_floor = radius-1

        self.crop_height = cropsize - self.radius_floor
        self.crop_width = cropsize - 2 * self.radius_floor
        return

    def __call__(self, label):

        labels_from = label[:-self.radius_floor, self.radius_floor:-self.radius_floor]
        labels_from = np.reshape(labels_from, [-1])

        labels_to_list = []
        valid_pair_list = []

        for dy, dx in self.search_dist:
            labels_to = label[dy:dy+self.crop_height, self.radius_floor+dx:self.radius_floor+dx+self.crop_width]
            labels_to = np.reshape(labels_to, [-1])

            valid_pair = np.logical_and(np.less(labels_to, 255), np.less(labels_from, 255))

            labels_to_list.append(labels_to)
            valid_pair_list.append(valid_pair)

        bc_labels_from = np.expand_dims(labels_from, 0)
        concat_labels_to = np.stack(labels_to_list)
        concat_valid_pair = np.stack(valid_pair_list)

        pos_affinity_label = np.equal(bc_labels_from, concat_labels_to)

        bg_pos_affinity_label = np.logical_and(pos_affinity_label, np.equal(bc_labels_from, 0)).astype(np.float32)

        fg_pos_affinity_label = np.logical_and(np.logical_and(pos_affinity_label, np.not_equal(bc_labels_from, 0)), concat_valid_pair).astype(np.float32)

        neg_affinity_label = np.logical_and(np.logical_not(pos_affinity_label), concat_valid_pair).astype(np.float32)

        return torch.from_numpy(bg_pos_affinity_label), torch.from_numpy(fg_pos_affinity_label), torch.from_numpy(neg_affinity_label)


def get_indices_in_radius(height, width, radius):

    search_dist = []
    for x in range(1, radius):
        search_dist.append((0, x))

    for y in range(1, radius):
        for x in range(-radius+1, radius):
            if x*x + y*y < radius*radius:
                search_dist.append((y, x))

    full_indices = np.reshape(np.arange(0, height * width, dtype=np.int64),
                              (height, width))
    radius_floor = radius-1
    cropped_height = height - radius_floor
    cropped_width = width - 2 * radius_floor

    indices_from = np.reshape(full_indices[:-radius_floor, radius_floor:-radius_floor], [-1])

    indices_from_to_list = []

    for dy, dx in search_dist:

        indices_to = full_indices[dy:dy + cropped_height, radius_floor + dx:radius_floor + dx + cropped_width]
        indices_to = np.reshape(indices_to, [-1])

        indices_from_to = np.stack((indices_from, indices_to), axis=1)

        indices_from_to_list.append(indices_from_to)

    concat_indices_from_to = np.concatenate(indices_from_to_list, axis=0)

    return concat_indices_from_to




class ResBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride=1, first_dilation=None, dilation=1):
        super(ResBlock, self).__init__()

        self.same_shape = (in_channels == out_channels and stride == 1)

        if first_dilation == None: first_dilation = dilation

        self.bn_branch2a = nn.BatchNorm2d(in_channels)

        self.conv_branch2a = nn.Conv2d(in_channels, mid_channels, 3, stride,
                                       padding=first_dilation, dilation=first_dilation, bias=False)

        self.bn_branch2b1 = nn.BatchNorm2d(mid_channels)

        self.conv_branch2b1 = nn.Conv2d(mid_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False)

        if not self.same_shape:
            self.conv_branch1 = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)

    def forward(self, x, get_x_bn_relu=False):

        branch2 = self.bn_branch2a(x)
        branch2 = F.relu(branch2)

        x_bn_relu = branch2

        if not self.same_shape:
            branch1 = self.conv_branch1(branch2)
        else:
            branch1 = x

        branch2 = self.conv_branch2a(branch2)
        branch2 = self.bn_branch2b1(branch2)
        branch2 = F.relu(branch2)
        branch2 = self.conv_branch2b1(branch2)

        x = branch1 + branch2

        if get_x_bn_relu:
            return x, x_bn_relu

        return x

    def __call__(self, x, get_x_bn_relu=False):
        return self.forward(x, get_x_bn_relu=get_x_bn_relu)

class ResBlock_bot(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1, dropout=0.):
        super(ResBlock_bot, self).__init__()

        self.same_shape = (in_channels == out_channels and stride == 1)

        self.bn_branch2a = nn.BatchNorm2d(in_channels)
        self.conv_branch2a = nn.Conv2d(in_channels, out_channels//4, 1, stride, bias=False)

        self.bn_branch2b1 = nn.BatchNorm2d(out_channels//4)
        self.dropout_2b1 = torch.nn.Dropout2d(dropout)
        self.conv_branch2b1 = nn.Conv2d(out_channels//4, out_channels//2, 3, padding=dilation, dilation=dilation, bias=False)

        self.bn_branch2b2 = nn.BatchNorm2d(out_channels//2)
        self.dropout_2b2 = torch.nn.Dropout2d(dropout)
        self.conv_branch2b2 = nn.Conv2d(out_channels//2, out_channels, 1, bias=False)

        if not self.same_shape:
            self.conv_branch1 = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)

    def forward(self, x, get_x_bn_relu=False):

        branch2 = self.bn_branch2a(x)
        branch2 = F.relu(branch2)
        x_bn_relu = branch2

        branch1 = self.conv_branch1(branch2)

        branch2 = self.conv_branch2a(branch2)

        branch2 = self.bn_branch2b1(branch2)
        branch2 = F.relu(branch2)
        branch2 = self.dropout_2b1(branch2)
        branch2 = self.conv_branch2b1(branch2)

        branch2 = self.bn_branch2b2(branch2)
        branch2 = F.relu(branch2)
        branch2 = self.dropout_2b2(branch2)
        branch2 = self.conv_branch2b2(branch2)

        x = branch1 + branch2

        if get_x_bn_relu:
            return x, x_bn_relu

        return x

    def __call__(self, x, get_x_bn_relu=False):
        return self.forward(x, get_x_bn_relu=get_x_bn_relu)

class Normalize():
    def __init__(self, mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)):

        self.mean = mean
        self.std = std

    def __call__(self, img):
        imgarr = np.asarray(img)
        proc_img = np.empty_like(imgarr, np.float32)

        proc_img[..., 0] = (imgarr[..., 0] / 255. - self.mean[0]) / self.std[0]
        proc_img[..., 1] = (imgarr[..., 1] / 255. - self.mean[1]) / self.std[1]
        proc_img[..., 2] = (imgarr[..., 2] / 255. - self.mean[2]) / self.std[2]

        return proc_img

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1a = nn.Conv2d(3, 64, 3, padding=1, bias=False)

        self.b2 = ResBlock(64, 128, 128, stride=2)
        self.b2_1 = ResBlock(128, 128, 128)
        self.b2_2 = ResBlock(128, 128, 128)

        self.b3 = ResBlock(128, 256, 256, stride=2)
        self.b3_1 = ResBlock(256, 256, 256)
        self.b3_2 = ResBlock(256, 256, 256)

        self.b4 = ResBlock(256, 512, 512, stride=2)
        self.b4_1 = ResBlock(512, 512, 512)
        self.b4_2 = ResBlock(512, 512, 512)
        self.b4_3 = ResBlock(512, 512, 512)
        self.b4_4 = ResBlock(512, 512, 512)
        self.b4_5 = ResBlock(512, 512, 512)

        self.b5 = ResBlock(512, 512, 1024, stride=1, first_dilation=1, dilation=2)
        self.b5_1 = ResBlock(1024, 512, 1024, dilation=2)
        self.b5_2 = ResBlock(1024, 512, 1024, dilation=2)

        self.b6 = ResBlock_bot(1024, 2048, stride=1, dilation=4, dropout=0.3)

        self.b7 = ResBlock_bot(2048, 4096, dilation=4, dropout=0.5)

        self.bn7 = nn.BatchNorm2d(4096)

        self.not_training = [self.conv1a]

        self.normalize = Normalize()

        return

    def forward(self, x):
        return self.forward_as_dict(x)['conv6']

    def forward_as_dict(self, x):

        x = self.conv1a(x)

        x = self.b2(x)
        x = self.b2_1(x)
        x = self.b2_2(x)

        x = self.b3(x)
        x = self.b3_1(x)
        x = self.b3_2(x)

        x = self.b4(x)
        x = self.b4_1(x)
        x = self.b4_2(x)
        x = self.b4_3(x)
        x = self.b4_4(x)
        x = self.b4_5(x)

        x, conv4 = self.b5(x, get_x_bn_relu=True)
        x = self.b5_1(x)
        x = self.b5_2(x)

        x, conv5 = self.b6(x, get_x_bn_relu=True)

        x = self.b7(x)
        conv6 = F.relu(self.bn7(x))

        return dict({'conv4': conv4, 'conv5': conv5, 'conv6': conv6})




class CAMHead(Net):
    def __init__(self):
        super().__init__()

        self.dropout7 = torch.nn.Dropout2d(0.5)

        self.fc8 = nn.Conv2d(4096, 20, 1, bias=False)
        torch.nn.init.xavier_uniform_(self.fc8.weight)

        self.not_training = [self.conv1a, self.b2, self.b2_1, self.b2_2]
        self.from_scratch_layers = [self.fc8]
    
    def forward(self, x):
        x = super().forward(x)
        x = self.dropout7(x)

        x = F.avg_pool2d(
            x, kernel_size=(x.size(2), x.size(3)), padding=0)

        x = self.fc8(x)
        x = x.view(x.size(0), -1)

        return x

    def forward_cam(self, x):
        x = super().forward(x)

        x = F.conv2d(x, self.fc8.weight)
        x = F.relu(x)

        return x



class AffinityHead(Net):
    def __init__(self):

        super().__init__()
        self.f8_3 = torch.nn.Conv2d(512, 64, 1, bias=False)
        self.f8_4 = torch.nn.Conv2d(1024, 128, 1, bias=False)
        self.f8_5 = torch.nn.Conv2d(4096, 256, 1, bias=False)

        self.f9 = torch.nn.Conv2d(448, 448, 1, bias=False)
        
        torch.nn.init.kaiming_normal_(self.f8_3.weight)
        torch.nn.init.kaiming_normal_(self.f8_4.weight)
        torch.nn.init.kaiming_normal_(self.f8_5.weight)
        torch.nn.init.xavier_uniform_(self.f9.weight, gain=4)


        self.predefined_featuresize = int(448//8)

        self.ind_from, self.ind_to = get_indices_of_pairs(radius=5, size=(self.predefined_featuresize, self.predefined_featuresize))
        self.ind_from = torch.from_numpy(self.ind_from); self.ind_to = torch.from_numpy(self.ind_to)

        self.beta = 8.0
        self.logt = 8
        self.alpha = 16.0

        # # FREEZE BATCH NORMS
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    @torch.no_grad()
    def get_trans_mat(self, batch):        
        img = ms.pad_image(batch["images"])
        n, c, h, w = img.shape
        aff_mat = torch.pow(self.forward(img.cuda(), True), self.beta)
        trans_mat = aff_mat / torch.sum(aff_mat, dim=0, keepdim=True)

        for _ in range(self.logt):
            trans_mat = torch.matmul(trans_mat, trans_mat)

        return {"trans_mat":trans_mat, "h":h/8, "w":w/8}


    def forward_trans(self, images):        
        # img = ms.pad_image(images)
        # n, c, h, w = img.shape
        aff_mat = torch.pow(self.forward(images, False), self.beta)
        
        trans_mat = aff_mat / torch.sum(aff_mat, dim=0, keepdim=True)

        for _ in range(self.logt):
            trans_mat = torch.matmul(trans_mat, trans_mat)

        return trans_mat


    # @torch.no_grad()
    # def apply_affinity(self, batch, blob_probs=None):
        
    #     # img = ms.pad_image(batch["images"])
    #     img = batch["images"]
    #     n, c, h, w = img.shape

    #     dheight = int(np.ceil(img.shape[2]/8))
    #     dwidth = int(np.ceil(img.shape[3]/8))
        
    #     aff_mat = torch.pow(self.aff.forward(img.cuda(), True), self.beta)
    #     trans_mat = aff_mat / torch.sum(aff_mat, dim=0, keepdim=True)

    #     for _ in range(self.logt):
    #         trans_mat = torch.matmul(trans_mat, trans_mat)

        
    #     blob_probs_rw = torch.matmul(blob_probs.cuda(), trans_mat)
    #     blob_probs_rw = blob_probs_rw.view(1, dheight, dwidth)
    #     return blob_probs_rw

    def forward_as_dict(self, x):
        self.eval()
        x = self.conv1a(x)

        x = self.b2(x)
        x = self.b2_1(x)
        x = self.b2_2(x)

        x = self.b3(x)
        x = self.b3_1(x)
        x = self.b3_2(x)

        x = self.b4(x)
        x = self.b4_1(x)
        x = self.b4_2(x)
        x = self.b4_3(x)
        x = self.b4_4(x)
        x = self.b4_5(x)

        x, conv4 = self.b5(x, get_x_bn_relu=True)
        x = self.b5_1(x)
        x = self.b5_2(x)

        x, conv5 = self.b6(x, get_x_bn_relu=True)

        x = self.b7(x)
        conv6 = F.relu(self.bn7(x))

        return dict({'conv4': conv4, 'conv5': conv5, 'conv6': conv6})             
    
    def forward(self, x_input, to_dense=False):
        self.eval()
        d = self.forward_as_dict(x_input)

        f8_3 = F.elu(self.f8_3(d['conv4']))
        f8_4 = F.elu(self.f8_4(d['conv5']))
        f8_5 = F.elu(self.f8_5(d['conv6']))
        x = F.elu(self.f9(torch.cat([f8_3, f8_4, f8_5], dim=1)))

        if x.size(2) == self.predefined_featuresize and x.size(3) == self.predefined_featuresize:
            ind_from = self.ind_from
            ind_to = self.ind_to
        else:
            ind_from, ind_to = get_indices_of_pairs(5, (x.size(2), x.size(3)))
            ind_from = torch.from_numpy(ind_from); ind_to = torch.from_numpy(ind_to)

        x = x.view(x.size(0), x.size(1), -1)

        ff = torch.index_select(x, dim=2, index=ind_from.cuda(non_blocking=True))
        ft = torch.index_select(x, dim=2, index=ind_to.cuda(non_blocking=True))

        ff = torch.unsqueeze(ff, dim=2)
        ft = ft.view(ft.size(0), ft.size(1), -1, ff.size(3))

        aff = torch.exp(-torch.mean(torch.abs(ft-ff), dim=1))

        if to_dense:

            aff = aff.view(-1).cpu()

            ind_from_exp = torch.unsqueeze(ind_from, dim=0).expand(ft.size(2), -1).contiguous().view(-1)
            indices = torch.stack([ind_from_exp, ind_to])
            indices_tp = torch.stack([ind_to, ind_from_exp])

            area = x.size(2)
            indices_id = torch.stack([torch.arange(0, area).long(), torch.arange(0, area).long()])

            aff_mat = sparse.FloatTensor(torch.cat([indices, indices_id, indices_tp], dim=1),
                                      torch.cat([aff, torch.ones([area]), aff])).to_dense().cuda()

            return aff_mat

        else:
            aff_mat = torch.zeros((x.shape[-1],x.shape[-1])).cuda()
            aff = aff.view(-1)

            ind_from_exp = torch.unsqueeze(ind_from, dim=0).expand(ft.size(2), -1).contiguous().view(-1)
            indices = torch.stack([ind_from_exp, ind_to])
            indices_tp = torch.stack([ind_to, ind_from_exp])

            area = x.size(2)
            indices_id = torch.stack([torch.arange(0, area).long(), torch.arange(0, area).long()])
            rows_cols = torch.cat([indices.cuda(), indices_id.cuda(), indices_tp.cuda()], dim=1)
            values = torch.cat([aff, torch.ones([area]).cuda(), aff])
            aff_mat[rows_cols[0], rows_cols[1]] = values
            return aff_mat

            # return aff






class AFFNet(bm.BaseModel):
    def __init__(self, train_set, **model_options):
        from models.iprm import PRM
        super().__init__(train_set, **model_options)

        self.cam = CAMHead()        
        self.aff =  AffinityHead()
        self.cam.load_state_dict(torch.load("/mnt/datasets/public/issam/res38_cls.pth"))
        self.aff.load_state_dict(torch.load("/mnt/datasets/public/issam/res38_aff.pth"))
        self.prm = PRM(train_set)
        self.beta = 8.0
        self.logt = 8
        self.alpha = 16.0
    # def forward(self, x, to_dense=False):
    #     x_8s, x_16s, x_32s = self.feature_extracter.extract_features(x)
    #     import ipdb; ipdb.set_trace()  # breakpoint b2b482cc //

    #     self.aff([x_8s, x_16s, x_32s])

    @torch.no_grad()
    def predict(self, batch, predict_method="counts"):
        self.eval()
        img = batch["images"]
        
        padded_size = (int(np.ceil(img.shape[2]/8)*8), int(np.ceil(img.shape[3]/8)*8))

        p2d = (0, padded_size[1] - img.shape[3], 0, padded_size[0] - img.shape[2])
        img = F.pad(img, p2d)

        dheight = int(np.ceil(img.shape[2]/8))
        dwidth = int(np.ceil(img.shape[3]/8))
        n, c, h, w = img.shape
        label = (batch["counts"]>0)
        ###### CAM
        # cam = np.load(os.path.join(args.cam_dir, name + '.npy')).item()
        
        cam = self.cam.forward_cam(img.cuda())
        cam = F.interpolate(cam, (h,w), mode='bilinear', align_corners=False)[0]
        #  ms.images(ms.gray2cmap(cam[18]))
        cam = cam.cpu().numpy() * label.clone().view(20, 1, 1).numpy()
        sum_cam =cam
        norm_cam = sum_cam / (np.max(sum_cam, (1, 2), keepdims=True) + 1e-5)

        cam_dict = {}

        for i in range(20):
            if label.squeeze()[i].item() > 1e-5:
                cam_dict[i] = norm_cam[i]

        # bg_score = [np.ones_like(norm_cam[0])*0.2]
        # pred_cam = np.argmax(np.concatenate((bg_score, norm_cam)), 0)

        # ms.images(img, pred_cam, denorm=1)
        ######
        cam_full_arr = np.zeros((21, h, w), np.float32)

        for k, v in cam_dict.items():
            cam_full_arr[k+1] = v

        cam_full_arr[0] = (1 - np.max(cam_full_arr[1:], (0), keepdims=False))**self.alpha
        cam_full_arr = np.pad(cam_full_arr, ((0, 0), (0, p2d[3]), (0, p2d[1])), mode='constant')

        aff_mat = torch.pow(self.aff.forward(img.cuda(), True), self.beta)

        trans_mat = aff_mat / torch.sum(aff_mat, dim=0, keepdim=True)
        for _ in range(self.logt):
            trans_mat = torch.matmul(trans_mat, trans_mat)

        cam_full_arr = torch.from_numpy(cam_full_arr)
        cam_full_arr = F.avg_pool2d(cam_full_arr, 8, 8)

        cam_vec = cam_full_arr.view(21, -1)

        cam_rw = torch.matmul(cam_vec.cuda(), trans_mat)
        cam_rw = cam_rw.view(1, 21, dheight, dwidth)

        cam_rw = torch.nn.Upsample((img.shape[2], img.shape[3]), mode='bilinear')(cam_rw)
        _, cam_rw_pred = torch.max(cam_rw, 1)

        res = np.uint8(cam_rw_pred.cpu().data[0])[:h, :w]

        if predict_method == "annList":
            pass
        else:
            return img, res 
        # scipy.misc.imsave(os.path.join(args.out_rw, name + '.png'), res)


    @torch.no_grad()
    def visualize(self, batch, predict_method="counts"):
        img, res  = self.predict(batch)
        ms.images(img, res, denorm=1)


