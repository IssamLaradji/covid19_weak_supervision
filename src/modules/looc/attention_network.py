import torch.nn as nn
import torchvision
from haven import haven_img
import torch
import cv2
import numba
# from . import utils as ut
from haven import haven_img as hi
from skimage import morphology as morph
import numpy as nopython
import torch.utils.model_zoo as model_zoo
# from . import _losses as _losses
from torch import optim
import torch.nn.functional as F
from torchvision.transforms import functional as FT
# from . import lcs_utils
# from models.scoring_net import lcs_utils
from torchvision.models.detection.rpn import concat_box_prediction_layers
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import torch
import numpy as np
import torch.nn.functional as F
import copy
from pycocotools import mask as maskUtils
# from datasets import _transforms
from torchvision.transforms import functional as FT
import numba
import torch.nn as nn
import torchvision
from haven import haven_utils as hu
import torch
# from haven._toolbox import misc as ms
from skimage import morphology as morph
import numpy as np
# from haven.managers import exp_manager as em
import torch.utils.model_zoo as model_zoo
from torchvision.models import detection 
# from . import _losses as _losses
from torch import optim
import torch.nn.functional as F
# from maskrcnn_benchmark.structures.image_list import to_image_list
from src import models
from collections import OrderedDict
 
# import utils as ut

from scipy.spatial.distance import cdist
# import cvxpy as cp
# from cvxpylayers.torch import CvxpyLayer
import os


class AttModel(nn.Module):
    def __init__(self, exp_dict):
        super().__init__()
        self.att_dict = exp_dict['attention']

        self.rcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.rcnn.rpn.nms_thresh = 0.7

     
        self.box_score_method = self.att_dict['box_score_method']
        self.agg_score_method = self.att_dict['agg_score_method']
        self.threshold = self.att_dict['agg_score_method']
    
    @torch.no_grad()
    def get_attention_dict(self, images_original, counts, probs, return_roi=True):
        
        self.eval()

        assert(counts > 0)

        # 1. get bbox and their scores
        _, H_img, W_img = images_original[0].shape
        images, targets = self.rcnn.transform([images_original.cuda()[0]], None)
        bbox, obj_scores, fpn_features = rpn_forward(self.rcnn, images)
        
        # 2. filter based on max area
        # ---------------------------
        bbox, obj_scores = filter_by_area(bbox, obj_scores, images, counts=counts, T=0.01)

        # box_score_method
        if self.box_score_method == 'mean':
            box_scores = score_bbox_heatmap(probs, bbox, images, 
                                    reduction='mean')

        elif self.box_score_method == 'center':
            box_scores = score_bbox_heatmap(probs, bbox, images, 
                                    reduction='center')

        elif self.box_score_method == 'objectness':
            box_scores = obj_scores
        else:
            stop

        # agg_score_method
        if self.agg_score_method == 'max':
            scores = torch.max(obj_scores, box_scores)
        elif self.agg_score_method == 'mean':
            scores = (obj_scores + box_scores) / 2.
        elif self.agg_score_method == 'replace_0.8':
            scores[box_scores>0.8] = box_scores[box_scores>0.8]
        elif self.agg_score_method == 'replace_0.7':
            scores[box_scores>0.7] = box_scores[box_scores>0.7]
        elif self.agg_score_method == 'replace':
            scores = box_scores
        else:
            stop

        print()

        # 3. rescore bbox and sort
        # if np.random.rand() < 0.3:
        ind = scores.argsort(descending=True)
        bbox, scores = bbox[ind], scores[ind]
        
        scores_top = copy.deepcopy(scores)
        
        # 5. normalize bbox
        bbox_norm = normalize_bbox(bbox.clone(), images)

        # 4. get top bbox 
        ind_fg = get_top_bbox(counts, bbox, scores_top, T=0.01, ratio_curr=1.0)
        if np.random.rand() <= self.att_dict['select_prob']:
        # if np.random.rand() <= 1.:
            end = max(int((scores[ind_fg] > 0.7).float().sum()), 1)
            ind_fg = ind_fg[:end]

            # get roi_mask 
            mask_bg = 1 - bbox_to_mask(bbox_norm, H_img, W_img)
            mask_fg = bbox_to_mask(bbox_norm[ind_fg], H_img, W_img)
            roi_mask = mask_fg + mask_bg

        else:
            roi_mask = None
        
        points = bbox2points(bbox_norm[ind_fg], H_img, W_img)

        return {
                # 'bbox_norm':bbox_norm,
                # 'scores': scores,
                # 'ind_fg':ind_fg,
                'points':torch.LongTensor(points[None]),
                'roi_mask':roi_mask
                }

        
def score_bbox_points(probs, points, bbox, batch, images):
    
    if points.sum() == 0:
        return torch.zeros(bbox.shape[0]).cuda()

    points_probs = points * probs.cpu().numpy()

    Hp, Wp = points.squeeze().shape
    yp, xp = np.where(points.squeeze())
    points_probs_list = points_probs.squeeze()[yp, xp]
    bbox_norm = normalize_bbox(bbox.clone(), images)
    xmin, ymin, xmax, ymax = torch.chunk(bbox_norm, 4, dim=1)
    yp = torch.from_numpy(yp).cuda().float()[None] / Hp
    xp = torch.from_numpy(xp).cuda().float()[None] / Wp
    
    D = (xp - xmin) >= 0 
    D &= (yp - ymin) >= 0 
    D &= (xmax - xp) >= 0 
    D &= (ymax - yp) >= 0

    scores = (D.float() * torch.FloatTensor(points_probs_list)[None].cuda())
    scores = scores.max(dim=1)[0]
    
    return scores


def get_roi_mask(bbox, ind_fg, mask_bg, images):
    H, W = mask_bg.shape
    bbox_norm = normalize_bbox(bbox.clone(), images)
    mask_fg = bbox_to_mask(bbox_norm[ind_fg], H, W)

    roi_mask = ((mask_bg) + mask_fg)

    return roi_mask


def bbox_on_image(batch, bbox_norm, scores, ind_fg):
    img_org = hu.denormalize(batch['images'], mode='rgb') 
    img_org = img_org / img_org.max()
    img_box = bbox_on_image(bbox_norm[ind_fg], img_org.squeeze(), text_list=scores)
    
    return np.hstack([img_box])

@torch.no_grad()
def score_bbox_heatmap(probs, bbox, images, reduction):
    # import ipdb; ipdb.set_trace()
    # points = lcfcn.predict_on_batch(batch, method='points')
    bbox_norm = normalize_bbox(bbox.clone(), images)
    scores = torch.zeros(bbox.shape[0]).cuda()

    H, W = probs.shape
    for i, bb in enumerate(bbox_norm):
        xmin, ymin, xmax, ymax = bb
        
        prob_bb = probs[int(ymin*H):int(ymax*H), int(xmin*W):int(xmax*W)]
        if reduction == 'mean':
            scores[i] = torch.mean(prob_bb)
        elif reduction == 'center':
            hi, wi = prob_bb.shape
            scores[i] = prob_bb[hi//2, wi//2]
        elif reduction == 'max':
            scores[i] = torch.max(prob_bb)
        else:
            raise ValueError('heatmap_method does not exist')


    return scores

    
    if points.sum() == 0:
        return torch.zeros(bbox.shape[0]).cuda()

    points_probs = points * probs.cpu().numpy()

    Hp, Wp = points.squeeze().shape
    yp, xp = np.where(points.squeeze())
    points_probs_list = points_probs.squeeze()[yp, xp]
    bbox_norm = normalize_bbox(bbox.clone(), images)
    xmin, ymin, xmax, ymax = torch.chunk(bbox_norm, 4, dim=1)
    yp = torch.from_numpy(yp).cuda().float()[None] / Hp
    xp = torch.from_numpy(xp).cuda().float()[None] / Wp
    
    D = (xp - xmin) >= 0 
    D &= (yp - ymin) >= 0 
    D &= (xmax - xp) >= 0 
    D &= (ymax - yp) >= 0

    scores = (D.float() * torch.FloatTensor(points_probs_list)[None].cuda())
    scores = scores.max(dim=1)[0]
    
    return scores


def filter_by_area(bbox, scores, images, counts, T, top_void=None):
    ind_fg = get_top_bbox(counts, bbox, scores, T=T, top_void=top_void)
    bbox_norm = normalize_bbox(bbox.clone(), images)
    areas = get_bbox_areas(bbox_norm)
    if len(ind_fg):

        max_area = areas[ind_fg].max()
    else:
        max_area = 0
    ind = areas <= max_area
    bbox, scores = bbox[ind], scores[ind]

    return bbox, scores


def normalize_bbox(bbox, images):
    H, W = images.image_sizes[0]

    bbox[:, [0,2]] = bbox[:, [0,2]] / W
    bbox[:, [1,3]] = bbox[:, [1,3]] / H

    return bbox

def bbox2points(bbox, H, W):
    points = np.zeros((H, W), dtype=int)
    for bb in bbox:
        x1, y1, x2, y2 = bb
        x1, y1 = (int(x1*W), int(y1*H),) 
        x2, y2 = (int(x2*W), int(y2*H),)

        xc, yc = (x2 + x1)//2, (y2 + y1)//2
        points[int(yc), int(xc)] = 1

    return points


def rpn_forward(rcnn, images):
    features = rcnn.backbone(images.tensors)

    if isinstance(features, torch.Tensor):
        features = OrderedDict([('0', features)])

    features_list = list(features.values())

    objectness, pred_bbox_deltas = rcnn.rpn.head(features_list)
    anchors = rcnn.rpn.anchor_generator(images, features_list)

    num_images = len(anchors)
    num_anchors_per_level = [o[0].numel() for o in objectness]
    objectness, pred_bbox_deltas = \
        concat_box_prediction_layers(objectness, pred_bbox_deltas)

    proposals = rcnn.rpn.box_coder.decode(pred_bbox_deltas.detach(), anchors)
    proposals = proposals.view(num_images, -1, 4)
    boxes, scores = rcnn.rpn.filter_proposals(proposals, objectness, images.image_sizes, 
                            num_anchors_per_level)
    

    # normalize boxes
    bbox = boxes[0]

    # normalize scores
    scores = scores[0]
    scores += abs(scores.min())
    scores /= scores.max()

    return bbox, scores, features


def get_top_bbox(counts, bbox, scores,  T, top_void=None, ratio_curr=None):
    if counts == 0:
        return [] 

    if ratio_curr:
        counts_top = max(1, int(counts * ratio_curr))
    elif top_void:
        counts_top = min(int(counts), (scores >= top_void).sum().item())
    else:
        counts_top = counts

    ind_fg = []
    while len(ind_fg) == 0 and T <= 1.0 and counts_top != 0:
        ind_fg = filter_c_wsl(bbox, scores, counts=counts_top, T=T)
        T = T *1.1

    # ind_bg = np.setdiff1d(np.arange(bbox.shape[0]), ind_fg)

    return ind_fg


def filter_c_wsl(bbox, scores, counts, T=0.5):
    # assert ordered
    # assert(1 == torch.mean((torch.arange(bbox.shape[0]) == scores.argsort(descending=True).cpu()).float()).item())
    bbox_overlaps = torchvision.ops.box_iou(bbox, bbox)
        
    ind = c_wsl_fast(bbox_overlaps.cpu().numpy(), scores.cpu().numpy(), 
                        int(counts), T=T)

    return ind

@numba.jit(nopython=True)
def c_wsl_fast(bbox_overlaps, scores, counts, T=0.5):
    N = scores.shape[0]

    G_score_best = 0

    G_matrix = np.ones((N, N), dtype=np.int32) * -1
    G_size_vector = np.zeros(N, dtype=np.int32)

    # Find best proposals
    for i in range(N):
        box_i = i
        G = G_matrix[i]
        G_size = G_size_vector[i]

        G[0] = box_i
        G_size += 1
        G_score = scores[box_i]

        # Find the best compatible proposals to add
        for j in range(i + 1, N):
            # Check if we selected enough proposals
            if G_size == counts:  # j == N-1?
                # Check if the total score is hier than the previous one
                if G_score >= G_score_best:
                    G_score_best = G_score
                    G_best = G
                    G_size_best = G_size
                break

            box_j = j
            # Check if this proposal is compatible with the selected ones in G
            compatible = True
            for k in range(G_size):
                box_k = G[k]

                overlapping = bbox_overlaps[box_k, box_j]

                if overlapping > T:
                    compatible = False
                    break

            if not compatible:
                continue

            # Add this proposal to the list of selected ones
            G[G_size] = box_j
            G_size += 1
            G_score += scores[j]


    best_ind = G_best[:G_size_best]
    return best_ind


import torch
import tqdm
import argparse
import pandas as pd
import pickle, os
import numpy as np
from torch.utils.data import sampler
import json
import cv2
from haven import haven_utils as hu


def get_bbox_areas(bbox_norm):
    area_list = np.ones(bbox_norm.shape[0])*-1

    for i, bb in enumerate(bbox_norm):
        x1, y1, x2, y2 = bb

        area = (y2-y1) * (x2-x1)
        area_list[i] = area

    return area_list

def bbox_to_mask(bbox_norm, H, W):
    mask = np.zeros((H, W))
    for i, bb in enumerate(bbox_norm):
        x1, y1, x2, y2 = bb
        x1, y1 = int(x1*W), int(y1*H) 
        x2, y2 = int(x2*W), int(y2*H)

        mask[y1:y2, x1:x2] = 1
    return mask

def bbox_on_image(bbox_xyxy, image, text_list=None):
    image = hu.f2l(image.squeeze())
    image_uint8 = (image*254).astype("uint8").copy()
    H, W, _ = image.shape
    for i, bb in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = bb
        x1, y1 = int(x1*W), int(y1*H) 
        x2, y2 = int(x2*W), int(y2*H)
        
        # Blue color in BGR 
        color = (255, 0, 0) 
        
        # Line thickness of 2 px 
        thickness = 2
        
        # Using cv2.rectangle() method 
        # Draw a rectangle with blue line borders of thickness of 2 px 
        image_uint8 = cv2.rectangle(image_uint8, (x1, y1), (x2, y2), color, thickness) 
        if text_list is not None:
            image_uint8 = cv2.putText(image_uint8, str(int(text_list[i]*100)), 
                        (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,  0.6, (0, 0, 255) , 1, cv2.LINE_AA) 
    # hu.save_image("/mnt/datasets/public/issam/prototypes/wscl/tmp.jpg", image_uint8)
    return image_uint8 / 255.
