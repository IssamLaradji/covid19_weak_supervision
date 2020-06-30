import os, pprint, tqdm
import numpy as np
import pandas as pd
from haven import haven_utils as hu 
from haven import haven_img as hi
import torch
import torch.nn as nn
import torch.nn.functional as F
from .networks import infnet, fcn8_vgg16
from . import semseg
from .losses import  density_loss
from src.modules.lcfcn import lcfcn_loss
from src import utils as ut
from skimage import morphology as morph
from kornia.geometry.transform import flips
class SemSegCounting(semseg.SemSeg):
    def __init__(self, exp_dict):
        super().__init__(exp_dict)
        

    def train_on_batch(self, batch):
        self.opt.zero_grad()

        images = batch["images"]
        images = images.cuda()
        
        logits = self.model_base(images)

        # compute loss
        loss_name = self.exp_dict['model']['loss']
        if loss_name == 'lcfcn_loss':
            points = batch['points'].cuda()
            loss = 0.
  
            for lg, pt in zip(logits, points):
                loss += lcfcn_loss.compute_loss((pt==1).long(), lg.sigmoid())

        elif loss_name == 'const_lcfcn_loss':
            points = batch['points'].cuda()
         
            
            logits_flip = self.model_base(flips.Hflip()(images))
            loss = torch.mean(torch.abs(flips.Hflip()(logits_flip)-logits))

            for lg, pt in zip(logits, points):
                loss += lcfcn_loss.compute_loss((pt==1).long(), lg.sigmoid())

        elif loss_name == 'point_loss':
            points = batch['points'].cuda()[:,None]
            ind = points!=255

            if ind.sum() == 0:
                loss = 0.
            else:
                loss = F.binary_cross_entropy_with_logits(logits[ind], 
                                        points[ind].float().cuda(), 
                                        reduction='mean')
       

        
        if loss != 0:
            loss.backward()
        if self.exp_dict['model'].get('clip_grad'):
            ut.clip_gradient(self.opt, 0.5)

        self.opt.step()

        return {'train_loss': float(loss)}


    def val_on_loader(self, loader, savedir_images=None, n_images=0):

        self.eval()
        val_list = []
        for i, batch in enumerate(tqdm.tqdm(loader)):
            val_list += [self.val_on_batch(batch)]
            if i < n_images:
                self.vis_on_batch(batch, savedir_image=os.path.join(savedir_images, 
                    '%d.png' % batch['meta'][0]['index']))
            
        return pd.DataFrame(val_list).mean().to_dict()
        
    def val_on_batch(self, batch):
        self.eval()

        pred_mask = self.predict_on_batch(batch)
        blobs = morph.label(pred_mask[0])
        pred_counts = blobs.max()

        points_mask = batch['points'][0] == 1

        gt_counts =  float(points_mask.sum())
        val_dict = {}
        split = batch['meta'][0]['split']
        val_dict['%s_mae' % split] = np.abs(gt_counts - pred_counts)
        val_dict['%s_game' % split] = lcfcn_loss.compute_game(pred_points=lcfcn_loss.blobs2points(blobs), 
                        gt_points=points_mask, L=3)
        val_dict['%s_score' % split] = - val_dict['%s_mae' % split]
        return val_dict