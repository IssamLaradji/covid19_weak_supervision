import os, pprint, tqdm
import numpy as np
import pandas as pd
from haven import haven_utils as hu 
from haven import haven_img as hi
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_networks import infnet, fcn8_vgg16
from . import semseg
from .losses import  density_loss
from src.modules.lcfcn import lcfcn_loss
from src import utils as ut


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
        if loss_name == 'lcfcn':
            loss = lcfcn_loss.compute_lcfcn_loss(logits, 
                      torch.LongTensor(batch['points'][0]).cuda())
        if loss_name == 'density':       
            loss = density_loss.compute_density_loss(logits, 
                    torch.LongTensor(batch['points'][0]).cuda())
       

        
        if loss != 0:
            loss.backward()
        if self.exp_dict['model'].get('clip_grad'):
            ut.clip_gradient(self.opt, 0.5)

        self.opt.step()

        return {'train_loss': float(loss)}

    @torch.no_grad()
    def predict_on_batch(self, batch, mode='mask'):
        images = batch["images"].cuda()
        n = images.shape[0]
        logits = self.model_base.forward(images)
        # match image size
        # logits = match_image_size(images, logits)
        if mode == 'mask':
            return logits.argmax(dim=1)

        elif mode == 'counts_density' and self.exp_dict['model']['loss'] == 'lcfcn':
            blobs = lcfcn_loss.get_blobs(logits)
            density = lcfcn_loss.blobs2points(blobs)
            uniques = [u for u in np.unique(blobs) if u not in [0]]
            return len(uniques), density

        elif mode == 'counts_density' and self.exp_dict['model']['loss'] == 'density':
            density = logits[:, 1]
            
            return float(density.sum()), density.cpu().numpy().squeeze()


    def vis_on_batch(self, batch, savedir_image):
        image = batch['images']
        res = self.predict_on_batch(batch)

        img_gt = hu.save_image('tmp.png',
                  hu.denormalize(image, mode='rgb')[0],
                        points=batch['points'][0], radius=1, 
                        return_image=True)

        img_res = hu.save_image(savedir_image,
                     hu.denormalize(image, mode='rgb')[0],
                      mask=res.cpu().numpy(), return_image=True)

        # img_gt = semseg.text_on_image( 'Groundtruth', np.array(img_gt), color=(0,0,0))
        # img_res = semseg.text_on_image( 'Prediction', np.array(img_res), color=(0,0,0))
        hu.save_image(savedir_image, np.hstack([np.array(img_gt), np.array(img_res)]))

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
        # image = batch['images']
        # gt_mask = np.array(batch['masks'])
        # prob_mask = self.predict_on_batch(batch)

        pred_counts, pred_density = self.predict_on_batch(batch,
                           mode='counts_density')
        gt_counts =  float(batch['points'][0].sum())
        val_dict = {}
        val_dict['mae'] = np.abs(gt_counts - pred_counts)
        val_dict['game'] = lcfcn_loss.compute_game(pred_points=pred_density, 
                        gt_points=batch['points'][0], L=3)
        val_dict['val_score'] = - val_dict['mae']
        return val_dict