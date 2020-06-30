import os, pprint, tqdm
import numpy as np
import pandas as pd
from haven import haven_utils as hu 
from haven import haven_img as hi
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import semseg
from src import utils as ut
import cv2
from src import models
from kornia.geometry.transform import flips
from src.modules.lcfcn import lcfcn_loss


def get_semsegactive(base_class):
    class SemSegActive(base_class):
        def __init__(self, exp_dict, train_set):
            super().__init__(exp_dict)

            self.train_set = train_set
            self.rng = np.random.RandomState(1)

            self.active_learning = exp_dict['active_learning']

            self.heuristic = self.active_learning['heuristic']
            self.init_sample_size = self.active_learning['init_sample_size']
            self.sample_size = self.active_learning['sample_size']
            self.n_regions = self.active_learning['n_regions']
            self.label_map = np.zeros((len(train_set), self.n_regions))
            self.cost_map = np.zeros((len(train_set), self.n_regions))
    
        def get_state_dict(self):
            state_dict = {"model": self.model_base.state_dict(),
                            "opt": self.opt.state_dict(),
                            'label_map':self.label_map,
                            'cost_map':self.cost_map,
                            'epoch':self.epoch}

            return state_dict

        def load_state_dict(self, state_dict):
            self.model_base.load_state_dict(state_dict["model"])
            if 'opt' not in state_dict:
                return
            self.opt.load_state_dict(state_dict["opt"])
            self.epoch = state_dict['epoch']
            self.label_map = state_dict['label_map']
            self.cost_map = state_dict['cost_map']

        def train_on_batch(self, batch):
            # add to seen images
            for m in batch['meta']:
                self.train_hashes.add(m['hash'])

            self.opt.zero_grad()

            images = batch["images"].cuda()
            logits = self.model_base(images)

            index = batch['meta'][0]['index']
            n,c,h,w = images.shape
            bbox_yxyx = get_rect_bbox(h, w, n_regions=self.n_regions)[self.label_map[index] == 1]

            assert(len(bbox_yxyx) > 0)
            loss_name = self.exp_dict['model']['loss']

            if loss_name == 'joint_cross_entropy':
                if len(bbox_yxyx) == self.n_regions:
                    masks = batch['masks'].cuda()
                    roi_mask = torch.ones((h,w), dtype=bool)
                    loss = ut.joint_loss_flat(logits, masks.float(), roi_mask)
                else:
                    masks = batch['masks'].cuda()
                    roi_mask = torch.zeros((h,w), dtype=bool)
                    for y1, x1, y2, x2 in bbox_yxyx:
                        roi_mask[y1:y2, x1:x2] = 1
                    loss = ut.joint_loss_flat(logits, masks.float(), roi_mask)

            elif loss_name == 'cross_entropy':
                if len(bbox_yxyx) == self.n_regions:
                    masks = batch['masks'].cuda()
                    roi_mask = torch.ones((h,w), dtype=bool)
                    L = F.binary_cross_entropy_with_logits(logits, masks.float(), reduction='none').squeeze()
                    loss = L.mean()
                else:
                    masks = batch['masks'].cuda()
                    roi_mask = torch.zeros((h,w), dtype=bool)
                    for y1, x1, y2, x2 in bbox_yxyx:
                        roi_mask[y1:y2, x1:x2] = 1
                    
                    L = F.binary_cross_entropy_with_logits(logits, masks.float(), reduction='none').squeeze()
                    L = L[roi_mask]
                    loss = L.mean()

            elif loss_name in ['image_level', 'image_level3']:
                masks = batch['masks'].cuda().squeeze()
                logits = logits.squeeze()

                loss_fg = 0.
                n_fg = 0
                loss_bg = 0.
                n_bg = 0
                loss = 0.
                assert(logits.ndim==2)
                assert(masks.ndim==2)
                for y1, x1, y2, x2 in bbox_yxyx:
                    u_list = masks[y1:y2, x1:x2].unique()
                    l_box = logits[y1:y2, x1:x2]
                    # lcfcn_loss.get_random_points(masks[y1:y2, x1:x2], n_points=1)
                    if 1 in u_list:
                        n_fg += 1
                        # foreground
                        loss_fg += F.binary_cross_entropy_with_logits(l_box.max()[None], 
                                                torch.ones(1, device=l_box.device), 
                                                reduction='mean')
                        if 0 in u_list:
                            loss_fg += F.binary_cross_entropy_with_logits(l_box.min()[None], 
                                                    torch.zeros(1, device=l_box.device), 
                                                    reduction='mean')
                    elif 0 in u_list:
                        # only background
                        n_bg += 1
                        loss_bg += F.binary_cross_entropy_with_logits(l_box.max()[None], 
                                                torch.zeros(1, device=l_box.device), 
                                                reduction='mean')
                if loss_name in 'image_level3':
                    logits_flip = self.model_base(flips.Hflip()(images)).squeeze()
                    loss += torch.mean(torch.abs(flips.Hflip()(logits_flip)-logits))

                loss += loss_fg / max(n_fg, 1) + loss_bg / max(n_bg, 1)

            elif loss_name in ['const_point_level', 'point_level', 'point_level_2']:
                masks = batch['masks'].cuda().squeeze()
                logits = logits.squeeze()
                assert(logits.ndim==2)
                assert(masks.ndim==2)
                points = torch.ones(masks.shape)*255
                for i, (y1, x1, y2, x2) in enumerate(bbox_yxyx):
                    mask_box = masks[y1:y2, x1:x2]
                    u_list = mask_box.unique()
                    if 1 in u_list:
                        pts = lcfcn_loss.get_random_points(mask_box.cpu().numpy()==1, n_points=1, seed=y1+ x1+ y2+ x2)
                        yi, xi = np.where(pts)
                        points[y1:y2, x1:x2][yi, xi] = 1
                        if loss_name == 'point_level_2':
                            if 0 in u_list:
                                pts = lcfcn_loss.get_random_points(mask_box.cpu().numpy()==0, n_points=1, seed=y1+ x1+ y2+ x2)
                                yi, xi = np.where(pts)
                                points[y1:y2, x1:x2][yi, xi] = 0
                    elif 0 in u_list:
                        pts = lcfcn_loss.get_random_points(mask_box.cpu().numpy()==0, n_points=1, seed=y1+ x1+ y2+ x2)
                        yi, xi = np.where(pts)
                        points[y1:y2, x1:x2][yi, xi] = 0
                loss = 0. 

                # vis
                # if 1:
                #     original = hu.denormalize(batch['images'], mode='rgb')[0]
                #     pts = points.clone()
                #     pts[pts == 1] = 2
                #     pts[pts == 0] = 1
                #     pts[pts == 255] = 0
                #     img_bbox = bbox_yxyx_on_image(bbox_yxyx, original)
                #     hu.save_image('tmp_mask.png', img_bbox,
                #                         mask=masks.cpu().numpy(), return_image=False)
                #     hu.save_image('tmp.png', img_bbox,
                #                         points=pts.long().numpy(), radius=2, return_image=False)

                # foreground loss
                ind = ((points != 255) & (points != 0))
                if ind.sum() > 0:
                    loss += F.binary_cross_entropy_with_logits(logits[ind], 
                                            points[ind].float().cuda(), 
                                            reduction='mean')
                # background loss
                ind = ((points != 255) & (points != 1))
                if ind.sum() > 0:
                    loss += F.binary_cross_entropy_with_logits(logits[ind], 
                                        points[ind].float().cuda(), 
                                        reduction='mean')
                                        
                if loss_name == 'const_point_level':
                    logits_flip = self.model_base(flips.Hflip()(images)).squeeze()
                    loss += torch.mean(torch.abs(flips.Hflip()(logits_flip)-logits))

            elif loss_name in ['image_level2']:
                masks = batch['masks'].cuda().squeeze()
                logits = logits.squeeze()

                loss_fg = 0.
                n_fg = 0
                loss_bg = 0.
                n_bg = 0
                assert(logits.ndim==2)
                assert(masks.ndim==2)
                for y1, x1, y2, x2 in bbox_yxyx:
                    u_list = masks[y1:y2, x1:x2].unique()
                    l_box = logits[y1:y2, x1:x2]
                    # lcfcn_loss.get_random_points(masks[y1:y2, x1:x2], n_points=1)
                    if 1 in u_list:
                        n_fg += 1
                        # foreground
                        loss_fg += F.binary_cross_entropy_with_logits(l_box.max()[None], 
                                                torch.ones(1, device=l_box.device), 
                                                reduction='mean')
                    elif 0 in u_list:
                        # only background
                        n_bg += 1
                        loss_bg += F.binary_cross_entropy_with_logits(l_box.max()[None], 
                                                torch.zeros(1, device=l_box.device), 
                                                reduction='mean')
                loss = loss_fg / max(n_fg, 1) + loss_bg / max(n_bg, 1)
            if loss != 0:
                loss.backward()
                if self.exp_dict['model'].get('clip_grad'):
                    ut.clip_gradient(self.opt, 0.5)
                self.opt.step()

            return {'train_loss': float(loss)}
        
        def vis_on_batch(self, batch, savedir_image):
            image = batch['images']
            index = batch['meta'][0]['index']
            gt = np.asarray(batch['masks'], np.float32)
            gt /= (gt.max() + 1e-8)
            res = self.predict_on_batch(batch)

            image = F.interpolate(image, size=gt.shape[-2:], mode='bilinear', align_corners=False)
            original = hu.denormalize(image, mode='rgb')[0]
            img_res = hu.save_image(savedir_image,
                        original,
                        mask=res[0], return_image=True)

            img_gt = hu.save_image(savedir_image,
                        original,
                        mask=gt[0], return_image=True)
            img_gt = models.text_on_image( 'Groundtruth', np.array(img_gt), color=(0,0,0))
            img_res = models.text_on_image( 'Prediction', np.array(img_res), color=(0,0,0))
            
            if 'points' in batch:
                pts = batch['points'][0].numpy().copy()
                pts[pts == 1] = 2
                pts[pts == 0] = 1
                pts[pts == 255] = 0
                img_gt = np.array(hu.save_image(savedir_image, img_gt/255.,
                                    points=pts, radius=2, return_image=True))

            # score map
            if self.heuristic != 'random':
                score_map = self.compute_uncertainty(batch['images'], replicate=True,  
                                            scale_factor=1, method=self.heuristic)
                score_map = F.interpolate(score_map[None], size=gt.shape[-2:], mode='bilinear', align_corners=False).squeeze()
                h, w = score_map.shape
                bbox_yxyx = get_rect_bbox(h, w, n_regions=self.n_regions)

                heatmap =  hu.f2l(hi.gray2cmap((score_map/score_map.max()).cpu().numpy().squeeze()))
    
                s_list = np.zeros(len(bbox_yxyx))
                for i, (y1, x1, y2, x2) in enumerate(bbox_yxyx):
                    s_list[i] = score_map[y1:y2, x1:x2].mean()

                
                img_bbox = bbox_yxyx_on_image(bbox_yxyx[[s_list.argmax()]], original)
                img_score_map = img_bbox*0.5 + heatmap*0.5

                img_list = [np.array(img_gt), np.array(img_res), (img_score_map*255).astype('uint8')]
            else:
                img_list = [np.array(img_gt), np.array(img_res)]
            hu.save_image(savedir_image, np.hstack(img_list))
            
        def get_pool_ind(self):
            return np.where(self.label_map.mean(axis=1) != 1)[0]

        def get_train_ind(self):
            return np.where(self.label_map.mean(axis=1) > 0)[0]

        def label_next_batch(self):
            pool_ind = self.get_pool_ind()

            if self.label_map.sum() == 0:
                with hu.random_seed(1):
                    # find k with infected regions
                    n_batches = 0
                    ind_list = []
                    for ind in np.random.permutation(len(self.train_set)):
                        if n_batches == self.init_sample_size:
                            break

                        batch = ut.collate_fn([self.train_set[ind]])

                        if batch['masks'].sum() == 0:
                            continue 

                        n_batches += 1
                        ind_list += [{'bbox_yxyx':np.arange(self.n_regions),
                                      'index':ind}] 
            else:
                if self.heuristic == 'random':
                    with hu.random_seed(1):
                        img_ind_list = self.rng.choice(pool_ind, 
                                        min(self.sample_size, len(pool_ind)), replace=False)
                        ind_list = []
                        for ind in img_ind_list:
                            bbox_ind = np.random.choice(np.where(self.label_map[ind] == 0)[0])
                            ind_list += [{'bbox_yxyx':[bbox_ind], 'index':ind}] 
                else:
                    ind_list = []
                    print('%s Scoring' % self.heuristic)
                    arange = np.arange(self.n_regions)
                    for ind in tqdm.tqdm(pool_ind):
                        batch = ut.collate_fn([self.train_set[ind]])
                        score_map = self.compute_uncertainty(batch['images'], replicate=True,  
                                        scale_factor=1, method=self.heuristic).squeeze()
                        h, w = score_map.shape
                        bbox_yxyx = get_rect_bbox(h, w, n_regions=self.n_regions)

                        unlabeled = self.label_map[ind] == 0

                        bbox_yxyx = bbox_yxyx[unlabeled]
                        bbox_ind = arange[unlabeled]

                        assert len(bbox_yxyx) > 0 
                        
                        s_list = np.zeros(len(bbox_yxyx))
                        for i, (y1, x1, y2, x2) in enumerate(bbox_yxyx):
                            s_list[i] = score_map[y1:y2, x1:x2].mean()

                        # if 1:
                        #     heatmap =  hu.f2l(hi.gray2cmap((score_map/score_map.max()).cpu().numpy().squeeze()))
                        #     original = hu.denormalize(batch['images'], mode='rgb')
                            
                        #     img_bbox = bbox_yxyx_on_image(bbox_yxyx[[s_list.argmax()]], original)
                        #     hu.save_image('.tmp/tmp.png' , img_bbox*0.5 + heatmap*0.5)
    
                        ind_list += [{'score':float(score_map.mean()), 
                                      'bbox_yxyx':[bbox_ind[s_list.argmax()]],
                                      'index':ind}] 
                    
                    # sort ind_list and pick top k
                    ind_list = sorted(ind_list, key=lambda x:-x['score'])
                    ind_list = ind_list[:self.sample_size]
                
            # update labeled indices
            for ind_dict in ind_list:
                assert self.label_map[ind_dict['index'], ind_dict['bbox_yxyx']].sum() == 0
                self.label_map[ind_dict['index'], ind_dict['bbox_yxyx']] = 1
                assert self.label_map[ind_dict['index'], ind_dict['bbox_yxyx']].mean() == 1

                if self.exp_dict['model']['loss'] in ['const_point_level', 'point_level']:
                    self.cost_map[ind_dict['index'], ind_dict['bbox_yxyx']] = 3
                
                elif self.exp_dict['model']['loss'] in ['point_level_2']:
                    self.cost_map[ind_dict['index'], ind_dict['bbox_yxyx']] = 4

                elif self.exp_dict['model']['loss'] in ['joint_cross_entropy', 'cross_entropy']:
                    batch = self.train_set[ind_dict['index']]
                    mask = batch['masks'].squeeze()
                    h, w = mask.shape
                    bbox_yxyx = get_rect_bbox(h, w, n_regions=self.n_regions)
                    for i in ind_dict['bbox_yxyx']:
                        (y1, x1, y2, x2) = bbox_yxyx[i]
                        
                        # in enumerate(bbox_yxyx):
                        patch = mask[y1:y2, x1:x2]
                        u_list = patch.unique()
                        if 1 in u_list:
                            from scipy.spatial import ConvexHull, convex_hull_plot_2d
                            pts = np.stack(np.where(patch)).transpose()
                            if len(pts) <= 2:
                                cost = len(pts)*2
                            else:
                                hull = ConvexHull(pts, qhull_options='QJ')
                                cost = len(hull.simplices)*3
                            self.cost_map[ind_dict['index'], i] = cost
                        elif 0 in u_list:
                            self.cost_map[ind_dict['index'], i] = 4

                else:
                    raise ValueError

            # return active dataset
            train_list = sorted(self.get_train_ind())
            return torch.utils.data.Subset(self.train_set, train_list)

            
    return SemSegActive
    

def xlogy(x, y=None):
    z = torch.zeros(())
    if y is None:
        y = x
    assert y.min() >= 0
    return x * torch.where(x == 0., z.cuda(), torch.log(y))

def set_dropout_train(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout) or isinstance(module, torch.nn.Dropout2d):
            module.train()

def get_rect_bbox(H, W, n_regions):
    n_regions_dim = int(np.sqrt(n_regions))
    assert(n_regions_dim == np.sqrt(n_regions))
    bbox = np.zeros((n_regions, 4)).astype(int)
    ind_map = np.ones((n_regions_dim, n_regions_dim))

    w = int(W/n_regions_dim)
    h = int(H/n_regions_dim)
    for i, (r, c) in enumerate(zip(*np.where(ind_map))):
        x1 = c * w
        x2 = x1 + w

        y1 = r * h
        y2 = y1 + h

        bbox[i] = (y1, x1, y2, x2)

        assert(x2 <= W and y2 <= H)

    return bbox

def get_negation(ind_list, n_samples):
    return np.setdiff1d(np.arange(len(n_samples)), ind_list)

def mask_on_image(mask, image):
    from skimage.color import color_dict, colorlabel
    from skimage.segmentation import mark_boundaries
    default_colors = ['red', 'blue', 'yellow', 'magenta', 'green',
                     'indigo', 'darkorange', 'cyan', 'pink', 'yellowgreen']
    image = hi.image_as_uint8(image) / 255.
    labels = [l for l in np.unique(mask) if l < len(color_dict)]
    colors =  (default_colors + list(color_dict.keys())[len(default_colors):])
    colors =  np.array(colors)[labels]
    
    image_label_overlay = label2rgb(mask, image=hu.f2l(image).squeeze().clip(0,1), 
                                    colors=colors, bg_label=0, bg_color=None, kind='overlay')
    return mark_boundaries(image_label_overlay, mask)


def bbox_yxyx_on_image(bbox_yxyx, original , color=(255, 0, 0)):
    image_uint8 = hi.image_as_uint8(original)
    H, W, _ = image_uint8.shape

    for bb in bbox_yxyx:
        y1, x1, y2, x2 = bb

        if x2 < 1:
            start_point = (int(x1*W), int(y1*H), ) 
            end_point = ( int(x2*W), int(y2*H),)
        else:
            start_point = (int(x1), int(y1))
            end_point = (int(x2), int(y2))
        
        # Line thickness of 2 px 
        thickness = 2
        assert(x2) <= W
        assert(y2) <= H 
        # Draw a rectangle with blue line borders of thickness of 2 px 
 
        image_uint8 = cv2.rectangle(image_uint8.copy(), start_point, end_point, color, thickness) 
     
    return image_uint8 / 255.


def gather_points(bbox_yxyx):
    pass