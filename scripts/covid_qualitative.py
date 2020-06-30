import sys, os
path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, path)

import torch
import torchvision
import tqdm
import pandas as pd
import pprint
import itertools
import os
import pylab as plt
import exp_configs
import time
import numpy as np
from kornia.geometry.transform import flips
from src import models
from src import datasets
from src import utils as ut


import argparse

from torch.utils.data import sampler
from torch.utils.data.sampler import RandomSampler
from torch.backends import cudnn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from src import datasets
# from src import optimizers 
import exp_configs
import torchvision

cudnn.benchmark = True

from haven import haven_utils as hu
from haven import haven_chk as hc
from haven import haven_img as hi
from haven import haven_results as hr
from haven import haven_chk as hc
# from src import looc_utils as lu
from PIL import Image

name2path = {'cityscapes':'/mnt/datasets/public/segmentation/cityscapes',
            'pascal':'/mnt/datasets/public/issam',
            'kitti':'/mnt/datasets/public/issam'}

import pprint
import pandas as pd

def create_latex_table(table, filter_dict, map_row_dict_dict, map_col_dict, **kwargs):
    # map columns
    table2 = pd.DataFrame()
    for col_old, col_new in map_col_dict.items():
        # map column
        table2[col_new] = table[col_old]
        
        # map rows
        if col_old in map_row_dict_dict:
            map_row_dict = map_row_dict_dict[col_old]
            table2[col_new] = table2[col_new].apply( lambda x: map_row_dict[x.replace("'","")] if x.replace("'","") in map_row_dict else x )
        
    # filter dict
    conds = None
    for k, v in filter_dict.items():
        if not isinstance(v, list):
            v = [v]
#         print(k, v)
        for vi in v:
            cond = table2[k] == vi
            if conds is None:
                conds = cond
            else:
                conds = conds | cond
        
        table2 = table2[conds]
        table2 = table2.set_index(k)
        table2 = table2.reindex(v)
        table2.insert(0, k, table2.index)
        table2 = table2.reset_index(drop=True)
        

    return table2.to_latex(**kwargs)



# for exp_name in ['weakly_covid19_v1', 'weakly_covid19_v2_mixed', 'weakly_covid19_v3_mixed']:
#     rm.exp_list = hr.get_exp_list_from_config([exp_name], exp_config_name)
#     table = (rm.get_score_table())
#     print(create_latex_table(table=table, 
#                              filter_dict=filter_dict, 
#                              map_row_dict_dict=map_row_dict_dict, 
#                              map_col_dict=map_col_dict,
#                              float_format='%.2f', 
#                              caption=caption_dict[exp_name], 
# #                              label=caption_dict, 
#                              index=False))

def get_stats(exp_dict):

    dataset_name = exp_dict['dataset']['name']
    n_classes = exp_dict['dataset']['n_classes']
    
    stat_list = []
        
    print('')
    print(dataset_name, '-', 'n_classes: %d' % n_classes)
    print('===========')
    
    fname = '.tmp/covid_stats/%s_c%d.csv' %  (dataset_name, n_classes)
    if not os.path.exists(fname):
        for split in ['train', 'val', 'test']:
            dataset = datasets.get_dataset(dataset_dict={'name':dataset_name},
                        datadir=None,
                                            split=split,exp_dict=exp_dict)
            loader = torch.utils.data.DataLoader(dataset, batch_size=1, 
                            num_workers=100, collate_fn=ut.collate_fn)
            
            
            for i, b in enumerate(tqdm.tqdm(loader)):
                u_list = np.unique(b['masks'])
                stat_dict = {'split':split}
                b['points'][b['points'] == 0] = 255
                for c in range(n_classes):
                    if c in u_list:
                        stat_dict['class_%d' % c] = 1
                    else:
                        stat_dict['class_%d' % c] = 0
                for c in range(n_classes):
                    if c == 0:
                        continue
                    stat_dict['n_regions_c%d' % c] = (b['points'] == c).sum().item()
                # stat_dict['n_regions_2'] = (b['points'] == 2).sum().item()
                stat_list += [stat_dict] 
        stats = pd.DataFrame(stat_list).groupby('split').sum()
        stats.to_csv(fname)
    else:
        stats = pd.read_csv(fname)

    return stats

def save_images(exp_dict):

    dataset_name = exp_dict['dataset']['name']
    n_classes = exp_dict['dataset']['n_classes']
    model = models.get_model(model_dict=exp_dict['model'],
                                    exp_dict=exp_dict,
                                    train_set=None).cuda()
    state_dict = hc.load_checkpoint(exp_dict, savedir_base, fname='model_best.pth')
    model.load_state_dict(state_dict)
    model.eval()
    np.random.seed(1)
    
    train_set = datasets.get_dataset(dataset_dict={'name':dataset_name},
                datadir=None,
                                    split="test",exp_dict=exp_dict)
    n_images = 0
    for _ in range(len(train_set)):
        i = np.random.choice(len(train_set))
        b = train_set[i]
        
        if n_images > 5:
            break

        if b['masks'].sum() == 0:
            print(i)
            continue
        n_images += 1
        batch = ut.collate_fn([b])

        image = batch['images']
        gt = np.asarray(batch['masks'], np.float32)
        gt /= (gt.max() + 1e-8)

        image = F.interpolate(image, size=gt.shape[-2:], mode='bilinear', align_corners=False)
        img_rgb = hu.f2l(hu.denormalize(image, mode='rgb')[0])
        img_rgb = (np.array(img_rgb)*255.).astype('uint8')

        # save rgb
        fname_rgb = '.tmp/covid_qualitative/%s/%s/%d_rgb.png' % (exp_group, 'gt', i)
        hu.save_image(fname_rgb,img_rgb)

        # save pts
        fname_pts = '.tmp/covid_qualitative/%s/%s/%d_pts.png' % (exp_group, 'gt', i) 
        img_gt = np.array(hu.save_image('',
                    img_rgb,
                    return_image=True))

        if 'points' in batch:
            pts = batch['points'][0].numpy()
            pts[pts == 1] = 2
            pts[pts == 0] = 1
            pts[pts == 255] = 0
            img_gt = np.array(hu.save_image('', img_gt/255.,
                                points=pts, radius=2, return_image=True))
        hu.save_image(fname_pts,img_gt)

        # save mask
        fname_mask = '.tmp/covid_qualitative/%s/%s/%d_mask.png' % (exp_group, 'gt', i) 
        

        img_mask = np.array(hu.save_image('',
                    img_rgb,
                    mask=gt[0], 
                    return_image=True))
        hu.save_image(fname_mask, img_mask)

        # pred
        fname_pred = '.tmp/covid_qualitative/%s/%s/%d_%s.png' % (exp_group, 'preds', i, exp_dict['model']['loss']) 
        res = model.predict_on_batch(batch)

        img_res = hu.save_image('',
                    img_rgb,
                    mask=res[0], return_image=True)
        hu.save_image(fname_pred, np.array(img_res))
                
if __name__ == "__main__":
    savedir_base = '/mnt/clients/covid19/borgy'
    for exp_group in [
        'weakly_covid19_v1_c2',

                      'weakly_covid19_v2_mixed_c2',
                      'weakly_covid19_v2_sep_c2',

                    #   'weakly_covid19_v2_mixed_c3',
                    #   'weakly_covid19_v2_sep_c3',

                      'weakly_covid19_v3_mixed_c2'
                      ]:
        exp_list = exp_configs.EXP_GROUPS[exp_group]
        # rm.exp_list = hr.get_exp_list_from_config([exp_name], exp_config_name)
        # latex
        map_row_dict_dict = {'model.loss': {"point_loss":'Point Loss', 
                                            "cons_point_loss":'CB Point Loss', 
                                            'joint_cross_entropy':'W-CE (Full Sup.)'}}

        map_col_dict = {'model.loss': 'Loss Function', 
                        'test_dice':'Dice', 
                        'test_iou':'IoU', 
                        'test_prec':'PPV', 
                        'test_recall':'Sens.', 
                        'test_spec':'Spec.'}

        filter_dict = {'Loss Function':['Point Loss', 'CB Point Loss','W-CE (Full Sup.)']}

        caption_dict = {'weakly_covid19_v1_c2':'COVID19-A',
                        'weakly_covid19_v2_mixed_c2':'COVID19-B-Mix',
                        'weakly_covid19_v3_mixed_c2':'COVID19-C-Mix',
                        'weakly_covid19_v2_sep_c2':'COVID19-B-Sep',
                        'weakly_covid19_v3_sep_c2':'COVID19-C-Sep'
                        }
        table = (hr.get_score_df(exp_list, savedir_base))
        print(create_latex_table(table=table, 
                                filter_dict=filter_dict, 
                                map_row_dict_dict=map_row_dict_dict, 
                                map_col_dict=map_col_dict,
                                float_format='%.2f', 
                                caption=caption_dict[exp_group], 
    #                              label=caption_dict, 
                                index=False))

        for exp_dict in exp_list:
            print(get_stats(exp_dict))
            # save_images(exp_dict)

            
        
            

            
        print('saved %s' % exp_group)