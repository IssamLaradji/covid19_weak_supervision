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
import torchvision

cudnn.benchmark = True

from haven import haven_utils as hu
from haven import haven_img as hi
from haven import haven_results as hr
from haven import haven_chk as hc
# from src import looc_utils as lu
from PIL import Image

name2path = {'cityscapes':'/mnt/datasets/public/segmentation/cityscapes',
            'pascal':'/mnt/datasets/public/issam',
            'kitti':'/mnt/datasets/public/issam'}

if __name__ == "__main__":
    result_list = []
    for exp_group in ['weakly_covid19_v1_c2',

                      'weakly_covid19_v2_mixed_c2',
                    #   'weakly_covid19_v2_sep_c2',

                    #   'weakly_covid19_v2_mixed_c3',
                    #   'weakly_covid19_v2_sep_c3',

                      'weakly_covid19_v3_mixed_c2']:
        exp_dict = exp_configs.EXP_GROUPS[exp_group][0]
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
        # pd.DataFrame(stat_list)
        
        map_dict = {'weakly_covid19_v1_c2':'COVID-19-A',
                    'weakly_covid19_v2_mixed_c2':'COVID-19-B',
                    'weakly_covid19_v3_mixed_c2':'COVID-19-C'}

        result_list += [{'Name':map_dict[exp_group],
                        #  'Resolution':'352 x 352',
                         '# Slices ':stats.sum()['class_0'], 
                         '# Slices with Infections':stats.sum()['class_1'],
                         '# Infected Regions':stats.sum()['n_regions_c1']}]
        
    print(pd.DataFrame(result_list).to_latex(index=False))
            # if b['masks'].sum() == 0:
            #     print(i)
            #     continue
            # break
        # batch = ut.collate_fn([b])
        
        # image = batch['images']
        # gt = np.asarray(batch['masks'], np.float32)
        # gt /= (gt.max() + 1e-8)

        # image = F.interpolate(image, size=gt.shape[-2:], mode='bilinear', align_corners=False)
        # # img_res = hu.save_image('',
        # #              hu.denormalize(image, mode='rgb')[0],
        # #               mask=res[0], return_image=True)

        # img_gt = hu.save_image('',
        #              hu.denormalize(image, mode='rgb')[0],
        #               mask=gt[0], return_image=True)
        # img_gt = models.text_on_image( 'Groundtruth', np.array(img_gt), color=(0,0,0))
        # # img_res = models.text_on_image( 'Prediction', np.array(img_res), color=(0,0,0))
        
        # if 'points' in batch:
        #     img_gt = np.array(hu.save_image('', img_gt/255.,
        #                         points=batch['points'][0].numpy()!=255, radius=2, return_image=True))
        # img_list = [np.array(img_gt)]
        # hu.save_image('.tmp/%s.png' % dataset_name, np.hstack(img_list))
        # print(batch.keys())
        # overlayed = hi.mask_on_image(batch['images'], batch['inst_pil'])
        # overlayed = Image.fromarray((overlayed*255).astype('uint8'))
        # overlayed.save('scripts/.tmp/dataset_%s_%s.jpg' % (dataset_name, supervision))
