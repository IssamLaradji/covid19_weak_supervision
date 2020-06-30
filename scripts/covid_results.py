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

if __name__ == "__main__":
    savedir_base = '/mnt/projects/vision_prototypes/pau/covid_v5/'
    for exp_group in [
                        # 'weakly_covid19_v1_c2',

                    #   'weakly_covid19_v2_mixed_c2',
                    #   'weakly_covid19_v2_sep_c2',

                    #   'weakly_covid19_v2_mixed_c3',
                    #   'weakly_covid19_v2_sep_c3',

                      'weakly_covid19_v3_mixed_c2'
                      ]:
        exp_list = exp_configs.EXP_GROUPS[exp_group]
        for exp_dict in exp_list:
            # if exp_dict['model']['loss'] != 'lcfcn_loss':
            #     continue

            dataset_name = exp_dict['dataset']['name']
            n_classes = exp_dict['dataset']['n_classes']
            model = models.get_model(model_dict=exp_dict['model'],
                                            exp_dict=exp_dict,
                                            train_set=None).cuda()
            state_dict = hc.load_checkpoint(exp_dict, savedir_base)
            model.load_state_dict(state_dict)
            train_set = datasets.get_dataset(dataset_dict={'name':dataset_name},
                datadir=None, split="test",exp_dict=exp_dict)
            test_loader = DataLoader(train_set,
                            # sampler=val_sampler,
                            batch_size=1,
                            collate_fn=ut.collate_fn,
                            num_workers=0)
            print(exp_dict['dataset'], '-', exp_dict['model']['loss'])
            score_dict = model.val_on_loader(test_loader)
            
            pprint.pprint(score_dict)
            print()
            np.random.seed(1)

            # for i in range(len(train_set)):
            #     b = train_set[40]
            #     if b['masks'].sum() == 0:
            #         print(i)
            #         continue
            #     batch = ut.collate_fn([b])
            #     model.vis_on_batch(batch, savedir_image='.tmp/tmp.png')
            #     for j in range(50):
            #         loss = model.train_on_batch(batch)
            #         print(j, loss['train_loss'])
            #     model.val_on_batch(batch)
            #     break
            


