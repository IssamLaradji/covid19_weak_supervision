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


class SemSegRegions(semseg.SemSeg):
    def __init__(self, exp_dict, train_set):
        super().__init__(exp_dict)

        self.train_set = train_set
        self.rng = np.random.RandomState(1)

        self.active_learning = exp_dict['active_learning']

        self.heuristic = self.active_learning['heuristic']
        self.init_sample_size = self.active_learning['init_sample_size']
        self.sample_size = self.active_learning['sample_size']

        self.labeled_indices = set()
        self.unlabeled_indices = set(np.arange(len(train_set)))

    def label_next_batch(self):
        uind_list = list(self.unlabeled_indices)
        if len(self.labeled_indices) == 0:
            ind_list = self.rng.choice(uind_list, 
                                min(self.init_sample_size, 
                                    len(uind_list)), 
                                replace=False)
        else:
            if self.heuristic == 'random':
                ind_list = self.rng.choice(uind_list, 
                                min(self.sample_size, 
                                    len(uind_list)), 
                                replace=False)

            elif self.heuristic == 'entropy':
                ind_list = []
                print('%s Scoring' % self.heuristic)
                for ind in tqdm.tqdm(self.unlabeled_indices):
                    batch = ut.collate_fn([self.train_set[ind]])
                    probs_mcmc = self.mcmc_on_batch(batch['images'], replicate=True,  
                                    scale_factor=1)
                    entropy = - xlogy(probs_mcmc).mean(dim=0).sum(dim=1)
                    score_map = entropy

                    ind_list += [{'score':float(score_map.mean()), 'index':ind}] 
                
                # sort ind_list and pick top k
                ind_list = [idict['index'] for idict in sorted(ind_list, key=lambda x:-x['score'])]
                ind_list = ind_list[:self.sample_size]
            else:
                raise ValueError('%s heuristic not available' % self.heuristic)
        
        # update labeled indices
        for ind in ind_list:
            assert ind not in self.labeled_indices, 'index already exists'
            self.labeled_indices.add(ind)

        # update unlabeled indices
        for ind in ind_list:
            self.unlabeled_indices.remove(ind)

        # return active dataset
        return DatasetWrapper(self.train_set, self.labeled_indices)

    @torch.no_grad()
    def mcmc_on_batch(self, images, replicate=False, scale_factor=None, n_mcmc=20):
        self.eval()
        set_dropout_train(self)

        # put images to cuda
        images = images.cuda()
        _, _, H, W= images.shape

        if scale_factor is not None:
            images = F.interpolate(images, scale_factor=scale_factor)
        # variables
        input_shape = images.size()
        batch_size = input_shape[0]

        if replicate and False:
            # forward on n_mcmc batch      
            images_stacked = torch.stack([images] * n_mcmc)
            images_stacked = images_stacked.view(batch_size * n_mcmc, *input_shape[1:])
            logits = self.model_base(images_stacked)
            

        else:
            # for loop over n_mcmc
            logits = torch.stack([self.model_base(images) for _ in range(n_mcmc)])
            
            logits = logits.view(batch_size * n_mcmc, *logits.size()[2:])

        logits = logits.view([n_mcmc, batch_size, *logits.size()[1:]])
        _, _, n_classes, _, _ = logits.shape
        # binary do sigmoid 
        if n_classes == 1:
            probs = logits.sigmoid()
        else:
            probs = F.softmax(logits, dim=2)

        if scale_factor is not None:
            probs = F.interpolate(probs, size=(probs.shape[2], H, W))

        self.eval()
        return probs 

class DatasetWrapper:
    def __init__(self, train_set, ind_list):
        ind_list = list(ind_list)
        ind_list.sort()
        
        self.train_set = train_set
        self.ind_list = ind_list

    def __getitem__(self, index):
        return self.train_set[self.ind_list[index]]
    
    def __len__(self):
        return len(self.ind_list)

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