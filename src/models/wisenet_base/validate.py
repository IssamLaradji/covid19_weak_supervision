import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import pickle
import cv2
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import sys
import os
import os.path as osp
import datetime                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
import random
import timeit, tqdm
import pandas as pd 
from pydoc import locate
start = timeit.default_timer()
import datetime as dt
import misc as ms
import time
from addons import val
import ann_utils as au

def validate(main_dict, train_only=False):

  ms.print_welcome(main_dict)

  # EXTRACT VARIABLES
  reset =  main_dict["reset"]
  epochs =  main_dict["epochs"]
  batch_size = main_dict["batch_size"]
  sampler_name = main_dict["sampler_name"]
  verbose = main_dict["verbose"]
  loss_name = main_dict["loss_name"]
  metric_name = main_dict["metric_name"]
  epoch2val = main_dict["epoch2val"]
  val_batchsize = main_dict["val_batchsize"]
  metric_class = main_dict["metric_dict"][metric_name]
  loss_function = main_dict["loss_dict"][loss_name]
  predictList = main_dict["predictList"]

  # Assert everything is available
  ## Sharp proposals
  ## LCFCN points
  ## gt_annDict


  # Dataset  
  train_set, val_set = ms.load_trainval(main_dict)
  train_set[0]

  # Model  
  if reset == "reset" or not ms.model_exists(main_dict):
    model, opt, history = ms.init_model_and_opt(main_dict, 
                                                train_set) 
    print("TRAINING FROM SCRATCH EPOCH: %d/%d" % (history["epoch"],
                                                  epochs))
  else:
    model, opt, history = ms.load_latest_model_and_opt(main_dict, 
                                                       train_set) 
    print("RESUMING EPOCH %d/%d" % (history["epoch"], epochs)) 
  

  # Get Dataloader
  trainloader = ms.get_dataloader(dataset=train_set, 
                                  batch_size=batch_size, 
                                  sampler_class=main_dict["sampler_dict"][sampler_name])
  
  # SAVE HISTORY
  history["epoch_size"] = len(trainloader)

  if "trained_batch_names" in history:
    model.trained_batch_names = set(history["trained_batch_names"])



  # START TRAINING
  start_epoch = history["epoch"]
  predict_name = predictList[0]
  if len(history["val"]) == 0:
    last_validation = 1
  else:
    last_validation = history["val"][-1]["epoch"]

  for epoch in range(start_epoch + 1, epochs):
    import ipdb; ipdb.set_trace()  # breakpoint a3e81a5d //
    
    # %%%%%%%%%%% 2. VALIDATION PHASE %%%%%%%%%%%%"
    validation_phase_mAP(history, main_dict, model, val_set, predict_name, 
                                              epoch)
    import ipdb; ipdb.set_trace()  # breakpoint 952cb52f //






@torch.no_grad()
def validation_phase_MAE(history, main_dict, model, val_set, metric_class, verbose, metric_name, 
                         predict_name, epoch):
    val_dict = ms.validate(dataset=val_set, 
                    model=model, 
                    verbose=verbose, 
                    metric_class=metric_class, 
                    batch_size=1,
                    epoch=epoch)
    val_dict["predict_name"] = predict_name
    val_dict["epoch"] = epoch

    # Update history
    history["val"] += [val_dict]

    # Higher is better

    if (history["best_model"] == {} or 
        history["best_model"][metric_name] >= val_dict[metric_name]):

      history["best_model"] = val_dict


    return history

@torch.no_grad()
def validation_phase_mAP(history, main_dict, model, val_set, predict_name, epoch):
    val_dict, pred_annList = au.validate(model, val_set, 
                predict_method=predict_name, 
                n_val=len(val_set), return_annList=True)
  
    val_dict["predict_name"] = predict_name
    val_dict["epoch"] = epoch
    val_dict["time"] = datetime.datetime.now().strftime("%b %d, 20%y")

    # Update history
    history["val"] += [val_dict]

    # Higher is better
    if (history["best_model"] == {} or 
        history["best_model"]["0.5"] <= val_dict["0.5"]):

      history["best_model"] = val_dict
    
    return history