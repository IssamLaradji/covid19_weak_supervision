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

def main(main_dict, train_only=False):

  ms.print_welcome(main_dict)

  # EXTRACT VARIABLES
  reset =  main_dict["reset"]
  epochs =  main_dict["epochs"] = 100
  batch_size = main_dict["batch_size"]
  sampler_name = main_dict["sampler_name"]
  verbose = main_dict["verbose"]
  loss_name = main_dict["loss_name"]
  metric_name = main_dict["metric_name"]
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

  ms.save_pkl(main_dict["path_history"], history)

  # START TRAINING
  start_epoch = history["epoch"]
  predict_name = predictList[0]

  for epoch in range(start_epoch + 1, epochs):
    # %%%%%%%%%%% 1. Training PHASE %%%%%%%%%%%%"    
    history = training_phase(history, main_dict, model, trainloader, 
                             opt, 
                             loss_function, verbose, epoch)

    # %%%%%%%%%%% 2. VALIDATION PHASE %%%%%%%%%%%%"
    if (epoch % 5) == 0:
      history = validation_phase_mAP(history, main_dict, model, 
                                     val_set, predict_name, epoch)

    ms.save_pkl(main_dict["path_history"], history)


############################### Helpers
@torch.enable_grad()
def training_phase(history, main_dict, model, trainloader, opt, loss_function, verbose, epoch):
  # %%%%%%%%%%% 1. TRAIN PHASE %%%%%%%%%%%%"    
  train_dict = ms.fit(model, trainloader, opt, 
                      loss_function=loss_function,
                      verbose=verbose, 
                      epoch=epoch)

  # Update history
  history["epoch"] = epoch 
  history["trained_batch_names"] = list(model.trained_batch_names)
  history["train"] += [train_dict]

  # Save model, opt and history
  ms.save_latest_model_and_opt(main_dict, model, opt, history)

  return history



@torch.no_grad()
def validation_phase_mAP(history, main_dict, model, val_set, 
                         predict_name, 
                         epoch):
    val_dict, pred_annList = au.validate(model, val_set, 
                predict_method=predict_name, 
                n_val=len(val_set), return_annList=True)
  
    val_dict["predict_name"] = predict_name
    val_dict["epoch"] = epoch
    val_dict["time"] = datetime.datetime.now().strftime("%b %d, 20%y")

    # Update history
    history["val"] += [val_dict]

    path = main_dict["path_train_model"].replace(".pth", "_{}.pth".format(epoch))
    ms.save_model(path, model)
    # ms.copy_code_best(main_dict)
    # # Higher is better
    # if (history["best_model"] == {} or 
    #     history["best_model"]["0.5"] <= val_dict["0.5"]):

    #   history["best_model"] = val_dict
    #   ms.save_best_model(main_dict, model)
      
    #   ms.save_pkl(main_dict["path_best_annList"], pred_annList)
    #   ms.copy_code_best(main_dict)

    return history