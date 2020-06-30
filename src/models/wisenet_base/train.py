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

  ms.save_pkl(main_dict["path_history"], history)

  # START TRAINING
  start_epoch = history["epoch"]
  predict_name = predictList[0]
  if len(history["val"]) == 0:
    last_validation = 1
  else:
    last_validation = history["val"][-1]["epoch"]

  if last_validation < start_epoch:
    state = "val"
  else:
    state = "train"

  for epoch in range(start_epoch + 1, epochs):
    # %%%%%%%%%%% 1. Training PHASE %%%%%%%%%%%%"
    if state == "train":
      history = training_phase(history, main_dict, model, trainloader, opt, 
                             loss_function, verbose, epoch)
      if epoch%2 == 0:
        state = "val"

    # %%%%%%%%%%% 2. VALIDATION PHASE %%%%%%%%%%%%"
    if state == "val":
      if metric_name == "mIoU":
        history = validation_phase_mIoU(history, main_dict, model, val_set, predict_name, epoch)
      else:
        history = validation_phase_mAP(history, main_dict, model, val_set, predict_name, 
                                     epoch)
      state = "train"

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
      ms.save_best_model(main_dict, model)

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
      ms.save_best_model(main_dict, model)
      
      ms.save_pkl(main_dict["path_best_annList"], pred_annList)
      ms.copy_code_best(main_dict)

    return history



@torch.no_grad()
def validation_phase_mIoU(history, main_dict, model, val_set, 
                          predict_name, epoch):
    category2name = lambda x: "person" if x == 1 else "car"
    iou_mean = {"person":0, "car":0}
    iou_sum = {"person":0, "car":0}
    n_objects = {"person":0, "car":0}
    n_images = len(val_set)
    for i in range(n_images):
        
        batch = ms.get_batch(val_set, [i])
        maskObjects = batch["maskObjects"].squeeze()
        pred_dict = model.predict(batch, predict_method="BestDice")
    
        for ann in pred_dict["annList"]:
            point = ann["point"]
            category_id = point["category_id"]

            label = maskObjects[point["y"], point["x"]].item()
            assert label != 0
            A = maskObjects == label
            B = au.ann2mask(ann)["mask"]

            iou_sum[category2name(category_id)] += au.compute_iou(ms.t2n(A), ms.t2n(B))
            n_objects[category2name(category_id)] += 1

            # ms.images(batch["images"], A, denorm=1, win="GT")
            # ms.images(batch["images"], B, denorm=1,  win="Pred")

        for name in ["person", "car"]:
          iou_mean[name] = iou_sum[name] / max(1, n_objects[name])

        print("{}/{} - {:.3f} {:.3f}".format(i, n_images, iou_mean["person"], 
                  iou_mean["car"]))


    # iou_mean = iou_sum / n_objects
    val_dict = {}
    val_dict["iou_mean"] = iou_mean 
    val_dict["predict_name"] = predict_name
    val_dict["epoch"] = epoch
    val_dict["time"] = datetime.datetime.now().strftime("%b %d, 20%y")

    # Update history
    history["val"] += [val_dict]

    # Higher is better
    if (history["best_model"] == {} or 
        history["best_model"]["iou_mean"]["person"] <= val_dict["iou_mean"]["person"]):

      history["best_model"] = val_dict
      ms.save_best_model(main_dict, model)
      

    return history