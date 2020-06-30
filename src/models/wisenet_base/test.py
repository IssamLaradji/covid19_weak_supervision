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
import misc as ms
import pandas as pd 
from pydoc import locate
start = timeit.default_timer()
import datetime as dt
import time
import glob
# from losses import losses
from skimage.segmentation import find_boundaries
from sklearn.metrics import confusion_matrix

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
from pycocotools import mask as maskUtils
from datasets import helpers as d_helpers
# from core import proposals as prp

def create_dataset(main_dict):
  test_set = ms.load_test(main_dict)
  path_base = "/mnt/datasets/public/issam/VOCdevkit/annotations/"
  d_helpers.pascal2cocoformat("{}/instances_val2012.json".format(path_base), test_set)
  data = ms.load_json("{}/instances_val2012.json".format(path_base))

def create_voc2007(main_dict):
  main_dict["dataset_name"] = "Pascal2007"
  test_set = ms.load_test(main_dict)
  ms.get_batch(test_set, [1])
  path_base = "/mnt/datasets/public/issam/VOCdevkit/annotations/"
  d_helpers.pascal2cocoformat("{}/instances_val2012.json".format(path_base), test_set)
  data = ms.load_json("{}/instances_val2012.json".format(path_base))


def test_list(model, cocoGt, val_set, indices, predict_proposal):
  annList = []
  for i in indices:
    batch = ms.get_batch(val_set, [i])
    annList += predict_proposal(model, batch, "annList")

  cocoEval, cocoDt = d_helpers.evaluateAnnList(annList)

    # probs = F.softmax(self(batch["images"].cuda()),dim=1).data
    # blobs = bu.get_blobs(probs)

  for i in indices:
      batch = ms.get_batch(val_set, [i])
      image_id = int(batch["name"][0])
      annList = cocoGt.imgToAnns[image_id]
      mask = d_helpers.annList2mask(annList)

      dt_mask = d_helpers.annList2mask(cocoDt.imgToAnns[image_id])
      ms.images(batch["images"], mask, denorm=1, win=str(i))
      ms.images(batch["images"], dt_mask, denorm=1, win=str(i)+"_pred")


def test_COCOmap(main_dict):
  # create_voc2007(main_dict)
  
  model = ms.load_best_model(main_dict)
  _, val_set = ms.load_trainval(main_dict)
  

  path_base = "/mnt/datasets/public/issam/VOCdevkit/annotations/"
  fname = "{}/instances_val2012.json".format(path_base)
    
  cocoGt = COCO(fname)
  # fname = "{}/instances_val2012.json".format(path_base)
  # cocoGt = COCO(fname)

  fname = (path_base + "/results/"+ main_dict["exp_name"] 
          +"_"+str(main_dict["model_options"]["predict_proposal"])+".json")
  # test_list(model, cocoGt, val_set, [0,1,2,3], prp.Blobs)
  import ipdb; ipdb.set_trace()  # breakpoint 06c353ef //
  # test_list(model, cocoGt, val_set, [0], prp.BestObjectness)
  # test_list(model, cocoGt, val_set, [0,1,2,3], prp.Blobs)
  if not os.path.exists(fname) or 1:
    annList = []
    for i in range(len(val_set)):
      batch = ms.get_batch(val_set, [i])
      try:
          annList += model.predict(batch, "annList")
      except Exception as exc:
          import ipdb; ipdb.set_trace()  # breakpoint 5f61b0cfx //
      if (i % 100) == 0:

       cocoEval, _ = d_helpers.evaluateAnnList(annList)
       ms.save_json(fname.replace(".json","inter.json"), annList)

      # ms.save_json("tmp.json", annList)
      # cocoDt = cocoGt.loadRes("tmp.json")


      # cocoEval = COCOeval(cocoGt, cocoDt, "segm")
      # cocoEval.params.imgIds  = list(set([v["image_id"] for v in cocoDt.anns.values()]))
      # cocoEval.evaluate()
      # cocoEval.accumulate()
      # cocoEval.summarize()


      print("{}/{}".format(i, len(val_set)))

    ms.save_json(fname, annList)
  # cocoEval = d_helpers.evaluateAnnList(ms.load_json(fname))

  
  
  

  # cocoEval = COCOeval(cocoGt, cocoDt, annType)
  #cocoEval.params.imgIds  = list(set([v["image_id"] for v in cocoDt.anns.values()]))
  if 1:
    #cocoEval.params.imgIds = [2007000033]
    cocoDt = cocoGt.loadRes(fname)
    cocoEval = COCOeval(cocoGt, cocoDt, "segm")
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    print("Images:", len(cocoEval.params.imgIds))
    print("Model: {}, Loss: {}, Pred: {}".format(main_dict["model_name"],
      main_dict["loss_name"],
            main_dict["model_options"]["predict_proposal"]))
  import ipdb; ipdb.set_trace()  # breakpoint c6f8f580 //
  # d_helpers.visGT(cocoGt, cocoDt,ms.get_batch(val_set, [169]))
  # d_helpers.valList(cocoGt, cocoDt, val_set, [173,174])
  # model.set_proposal(None); vis.visBlobs(model, ms.get_batch(val_set, [169]), "blobs")
  return "mAP25: {:.2f} - mAP75:{:.2f}".format(cocoEval.stats[1], cocoEval.stats[2])
    # d_helpers.evaluateAnnList(annList)
  # path_base = "/mnt/datasets/public/issam/VOCdevkit/annotations/"
  # annFile = "{}/pascal_val2012.json".format(path_base)
  # cocoGt = COCO(annFile)

  # test_set = ms.load_test(main_dict)

  # fname = "{}/instances_val2012.json".format(path_base)
  # if os.path.exists(fname):
  #   cocoGt = COCO(fname)
  # else:
  #   path_base = "/mnt/datasets/public/issam/VOCdevkit/annotations/"
  #   d_helpers.pascal2cocoformat("{}/instances_val2012.json".format(path_base), test_set)
  
  # cocoDt = cocoGt.loadRes(fname.replace(".json", "_best.json"))

   
  # cocoEval = COCOeval(cocoGt, cocoDt, "segm")
  # cocoEval.evaluate()
  # cocoEval.accumulate()
  # cocoEval.summarize()


def evaluateAnnList(annList):
    path_base = "/mnt/datasets/public/issam/VOCdevkit/annotations/"
    fname = "{}/instances_val2012.json".format(path_base)
    
    cocoGt = COCO(fname)
    ms.save_json("tmp.json", annList)
    cocoDt = cocoGt.loadRes("tmp.json")

    cocoEval = COCOeval(cocoGt, cocoDt, "segm")

    # cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.params.imgIds  = list(set([v["image_id"] for v in cocoDt.anns.values()]))

    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print("# images:", len(cocoEval.params.imgIds))

    return cocoEval

def test_run(main_dict, metric_name, save,
             reset, predict_proposal=None):
  if predict_proposal is None:
    predict_proposal = ""
  
  history = ms.load_history(main_dict)

  if history is None:
    best_epoch = 0
  else:
    best_epoch = history["best_model"]["epoch"]
    
  fname = main_dict["path_save"] + "/test_{}{}_{}.json".format(predict_proposal, metric_name, best_epoch)
  print("Testing: {} - {} - {} - {} - best epoch: {}".format(main_dict["dataset_name"],
                          main_dict["config_name"], 
                          main_dict["loss_name"],
                          metric_name,
                          best_epoch))

  if not os.path.exists(fname) or reset == "reset":   
    with torch.no_grad():
      score = ms.val_test(main_dict, metric_name=metric_name, n_workers=1)
    ms.save_json(fname, score)

  else:
    score = ms.load_json(fname)
      
  return score[metric_name]

def test_load(main_dict, metric_name, predict_proposal=None):
  if predict_proposal is None:
    predict_proposal = ""

  results = glob.glob(main_dict["path_save"] +
           "/test_{}{}_[0-9]*.json".format(predict_proposal, 
                                           metric_name))
  
  results_dict = {}
  for r in results:
    results_dict[int(os.path.basename(r).replace(".json","").split("_")[-1])] = r
  
  if len(results_dict) != 0:

    best = max(results_dict.keys())
    fname = results_dict[best]

    result = ms.load_json(fname)
    #ms.save_json(fname.replace("None", main_dict["metric_name"]), result)
    history = ms.load_history(main_dict)

    if history is None:
      return "{:.2f}".format(result[metric_name])
    best_epoch = history["best_model"]["epoch"]

    if best_epoch == best:            
      return "{:.2f} - ({})".format(result[metric_name], best, predict_proposal)
    else:
      return "{:.2f}* - ({})".format(result[metric_name], best, predict_proposal)
  else:
    return "empty"


      

