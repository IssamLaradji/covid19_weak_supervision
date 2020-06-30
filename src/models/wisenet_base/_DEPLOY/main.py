import matplotlib
matplotlib.use('Agg')
from addons.pycocotools.coco import COCO
import torch
import pandas as pd
import argparse
import numpy as np
from itertools import product
import experiments
from glob import glob
# from core import dataset2cocoformat
import misc as ms
import borgy
import test 
from collections import defaultdict
import train
import debug
import configs
import train_lcfcn
# from core import data_utils
import ann_utils as au
# 
from addons import vis


def main():
  parser = argparse.ArgumentParser()

  parser.add_argument('-e','--exp') 
  parser.add_argument('-b','--borgy', default=0, type=int)
  parser.add_argument('-br','--borgy_running', default=0, type=int)
  parser.add_argument('-m','--mode', default="summary")
  parser.add_argument('-r','--reset', default="None")
  parser.add_argument('-s','--status', type=int, default=0)
  parser.add_argument('-k','--kill', type=int, default=0)
  parser.add_argument('-g','--gpu', type=int)
  parser.add_argument('-c','--configList', nargs="+",
                      default=None)
  parser.add_argument('-l','--lossList', nargs="+",
                      default=None)
  parser.add_argument('-d','--datasetList', nargs="+",
                      default=None)
  parser.add_argument('-metric','--metricList', nargs="+",
                      default=None)
  parser.add_argument('-model','--modelList', nargs="+",
                      default=None)
  parser.add_argument('-p','--predictList', nargs="+",
                      default=None)

  args = parser.parse_args()

  if args.borgy or args.kill:
    global_prompt = input("Do all? \n(y/n)\n") 

  import os
  if args.borgy_running == 0:
    os.environ["CUDA_VISIBLE_DEVICES"]="6"

  ms.set_gpu(args.gpu)


  # SET SEED
  np.random.seed(1)
  torch.manual_seed(1) 
  torch.cuda.manual_seed_all(1)

  # SEE IF CUDA IS AVAILABLE
  assert torch.cuda.is_available()
  print("CUDA: %s" % torch.version.cuda)
  print("Pytroch: %s" % torch.__version__)


  mode = args.mode 
  exp_name = args.exp


  
  key_set = set()

  if mode == "train":
    train.main(main_dict)

      

if __name__ == "__main__":
    main()