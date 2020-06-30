import json
import torch
import numpy as np
import subprocess
import json
import torch
import pylab as plt
import numpy as np
from tqdm import tqdm 
from pycocotools.coco import COCO
from torchvision import transforms
from torchvision.transforms import functional as ft
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from torchvision.transforms import functional as ft
from importlib import reload
from skimage.segmentation import mark_boundaries
from torch.utils import data
import pickle 
import pandas as pd
import datetime as dt
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
from pycocotools import mask as maskUtils
from skimage import morphology as morph
import collections
import shlex
import inspect
from bs4 import BeautifulSoup
import tqdm
from torch.utils.data.dataloader import default_collate
import time 
import pprint
from importlib import import_module
import importlib
from torch.utils.data.sampler import SubsetRandomSampler
import misc as ms 
from losses import splits
from core import response_map as rm
from losses import helpers as l_helpers


@torch.no_grad()
def valPascal(model, dataset, 
              predict_method="BestObjectness", 
              n_val=100):
        
    model.eval()
    dataset.load_cocoGt()
    print("Get predicted proposals for {}".format(predict_method))
    # path_base = "/mnt/datasets/public/issam/VOCdevkit/annotations/"

    annList = []


    if type(n_val) != list:
        ind = range(n_val)
    else:
        ind = n_val
        n_val = ind[-1]

    annList = []

    image_id_list = []
    for i in ind:
        print("{}/{}".format(i, n_val))
        batch = ms.get_batch(dataset, [i])
        if "dataset" in batch and batch["dataset"][0] == "coco2014":
            image_id = int(batch["name"][0][:-4].split("_")[-1])
        else:
            image_id = batch["name"][0]
        image_id_list += [image_id] 

        assert image_id in dataset.cocoGt.getImgIds() 
        pred_dict = model.predict(batch, 
                                  predict_method=predict_method)
        image_annList = pred_dict["annList"]

        if "dataset" in batch and batch["dataset"][0] == "coco2014":
            catIds = dataset.coco.getCatIds()
            for ann in image_annList:
                ann["image_id"] = image_id
                ann["category_id"] = catIds[ann["category_id"]-1]
        
        annList += image_annList
    
    # if n_val == len(dataset):
    #     image_id_list = None
    
    return dataset.eval_cocoPred(annList, image_id_list=image_id_list)

