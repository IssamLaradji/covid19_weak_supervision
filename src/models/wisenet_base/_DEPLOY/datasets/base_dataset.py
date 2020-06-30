import os

import numpy as np
import torch
from PIL import Image
from torch.utils import data
import misc as ms
from torchvision import transforms
from scipy.ndimage.morphology import distance_transform_edt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
from pycocotools import mask as maskUtils
from skimage.transform import resize

def batch2annList(batch):
    annList = []
    image_id = int(batch["name"][0].replace("_",""))

    height, width = batch["images"].shape[-2:]

    maskObjects = batch["maskObjects"]
    maskClasses = batch["maskClasses"]
    n_objects = maskObjects[maskObjects!=255].max()
    id = 1
    for obj_id in range(1, n_objects+1):
        if obj_id == 0:
            continue

        binmask = (maskObjects == obj_id)

        segmentation = maskUtils.encode(np.asfortranarray(ms.t2n(binmask).squeeze())) 
        
        segmentation["counts"] = segmentation["counts"].decode("utf-8")
        uniques = (binmask.long()*maskClasses).unique()
        uniques = uniques[uniques!=0]
        assert len(uniques) == 1

        category_id = uniques[0].item()
        
        annList += [{"segmentation":segmentation,
                      "iscrowd":0,
                      # "bbox":maskUtils.toBbox(segmentation).tolist(),
                      "area":int(maskUtils.area(segmentation)),
                     "id":id,
                     "height":height,
                     "width":width,
                     "image_id":image_id,
                     "category_id":category_id}]
        id += 1

    return annList

class BaseDataset(data.Dataset):
    def __init__(self):
        dataset_name = type(self).__name__
        base = "/mnt/projects/counting/Saves/main/"

        if "Pascal" in dataset_name:
            self.lcfcn_path = base + "dataset:Pascal2007_model:Res50FCN_metric:mRMSE_loss:water_loss_B_config:basic/"
        
        elif "CityScapes" in dataset_name:
            self.lcfcn_path = base + "dataset:CityScapes_model:Res50FCN_metric:mRMSE_loss:water_loss_B_config:basic/"
        
        elif "CocoDetection2014" in dataset_name:
            self.lcfcn_path = base + "dataset:CocoDetection2014_model:Res50FCN_metric:mRMSE_loss:water_loss_B_config:sample3000/"

        elif "Kitti" in dataset_name:
            self.lcfcn_path = base + "dataset:Kitti_model:Res50FCN_metric:mRMSE_loss:water_loss_B_config:basic/"
            self.proposals_path = "/mnt/datasets/public/issam/kitti/ProposalsSharp/"

        elif "Plants" in dataset_name:
            self.lcfcn_path = base + "dataset:Plants_model:Res50FCN_metric:mRMSE_loss:water_loss_B_config:basic/"

        else:
            raise

        fname = base + "lcfcn_points/{}.pkl".format(dataset_name)
        
        if os.path.exists(fname):
            history = ms.load_pkl(self.lcfcn_path + "history.pkl")
            self.pointDict = ms.load_pkl(fname)

            if self.pointDict["best_model"]["epoch"] != history["best_model"]["epoch"]:            
                reset = "reset"
        else:
            if dataset_name == "PascalSmall":
                self.pointDict = ms.load_pkl(fname.replace("PascalSmall", "Pascal2012"))
            else:  
                import ipdb; ipdb.set_trace()  # breakpoint 5f76e230 //




    def get_lcfcn_pointList(self, name):
        if self.split == "val":
            return self.pointDict[name]
        else:
            return -1

    def get_sm_propList(self, name):
        propList = SharpProposals({"name":[name+".png"], 
                                   "proposals_path":[self.proposals_path] })
        return propList




