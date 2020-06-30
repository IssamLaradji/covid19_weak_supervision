import time
from torch.utils import data
import glob
import datetime
import pandas as pd 
import scipy.misc as m
from bs4 import BeautifulSoup
import numpy as np
import torch
from torch.nn import functional as F
import pickle
import os
from skimage.io import imread
from scipy.ndimage.filters import gaussian_filter 
from torch.autograd import Variable
import json
import glob
import glob
import datetime
import pandas as pd 
import scipy.misc as m
from bs4 import BeautifulSoup
import numpy as np
import torch
import imageio
from scipy.io import loadmat
import misc as ms
import cv2
from skimage.segmentation import mark_boundaries
from tqdm import tqdm
from skimage.segmentation import slic
from torchvision import transforms
from PIL import Image
import torchvision
import torchvision.transforms.functional as FT

from skimage.transform import rescale
import misc as ms
from addons.pycocotools import mask as maskUtils
from scipy.ndimage.morphology import distance_transform_edt
from datasets import base_dataset


from addons.pycocotools.coco import COCO
class CocoDetection(base_dataset.BaseDataset):

    # def save_gt_pointDict(self, path):
    #     fname = path + "gt_pointjson"
    #     ms.save_json()

    def __init__(self, root="",split=None, 
                 transform_function=None, ratio=None, year="2017"):        
        super().__init__()
        fname = split

        if fname == "test":
            fname = "val"

        dataset_name = "COCO"

        if year == "2014":
            dataset_name = "COCO2014"

        

        self.n_classes = 81
       
        self.path = "/mnt/datasets/public/issam/{}/".format(dataset_name)
        self.proposals_path = "{}/ProposalsSharp/".format(self.path)
        self.split = split
        self.year = year
        self.transform_function = transform_function()
        fname_names = self.path + "/{}.json".format(self.split)
        fname_catids = self.path + "/{}_catids.json".format(self.split)
        fname_categories = self.path + "/categories.json"
        fname_ids = self.path + "/{}_ids.json".format(self.split)
        
        if os.path.exists(fname_names):

            self.image_names = ms.load_json(fname_names)
            self.catids = ms.load_json(fname_catids)
            self.categories = ms.load_json(fname_categories)
            self.ids = ms.load_json(fname_ids)
        else:    
            # Save ids

            annFile = "{}/annotations/instances_{}{}.json".format(self.path, fname,year)
            self.coco = COCO(annFile)
            self.ids = list(self.coco.imgs.keys())

            self.image_names = []
            # Save Labels 
            for index in range(len(self.ids)):
                print(index, "/", len(self.ids))
                img_id = self.ids[index]
                ann_ids = self.coco.getAnnIds(imgIds=img_id)
                annList = self.coco.loadAnns(ann_ids)
                name = self.coco.loadImgs(img_id)[0]['file_name']

                self.image_names += [name]
                ms.save_pkl(self.path + 
                             "/groundtruth/{}_{}.pkl".format(self.split, name),
                             annList)

            ms.save_json(fname_names, self.image_names)

            # Catgory
            self.catids = self.coco.getCatIds()
            ms.save_json(fname_catids, self.catids)

            self.categories = []

            categories = self.coco.cats.values()

            for c in categories:
                c["id"] = self.category2label[c["id"]]
                self.categories += [c]
            ms.save_json(fname_categories, self.categories)


           
            ms.save_json(fname_ids, self.ids)


            if split == "val":
                # gt_annDict = ms.load_json(annFile)

                annDict = {}
                # fname_ann = '/mnt/datasets/public/issam/COCO2014//annotations/val_gt_annList.json'
                annDict["categories"] = self.categories
                annDict["images"] = self.coco.loadImgs(self.ids[:5000])

                annIDList = self.coco.getAnnIds(self.ids[:5000])  
                annList = self.coco.loadAnns(annIDList)

                for p in annList:
                   # p["id"] = str(p["id"])
                   p["image_id"] = str(p["image_id"])
                   p["category_id"] =  self.category2label[p["category_id"]]

                for p in annDict["images"]:
                    p["id"] = str(p["id"])
                annDict["annotations"] = annList

                ms.save_json('{}//annotations/val_gt_annList.json'.format(self.path),
                             annDict)



        self.category2label = {}
        self.label2category = {}

        for i, c in enumerate(self.catids):
            self.category2label[c] = i + 1
            self.label2category[i+1] = c


        if split == "val":
            # gt_annList_path = '/mnt/datasets/public/issam/COCO2014//annotations/val_gt_annList.json'

            annList_path = self.path + "/annotations/{}_gt_annList.json".format(split)
            
            assert os.path.exists(annList_path)
            self.annList_path = annList_path

            # self.image_names.sort()
            self.image_names = self.image_names[:5000]
            self.ids = self.ids[:5000]

        elif split == "test":
            # self.image_names.sort()
            self.image_names = self.ids[-5000:]

                    

    def __getitem__(self, index):
        name_id = self.ids[index]
        name = self.image_names[index]
        
        image = np.array(Image.open(self.path + "/{}/{}".format(self.split+self.year, name)).convert('RGB'))
        points = np.zeros(image.shape[:2], "uint8")[:,:,None]
        counts = np.zeros(80)
        maskVoid = np.zeros(points.shape[:2])
        annList = ms.load_pkl(self.path + 
                             "/groundtruth/{}_{}.pkl".format(self.split, name))

        h, w, _ = image.shape
        maskClasses = np.zeros((h,w),int)
        maskObjects = np.zeros((h,w),int)
        for obj_id, ann in enumerate(annList):
            mask = maskUtils.decode(COCO.annToRLE_issam(h, w, ann))
            if ann["iscrowd"]:
                maskVoid += mask
            else:

                dist = distance_transform_edt(mask)
                yx = np.unravel_index(dist.argmax(), dist.shape)

                label = self.category2label[int(ann["category_id"])]
                points[yx] = label
                counts[label - 1] += 1

                assert mask.max() <= 1
                mask_ind =  mask==1

                maskObjects[mask_ind] = obj_id+1
                maskClasses[mask_ind] = label


        maskVoid = maskVoid.clip(0, 1)
        assert maskVoid.max() <= 1
        counts = torch.LongTensor(counts)
        
        image, points, maskObjects, maskClasses = self.transform_function([image, points, 
                    maskObjects, maskClasses])



        # Sharp Proposals
        image_id = int(name[:-4].split("_")[-1])
        SharpProposals_name = self.proposals_path +  "{}".format(image_id)
        lcfcn_pointList = self.get_lcfcn_pointList(str(image_id))
        assert image_id == name_id
        if self.split == "train":
            return {"images":image, "points":points.squeeze(), 
                    "SharpProposals_name":str(name_id),
                    "counts":counts, "index":index,
                    "name":str(name_id),
                    "image_id":str(image_id),
                    "maskObjects":maskObjects*0,
                    "maskClasses":maskClasses*0,
                    "proposals_path":self.proposals_path,
                    "dataset":"coco2014",
                    "lcfcn_pointList":lcfcn_pointList,
                        "split":self.split}
        else:
            return {"images":image, "points":points.squeeze(), 
                    "SharpProposals_name":str(name_id),
                    "counts":counts, "index":index,
                    "name":str(name_id),
                    "image_id":str(image_id),
                    "maskObjects":maskObjects,
                    "maskClasses":maskClasses,
                    "proposals_path":self.proposals_path,
                    "dataset":"coco2014",
                    "lcfcn_pointList":lcfcn_pointList,
                        "split":self.split}


    def __len__(self):
        return len(self.image_names)


class CocoDetection2014(CocoDetection):
    def __init__(self, root="",split=None, 
                 transform_function=None):
        
        super().__init__(root=root,split=split, 
                 transform_function=transform_function,year="2014")

