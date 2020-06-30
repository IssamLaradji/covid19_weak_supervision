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
import json
import glob
from torch.utils import data
import glob
import datetime
import pandas as pd 
import scipy.misc as m
from bs4 import BeautifulSoup
import numpy as np
import torch
import imageio
from scipy.io import loadmat
# import utils_commons as ut
import cv2
from skimage.segmentation import mark_boundaries
from tqdm import tqdm
from skimage.segmentation import slic
from torchvision import transforms
from PIL import Image
# import visualize as vis
import torchvision
import torchvision.transforms.functional as FT
import glob
import h5py
from scipy.misc import imsave
import misc as ms 

def save_images_labels(path):
    img_folder =  h5py.File(path +
                            "CVPPP2017_training_images.h5", 
                            'r')["A1"]
    gt_folder =  h5py.File(path+"CVPPP2017_training_truth.h5", 
                                      'r')["A1"]

    img_names_train = list(img_folder.keys())

    # for name in img_names:
    #     image = np.array(img_folder[name]["rgb"])[:,:,:3] 
    #     points = np.array(img_folder[name]["centers"])
    #     img_path = path + "/images/{}.png".format(name)
    #     points_path = path + "/labels/{}_points.png".format(name)
    #     ms.imsave(img_path, image)
    #     ms.imsave(points_path, points)

    #     assert (ms.imread(img_path) == image).mean()==1
    #     assert (ms.imread(points_path) == points).mean()==1

    #     maskObjects_path = path + "/labels/{}_maskObjects.png".format(name)
    #     maskObjects = np.array(gt_folder[name]["label"])
    #     ms.imsave(maskObjects_path, maskObjects)

    #     assert (ms.imread(maskObjects_path) == maskObjects).mean()==1

    print("| DONE TRAIN", len(img_names_train))
    # Test
    img_folder =  h5py.File(path+
                                    "CVPPP2017_testing_images.h5", 
                                      'r')["A1"]


    img_names = list(img_folder.keys())
    assert np.in1d(img_names_train, img_names).mean() == 0

    for name in img_names:
        image = np.array(img_folder[name]["rgb"])[:,:,:3] 
        img_path = path + "/images/{}.png".format(name)
        ms.imsave(img_path, image)
        assert (ms.imread(img_path) == image).mean()==1
    print("| DONE TEST", len(img_names))
    
from datasets import base_dataset
class Plants(base_dataset.BaseDataset):
    def __init__(self,root,split=None, 
                                  transform_function=None):
        super().__init__()
        self.split = split

        self.path = "/mnt/datasets/public/issam/sbd/"
        self.proposals_path = self.path + "/ProposalsSharp/"
        img_folder =  h5py.File(self.path +
                            "CVPPP2017_training_images.h5", 
                            'r')["A1"]


        self.img_names = list(img_folder.keys())
        self.img_names.sort()
        self.annList_path = "/mnt/datasets/public/issam/sbd/annotations/val_gt_annList.json"
        # save_images_labels(self.path)
        self.transform_function = transform_function()

        # save_images_labels(self.path)
        if split == "train":
            self.img_indices = np.arange(28,len(self.img_names))

        elif split == "val":
            self.img_indices = np.arange(28)


        elif split == "test":
            self.img_folder =  h5py.File(self.path+
                                    "CVPPP2017_testing_images.h5", 
                                      'r')["A1"]

            self.img_names = list(self.img_folder.keys())
            self.img_indices = np.arange(len(self.img_names))
        else:
            raise ValueError("split does not exist...")

        self.n_images = len(self.img_indices)
        self.n_classes = 2
        self.categories = [{'supercategory': 'none', 
                             "id":1, 
                             "name":"plant"}]   

        # base = "/mnt/projects/counting/Saves/main/"
        # self.lcfcn_path = base + "dataset:Plants_model:Res50FCN_metric:mRMSE_loss:water_loss_B_config:basic/"
        # import ipdb; ipdb.set_trace()  # breakpoint 20711b9f //
        
        # self.pointDict = ms.load_pkl(self.lcfcn_path+"lcfcn_points/Pascal2012.pkl")

    def __len__(self):
        return self.n_images

    def __getitem__(self, index):        
        name = self.img_names[self.img_indices[index]]

        image = ms.imread(self.path + "/images/{}.png".format(name))
        h, w = image.shape[:2]
        if self.split in ["test"]:
            points = np.zeros((h, w), "uint8")
            points[0,0]=1
        else:
            points = ms.imread(self.path + "/labels/{}_points.png".format(name))

        

        
        counts = torch.LongTensor(np.array([int(points.sum())]))
        # LOAD IMG AND POINT
        collection = list(map(FT.to_pil_image, [image, 
                                                points[:,:,None]]))

        if self.split in ["train"]:
            if self.transform_function is not None:
                image, points = self.transform_function(collection)

            return {"images":image, "points":points, 
                    "counts":counts, "SharpProposals_name":name+".png", 
                    "name":name, 
                    "index":index,
                    "proposals_path":self.proposals_path,
                    "dataset":"Plants",
                    "single_point":True,
                    "split":self.split} 

        if self.split in ["test"]:
            collection = list(map(FT.to_pil_image, [image, 
                                                points[:,:,None],
                                                points[:,:,None],
                                                points[:,:,None]]))

            if self.transform_function is not None:
                image, points,_,_ = self.transform_function(collection)

            return {"images":image, "points":points, 
                    "counts":counts, "SharpProposals_name":name+".png", 
                    "name":name, 
                    "index":index,
                    "proposals_path":self.proposals_path,
                    "dataset":"Plants",
                    "single_point":True,
                    "split":self.split}  

        else:
            maskObjects_path = self.path + "/labels/{}_maskObjects.png".format(name)
            maskObjects =  ms.imread(maskObjects_path)
            maskClasses = np.zeros(maskObjects.shape)
            maskClasses[maskObjects!=0] = 1
            image, points, maskObjects, maskClasses = self.transform_function([image, points, 
                    maskObjects, maskClasses])

            lcfcn_pointList = self.get_lcfcn_pointList(name)
            return {"images":image, "points":points, "SharpProposals_name":name+".png",
                    "counts":counts, 
                    "name":name, "dataset":"Plants",
                    "index":index,
                    "maskObjects":maskObjects,
                    "maskClasses":maskClasses,
                    "proposals_path":self.proposals_path,
                    "lcfcn_pointList":lcfcn_pointList,
                    "single_point":True,
                    "split":self.split}            


            # if self.transform_function is not None:
            #     img, points = self.transform_function(collection)



            # return {"images":image, "SharpProposals_name":name+".png",
            #     "points":points, "counts":counts,
            #     "index":index,"dataset":"Plants",
            #     "name":name,
            #     "maskObjects":maskObjects,
            #     "maskClasses":maskClasses,
            #     "proposals_path":self.proposals_path}


def save_test_to_h5(main_dict):
    from torch.utils import data 
    model = ms.load_best_model(main_dict)
    model.eval()
    test_set = ms.load_test(main_dict)


    loader = data.DataLoader(test_set, batch_size=1, 
                             num_workers=0, drop_last=False)

    import ipdb; ipdb.set_trace()  # breakpoint cbf0e183 //

    sub = h5py.File("/mnt/datasets/public/issam/sbd/submission_example.h5", "r+")
    "/mnt/home/issam/Summaries/submission_example.h5"

    "cp /mnt/datasets/public/issam/sbd/submission_example.h5 /mnt/home/issam/Summaries/submission_example.h5"

   

    for i, batch in enumerate(loader):
        print(i/len(loader))
        pred_dict = model.predict(batch, predict_method="blobs")
        blobs = pred_dict["blobs"].astype("uint8").squeeze()
        data = sub["A1"][batch["name"][0]]["label"]     
        data[...] = blobs                    
    
    #hf.create_dataset('dataset_2', data=d2, compression="gzip", compression_opts=9)
    sub.close() 
    import ipdb; ipdb.set_trace()  # breakpoint 84427bd1 //

      