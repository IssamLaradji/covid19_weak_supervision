import os

import numpy as np
import torch
from PIL import Image
from torch.utils import data
import misc as ms
from torchvision import transforms
from scipy.ndimage.morphology import distance_transform_edt
from datasets import base_dataset

#root = '/home/arantxa_casanova/datasets/cityscapes'
root = '/mnt/datasets/public/segmentation/cityscapes'



import glob

category_id2name = {28:"bus", 26:"car", 27:"truck", 
                    24:"person", 25:"rider", 32:"motorcycle", 
                    31:"train", 33:"bicycle"}

category_id2label_id = {28:1, 26:2, 27:3, 24:4, 25:5,
                        32:6, 31:7, 33:8}


class CityScapes(base_dataset.BaseDataset):
    def __init__(self, root, split, transform_function=None,
                 resize=True):
        super().__init__()
        self.categories = []

        for k in category_id2label_id.keys():
            category_id = category_id2label_id[k]
            category_name = category_id2name[k]
            self.categories += [{'supercategory': 'none', 
                                 "id":category_id, 
                                 "name":category_name}]
        
        # for [{'supercategory': 'none', 
        #            'id': 1, 'name': 'pedestrian'}, 
        #           {'supercategory': 'none', 
        #            'id': 2, 'name': 'car'},
        #            {'supercategory': 'none', 
        #            'id': 2, 'name': 'car'}]

        self.resized = resize 
        quality = "fine"
        self.path = "/mnt/datasets/public/segmentation/cityscapes/"
        # annList_path = self.path + "/annotations/{}_gt_annList.json".format(split)
        self.annList_path = self.path + "/val_annList.json"
        assert os.path.exists(self.annList_path)
        # self.annList_path = annList_path
        
        
        self.points_path = "{}/val_lcfcnPoints.json".format(self.path)
        self.img_names = glob.glob(self.path + "/leftImg8bit/%s/*/*.png" % split)

        self.img_inst = [l.replace("_leftImg8bit.png","_gtFine_instanceIds.png").replace("/leftImg8bit/","/gtFine/") for l in self.img_names]
        self.img_labels = [l.replace("_leftImg8bit.png","_gtFine_labelIds.png").replace("/leftImg8bit/","/gtFine/") for l in self.img_names]
        self.img_json = [l.replace("_leftImg8bit.png","_gtFine_polygons.json").replace("/leftImg8bit/","/gtFine/") for l in self.img_names]


        if len(self.img_names) == 0:
            raise RuntimeError('Found 0 images, please check the data set')

        self.quality = quality
        self.split = split

        self.transform_function = transform_function()
        self.split = split
        self.n_classes = len(category_id2name) + 1
        
        assert len(category_id2name) == len(category_id2label_id) 
        # self.ratio = (1./3.)

    def __getitem__(self, index):
        
        img_path = self.img_names[index]
        mask_path = self.img_inst[index]

        name = os.path.basename(img_path)

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)

        # shape_original = image.size
        if self.resized:
            image = image.resize((1200, 600),Image.BILINEAR)
            mask = mask.resize((1200, 600),Image.NEAREST)
        
        # shape_new = image.size

        # w_ratio, h_ratio = np.array(shape_original) / np.array(shape_new)
        # w_min = 100./w_ratio
        # h_min = 100./h_ratio

        mask = np.array(mask)
        counts = np.zeros(self.n_classes-1, int)
        maskClasses = np.zeros(mask.shape,int)
        maskObjects = np.zeros(mask.shape,int)
        maskVoid = np.ones(mask.shape, int)
        points = np.zeros(mask.shape, int)
        
        uniques = np.unique(mask)

        # Pedestrians
        for category_id in category_id2label_id.keys():
            counts, points, maskClasses, maskObjects = mask2points(category_id, uniques, mask, counts, points, 
                                                           maskClasses, maskObjects)


        
        assert np.unique(maskObjects)[-1] == counts.sum()

        points = transforms.functional.to_pil_image(points[:,:,None].astype("uint8"))
        proposals_path = "/mnt/datasets/public/issam/Cityscapes/ProposalsSharp/"
        lcfcn_pointList = self.get_lcfcn_pointList(name)
        if self.split == "train":
            if self.transform_function is not None:
                image, points = self.transform_function([image, points])
            
            return {"images":image,
                    "points":points, 
                    "counts":counts,
                    "index":index,
                    "image_id":name,
                    "name":name,
                    # "maskVoid":1 - torch.LongTensor(maskVoid),
                    "dataset":"cityscapes",
                    "resized":self.resized,
                    "proposals_path":proposals_path,
                    "split":self.split,
                    "lcfcn_pointList":lcfcn_pointList}

        elif self.split in ["val", "test"]: 
            if self.transform_function is not None:
                image, points, maskObjects, maskClasses = self.transform_function([image, points, 
                    maskObjects, maskClasses])


            return {"images":image, 
                    "points":points, 
                    "counts":counts,
                    "index":index,
                    "name":name,
                    "image_id":name,
                    "maskObjects":maskObjects,
                    "maskClasses":maskClasses,
                     "maskVoid":torch.LongTensor(maskVoid),
                    "dataset":"cityscapes",
                    "resized":self.resized,
                    "proposals_path":proposals_path,
                    "split":self.split,
                    "lcfcn_pointList":lcfcn_pointList}

    def __len__(self):
        return len(self.img_names)

class CityScapesBig(CityScapes):
    def __init__(self,  root, split, transform_function=None):
        
        super().__init__(root, split, transform_function=transform_function, 
                        resize=False)


class CityScapesAll(CityScapes):
    def __init__(self,  root, split, transform_function=None):
        
        super().__init__(root, split, transform_function=transform_function, 
                        resize=True)



def mask2points(category_id, uniques, mask, counts, points, maskClasses, maskObjects):
    instances = uniques[(uniques>=category_id*1000) & (uniques<(category_id+1)*1000)]
    if len(instances) == 0:

        return counts, points, maskClasses, maskObjects

    ind = (mask>=category_id*1000) & (mask<(category_id+1)*1000)
    label_id = category_id2label_id[category_id]

    maskClasses[ind] = label_id

    instanceId = maskObjects.max()
    n_instances = 0
    for i, u in enumerate(instances):
        seg_ind = mask==u
        r, c = np.where(seg_ind)

        instanceId += 1
        n_instances += 1
        maskObjects[seg_ind] = instanceId
        dist = distance_transform_edt(seg_ind)
        yx = np.unravel_index(dist.argmax(), dist.shape)
        points[yx] = label_id

    counts[label_id-1] = n_instances
    return counts, points, maskClasses, maskObjects




import os

import numpy as np
import torch
from PIL import Image
from torch.utils import data
import misc as ms
from torchvision import transforms
from scipy.ndimage.morphology import distance_transform_edt

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]

zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

root = '/mnt/datasets/public/segmentation/cityscapes'
def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask


import glob


name2category = {"pedestrian":1, "car":2}


class CityScapesObject(data.Dataset):
    def __init__(self, root, split, transform_function=None):
        quality = "fine"
        self.root = "/mnt/datasets/public/segmentation/cityscapes/"
        
        self.img_names = glob.glob(self.root + "/leftImg8bit/%s/*/*.png" % split)

        self.img_inst = [l.replace("_leftImg8bit.png","_gtFine_instanceIds.png").replace("/leftImg8bit/","/gtFine/") for l in self.img_names]
        self.img_labels = [l.replace("_leftImg8bit.png","_gtFine_labelIds.png").replace("/leftImg8bit/","/gtFine/") for l in self.img_names]
        self.img_json = [l.replace("_leftImg8bit.png","_gtFine_polygons.json").replace("/leftImg8bit/","/gtFine/") for l in self.img_names]

        if len(self.img_names) == 0:
            raise RuntimeError('Found 0 images, please check the data set')

        self.quality = quality
        self.split = split

        self.ignore_index = ignore_label = 255

        self.id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                              3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                              7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                              14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                              18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                              28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}


        self.transform_function = transform_function()
        self.split = split
        self.n_classes = 3
        # self.ratio = (1./3.)

    def __getitem__(self, index):
        img_path = self.img_names[index]
        mask_path = self.img_inst[index]
        

        name = os.path.basename(img_path)

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)

        shape_original = image.size
        image = image.resize((1200, 600),Image.BILINEAR)
        mask = mask.resize((1200, 600),Image.NEAREST)

        shape_new = image.size

        w_ratio, h_ratio = np.array(shape_original) / np.array(shape_new)
        w_min = 100./w_ratio
        h_min = 100./h_ratio

        mask = np.array(mask)

        maskClasses = np.zeros(mask.shape,int)
        maskObjects = np.zeros(mask.shape,int)
        maskVoid = np.zeros(mask.shape, int)
        points = np.zeros(mask.shape, int)
    
        uniques = np.unique(mask)

        # Pedestrians
        ind = (mask>=24*1000) & (mask<25*1000)
        maskClasses[ind] = 1

        n_pedestrians = 0

        for i, u in enumerate(uniques[(uniques>=24*1000) & 
                             (uniques<25*1000)]):
            seg_ind = mask==u
            r, c = np.where(seg_ind)

            if (r.max()-r.min()) < h_min or (c.max()-c.min()) < w_min:
                maskVoid[seg_ind] = 1
                continue
            n_pedestrians += 1
            maskObjects[seg_ind] = n_pedestrians
            dist = distance_transform_edt(seg_ind)
            yx = np.unravel_index(dist.argmax(), dist.shape)
            points[yx] = 1


        # Cars
        ind = (mask>=26*1000) & (mask<27*1000)
        maskClasses[ind] = 2

        n_cars = 0
        for i, u in enumerate(uniques[(uniques>=26*1000) & 
                             (uniques<27*1000)]):
            
            seg_ind = mask==u
            r, c = np.where(seg_ind)
            if (r.max()-r.min()) < h_min or (c.max()-c.min()) < w_min:
                maskVoid[seg_ind] = 1
                continue

            n_cars += 1
            maskObjects[seg_ind] = n_cars + n_pedestrians
            dist = distance_transform_edt(seg_ind)
            yx = np.unravel_index(dist.argmax(), dist.shape)     
            points[yx] = 2


        counts = np.array([n_pedestrians, n_cars])
        assert np.unique(maskObjects)[-1] == counts.sum()

        points = transforms.functional.to_pil_image(points[:,:,None].astype("uint8"))
        proposals_path = "/mnt/datasets/public/issam/Cityscapes/ProposalsSharp/"
        if self.split == "train":
            if self.transform_function is not None:
                image, points = self.transform_function([image, points])
            
            return {"images":image,
                    "points":points, 
                    "counts":counts,
                    "index":index,
                    "image_id":name,
                    "name":name,
                    "resized":True,
                    "maskVoid":1 - torch.LongTensor(maskVoid),
                    "dataset":"cityscapes",
                    "split":self.split,
                    "proposals_path":proposals_path}

        elif self.split in ["val", "test"]: 
            if self.transform_function is not None:
                image, points, maskObjects, maskClasses = self.transform_function([image, points, 
                    maskObjects, maskClasses])


            return {"images":image, 
                    "points":points, 
                    "counts":counts,
                    "index":index,
                    "name":name,
                    "image_id":name,
                    "maskObjects":maskObjects,
                    "maskClasses":maskClasses,
                    "maskVoid":1 - torch.LongTensor(maskVoid),
                    "dataset":"cityscapes",
                    "split":self.split,
                    "resized":True,
                    "proposals_path":proposals_path}

    def __len__(self):
        return len(self.img_names)