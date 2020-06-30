import os

import glob
import numpy as np
import torch
from PIL import Image
from torch.utils import data

import misc as ms
from datasets import base_dataset
palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]

zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

#root = '/home/arantxa_casanova/datasets/cityscapes'
root = '/mnt/datasets/public/segmentation/cityscapes'
def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask



class Kitti(base_dataset.BaseDataset):
    def __init__(self, root, split, transform_function=None):
        super().__init__()
        self.path = "/mnt/datasets/public/issam/kitti/"
        path_splits = self.path + "/kitti_loader/"
        if split == "train":
            self.img_names = [t.replace("\n","") for t
             in ms.read_text(path_splits + "image_sets/train3712.txt")]
        
        elif split == "val":
            # self.img_names = [t.replace("\n","") for t
            #  in ms.read_text(path_splits + "image_sets/val120.txt")]

            self.img_names = [t.replace("\n","") for t
             in ms.read_text(path_splits + "image_sets/test144.txt")]
            
            annList_path = self.path + "/annotations/{}_gt_annList.json".format(split)
            assert os.path.exists(annList_path)
            self.annList_path = annList_path
            
        elif split == "test":
            self.img_names = [t.replace("\n","") for t 
                in ms.read_text(path_splits + "image_sets/test144.txt")]


        self.labels_path = self.path + "/training/label_2/" 
        self.image_path = self.path + "/training/image_2/" 
        self.ann_path = self.path + "/kitti_loader/annotations/"
        
        
        self.transform_function = transform_function()
        self.split = split
        self.n_classes = 2

        self.categories = [{'supercategory': 'none', 
                             "id":1, 
                             "name":"car"}]

        #self.ratio = 0.5

    def __getitem__(self, index):
        name = self.img_names[index]
        

        image_name = self.image_path + name+".png"
        label_name = self.labels_path + name+".txt"
        ann_name = self.ann_path + name+".png"

        image = Image.open(image_name).convert('RGB')
        labels_txt = ms.read_text(label_name)
        w, h = image.size

        points = np.zeros((h,w)).astype(int)
        counts = np.zeros(self.n_classes-1, int)
        maskClasses = np.zeros((h,w),int)
        maskObjects = np.zeros((h,w),int)

        for lb in labels_txt:
            axioms = lb.split()
            if axioms[0] == "Car":
                x_min, y_min, x_max, y_max = axioms[4:8]
                y = int((float(y_max) + float(y_min))/2)
                x = int((float(x_max) + float(x_min))/2)

                points[y, x] = 1

        # if self.split == "val":
        #     labels = Image.open(ann_name)
        # else:
        #     labels = np.zeros((h,w)).astype(int)
        #     labels = Image.fromarray(labels.astype(np.uint8))

        points = Image.fromarray(points.astype(np.uint8))
        #labels = labels.resize((int(w), int(h)), Image.NEAREST)
        #vis.images(image, mask)
        if self.split == "train":
            image, points = self.transform_function([image, points])
        else:
            labels = Image.open(ann_name)
            labels = np.array(labels)

            # maskObjects = labels
            for i, l in enumerate(np.unique(labels)):
                if l == 0:
                    continue
                assert i > 0
                maskObjects[labels==l] = i

            maskClasses[maskObjects!=0] = 1
            image, points, maskObjects, maskClasses = self.transform_function([image, points, 
                    maskObjects, maskClasses])

        counts = torch.zeros(1).long()
        counts[0] = int(points.sum())
        
        lcfcn_pointList = self.get_lcfcn_pointList(name)
        # sm_propList = self.get_sm_propList(name)

        batch = {"images":image, "SharpProposals_name":name+".png",
                "points":points, "counts":counts, "dataset":"Kitti",
                "index":index,
                "name":name,
                "maskObjects":maskObjects,
                "maskClasses":maskClasses,
                # "sm_propList":sm_propList.proposals,
                "proposals_path":self.proposals_path,
                "lcfcn_pointList":lcfcn_pointList,
                "split":self.split}

        return batch
                
        

    def __len__(self):
        return len(self.img_names)