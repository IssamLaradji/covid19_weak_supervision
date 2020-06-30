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


name2category = { "pedestrian":1, "car":2}


class CityScapesOld(data.Dataset):
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

        if self.split == "train":
            if self.transform_function is not None:
                image, points = self.transform_function([image, points])
            
            return {"images":image,
                    "points":points, 
                    "counts":counts,
                    "index":index,
                    "image_id":name,
                    "name":name,
                    "maskVoid":1 - torch.LongTensor(maskVoid),
                    "dataset":"cityscapes"}

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
                    "dataset":"cityscapes"}

    def __len__(self):
        return len(self.img_names)