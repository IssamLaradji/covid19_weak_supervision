import os

from addons import transforms as myTransforms
import numpy as np
import scipy.io as sio
import torch
from PIL import Image
from torch.utils import data
#from pycocotools import mask as maskUtils
import misc as ms
from datasets import helpers as d_helpers
from datasets import base_dataset
from torchvision import transforms

def load_mask(mask_path):
    if ".mat" in mask_path:
        inst_mask = sio.loadmat(mask_path)['GTcls']['Segmentation'][0][0]
        inst_mask = Image.fromarray(inst_mask.astype(np.uint8))
    else:
        inst_mask = Image.open(mask_path)

    return inst_mask


class Pascal2012(base_dataset.BaseDataset):
    def __init__(self, root, split, transform_function):
        super().__init__()
        self.split = split

        self.path = "/mnt/datasets/public/issam/VOCdevkit"

        self.categories = ms.load_json("/mnt/datasets/public/issam/"
                       "VOCdevkit/annotations/pascal_val2012.json")["categories"]

        assert split in ['train', 'val', 'test']
        self.img_names = []
        self.mask_names = []
        self.cls_names = []

        berkley_root =  os.path.join(self.path, 'benchmark_RELEASE')
        pascal_root = os.path.join(self.path)

        data_dict = d_helpers.get_augmented_filenames(pascal_root, 
                                                      berkley_root, 
                                                      mode=1)
        # train
        assert len(data_dict["train_imgNames"]) == 10582
        assert len(data_dict["val_imgNames"]) == 1449

        berkley_path = berkley_root + '/dataset/'
        pascal_path = pascal_root + '/VOC2012/'

        corrupted=["2008_005262",
                   "2008_004172",
                   "2008_004562",
                   "2008_005145",
                   "2008_008051",
                   "2008_000763",
                   "2009_000573"]

        if split == 'train':
            for name in  data_dict["train_imgNames"]:

                name_img = os.path.join(berkley_path, 'img/' + name + '.jpg')
                if os.path.exists(name_img):
                    name_img = name_img
                    name_mask = os.path.join(berkley_path, 'cls/' + name + '.mat')
                else:
                    name_img = os.path.join(pascal_path, 'JPEGImages/' + name + '.jpg')
                    name_mask =  os.path.join(pascal_path, 'SegmentationLabels/' +  name + '.jpg')


                self.img_names += [name_img]
                self.mask_names += [name_mask]

        elif split in ['val', "test"]:
            data_dict["val_imgNames"].sort() 
            for k, name in  enumerate(data_dict["val_imgNames"]):

                if name in corrupted:
                    continue
                name_img = os.path.join(pascal_path, 'JPEGImages/' + name + '.jpg')
                name_mask =  os.path.join(pascal_path, 'SegmentationObject/' + 
                                          name  + '.png')
                name_cls =  os.path.join(pascal_path, 'SegmentationClass/' + name + '.png')

                if not os.path.exists(name_img):
                    name_img = os.path.join(berkley_path, 'img/' + name + '.jpg')
                    name_mask =  os.path.join(berkley_path, 'inst/' + name + '.mat')
                    name_cls =  os.path.join(berkley_path, 'cls/' + name + '.mat')

                assert os.path.exists(name_img)
                assert os.path.exists(name_mask)
                assert os.path.exists(name_cls)

                self.img_names += [name_img]
                self.mask_names += [name_mask]
                self.cls_names += [name_cls]

        self.proposals_path = "/mnt/datasets/public/issam/VOCdevkit/VOC2012/ProposalsSharp/"
        if len(self.img_names) == 0:
            raise RuntimeError('Found 0 images, please check the data set')

        self.n_classes = 21
        self.transform_function = transform_function()
    
        self.ignore_index = 255
        self.pointsJSON = ms.jload(os.path.join( 
                                    '/mnt/datasets/public/issam/VOCdevkit/VOC2012',
                                    'whats_the_point/data', 
                                    "pascal2012_trainval_main.json"))


    def __getitem__(self, index):
        # Image
        img_path = self.img_names[index]
        image = Image.open(img_path).convert('RGB')

        # Points
        name = ms.extract_fname(img_path).split(".")[0]
        points, counts = ms.point2mask(self.pointsJSON[name], image, return_count=True, n_classes=self.n_classes-1)
        points = transforms.functional.to_pil_image(points)

        counts = torch.LongTensor(counts)
        original = transforms.ToTensor()(image)
        
        if self.split == "train":
            if self.transform_function is not None:
                image, points = self.transform_function([image, points])
            
            return {"images":image,
                    "original":original, 
                    "points":points, 
                    "counts":counts,
                    "index":index,
                    "name":name,
                    "dataset":"voc",
                    "resized":False,
                    "proposals_path":self.proposals_path,
                    "split":self.split}

        elif self.split in ["val", "test"]: 
            # Mask
            mask_path = self.mask_names[index]
            mask = load_mask(mask_path)

            # Mask
            cls_path = self.cls_names[index]
            maskClass = load_mask(cls_path)
            
            if self.transform_function is not None:
                image, points, mask, maskClass = self.transform_function([image, points, 
                    mask,maskClass])

            maskVoid = maskClass != 255
            maskClass[maskClass==255] = 0
            mask[mask==255] = 0
            lcfcn_pointList = self.get_lcfcn_pointList(name)
            return {"images":image, 
                    "original":original,
                    "points":points, 
                    "counts":counts,
                    "index":index,
                    "name":name,
                    "image_id":int(name.replace("_","")),
                    "maskObjects":mask,
                    "maskClasses":maskClass,
                    "maskVoid":maskVoid.long(),
                    "dataset":"voc",
                    "lcfcn_pointList":lcfcn_pointList,
                    "proposals_path":self.proposals_path,
                    "split":self.split}
    def __len__(self):
        return len(self.img_names)



#------ aux

def make_dataset(path, split):
    assert split in ['train', 'val', 'test']
    data_dict = {"img_names": [], "labels": []}

    if split == 'train':
        path = os.path.join(path, 'benchmark_RELEASE', 'dataset')
        img_path =  path +'/img'
        mask_path = path +'/cls'
        data_list = [l.strip('\n') for l in open(path + '/train.txt').readlines()]
        ext = ".mat"

    elif split == 'val':    
        path = os.path.join(path, 'VOC2012')
        img_path = path + '/JPEGImages'
        mask_path =  path + '/SegmentationClass'
        
        data_list = [l.strip('\n') for l in open(os.path.join(path,
            'ImageSets', 'Segmentation', 'seg11valid.txt')).readlines()]

        ext = ".png"
        corrupted = ["2008_000763", "2008_004172",
                     "2008_004562", "2008_005145",
                     "2008_005262", "2008_008051"] 

        data_list = np.setdiff1d(data_list, corrupted)

                                                                      
    elif split == 'test':  
        path = os.path.join(path, 'VOC2012')
        img_path = path + '/JPEGImages'
        mask_path =  path + '/SegmentationClass'
        
        data_list = [l.strip('\n') for l in open(os.path.join(path,
            'ImageSets', 'Segmentation', 'seg11valid.txt')).readlines()]

        ext = ".png"
        corrupted = ["2008_000763", "2008_004172",
                     "2008_004562", "2008_005145",
                     "2008_005262", "2008_008051"] 

        data_list = np.setdiff1d(data_list, corrupted)

    
    for it in data_list:
        data_dict["img_names"] += [os.path.join(img_path, it + '.jpg')]
        data_dict["labels"] += [os.path.join(mask_path, it + '%s' % ext)]

    return data_dict


palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128,
           128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128,
           64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128]

zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask
