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



class Trancos(data.Dataset):
    def __init__(self, root="",split=None, 
                 transform_function=None):
        
        self.split = split
        
        self.n_classes = 2
        self.n_channels = 3
        self.transform_function = transform_function()
        self.name = "Trancos"
        
        ############################
        self.path_base = "/mnt/datasets/public/issam/Trancos/"

        if split == "train":
            fname = self.path_base + "/image_sets/training.txt"

        elif split == "val":
            fname = self.path_base + "/image_sets/validation.txt"

        elif split == "test":
            fname = self.path_base + "/image_sets/test.txt"

        self.img_names = [name.replace(".jpg\n","") for name in ms.read_text(fname)]
        self.path = self.path_base + "/images/"
        assert os.path.exists(self.path + self.img_names[0] + ".jpg")
        

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        name = self.img_names[index]

        # LOAD IMG, POINT, and ROI
        image = imread(self.path + name + ".jpg")
        points = imread(self.path + name + "dots.png")[:,:,:1].clip(0,1)
        roi = loadmat(self.path + name + "mask.mat")["BW"][:,:,np.newaxis]
        
        # LOAD IMG AND POINT
        image = image * roi
        image = ms.shrink2roi(image, roi)
        points = ms.shrink2roi(points, roi).astype("uint8")
        
        counts = torch.LongTensor(np.array([int(points.sum())]))

        collection = list(map(FT.to_pil_image, [image, points]))
        if self.transform_function is not None:
            image, points = self.transform_function(collection)
            
        if np.all(points == -1):
            pass
        else:
            assert int(points.sum()) == counts[0]

        return {"images":image, "points":points, "split":self.split,
                "counts":counts,"path":self.path, "index":index, "name":name}
