from torch.utils import data
import pandas as pd 
from bs4 import BeautifulSoup
import numpy as np
import torch
import pickle
import os

from torchvision import transforms
from PIL import Image
from scipy.ndimage.filters import gaussian_filter 
import misc as ms

def read_xml(fname):
    with open(fname) as f:
        xml = f.readlines()
    xml = ''.join([line.strip('\t') for line in xml])
    
    xml = BeautifulSoup(xml, "lxml")

    return xml

def read_img(fname):
    pass

def get_meta(path, corrupted):
    xml = read_xml(path + "ImageData.xml")
    xml_meta = xml.findAll("image")

    meta = []
    for img in xml_meta:
        img_name = str(img.data.contents[0])

        if img_name in corrupted:
            continue

        points = img.findAll("point")
     
        pointList = []
        n_count = 0
        for p in points:
            xc = float(p.x.contents[0])
            yc = float(p.y.contents[0])

            pointList += [{"y":yc, "x":xc}]

            n_count += 1


        meta += [{
                  "img":img_name, 
                  "pointList":pointList,
                  "n_fisher":n_count}]
    return meta



class Fisheries(data.Dataset):
    def __init__(self, root, split,
                 lake="", transform_function=None, 
                 corrupted=(None,), resize=True):


        
        self.split = split
        self.transform_function = transform_function()
        
        # 2. GET META

        path = "/mnt/home/issam/DatasetsPrivate/fisheries/%s/" % lake


        mname = path + '/logs/meta_%s.pkl' % lake
        ms.create_dirs(mname)

        if os.path.exists(mname) and False:
            with open(mname, 'rb') as f:
                meta = pickle.load(f)
        else:
            meta = get_meta(path, corrupted)

            # read python dict back from the file
            with open(mname, 'wb') as f:
                pickle.dump(meta, f)

        n = len(meta)
        all_fishers = np.array(pd.DataFrame(list(meta))["n_fisher"])
        n_mean = all_fishers.mean()


        # 1. LAKE CONFIGURATIONS
        e_val = int(np.where(all_fishers.cumsum() > n_mean*100)[0].min())
        n_val = int(e_val*0.2)
        e_train = e_val - n_val

        e_train = 300
        e_val = 350
        if split == "train":
          indices = np.arange(e_train) 

          
        if split == "val":
          indices = np.arange(e_train, e_val)

        elif split == "test":
          indices = np.arange(e_val, n)

        self.meta = np.array(meta)[indices]
        self.n_objects = self.n_fishers = np.array(pd.DataFrame(list(self.meta))["n_fisher"])
        self.path = path
        self.n_classes = 2

        
        #score = self.n_objects.mean()
        #print("%s - Baseline (always-0): %.3f" % (self.split, score))
        
    def __len__(self):
        return len(self.meta)


    def getitem(self, index):
        img_meta = self.meta[index]
        img_path = self.path + img_meta["img"]

        _img = Image.open(img_path).convert('RGB')
        
        # GET POINTS
        w, h = _img.size

        _img = _img.resize((int(w * self.ratio), int(h * self.ratio)), Image.BILINEAR)
        #_img = _img.resize((224, 224), Image.BILINEAR)

        
        w, h = _img.size

        points = np.zeros((h, w), np.uint8)

        for p in img_meta["pointList"]:
            points[int(p["y"] * h), 
                   int(p["x"] * w)] = 1
        
        _points = transforms.functional.to_pil_image(points[:,:,np.newaxis])
        """
        iii, ppp = self.transform_function([_img, _points]); vis.images(np.array(iii), mask=np.array(ppp), enlarge=1)
        """
        
        if self.transform_function is not None:
            _img, _points = self.transform_function([_img, _points])


        # COMPUTE COUNTS
        counts = torch.from_numpy(np.array([_points.sum()])).long()

        # density = gaussian_filter(ms.t2n(_points).astype(float), sigma=8.0)
        # density = torch.from_numpy(density).float()

        # #### ADD DENSITY MAP
        # if self.density:
        #density = gaussian_filter(ms.t2n(_points).astype(float), sigma=8.0)
        density = gaussian_filter(ms.t2n(_points).astype(float), sigma=1.0)
        density = torch.from_numpy(density).float()

        return {"images":_img, "name":img_meta["img"],
        "points":_points, 
                "counts":counts, "density":density, "index":index,
                "split":self.split}

    

    def __getitem__(self, index):      
      return self.getitem(index)

class ComoLake(Fisheries):
    def __init__(self, root="", download=0, split=None, 
                 transform_function=None, 
                 ratio=0.1, density=0, sigma=8.0):
        self.split = split
        lake = "Como_Lake_2"
        self.ratio = ratio

        
        self.sigma = sigma
        self.density = density


        corrupted = []    
        super(ComoLake, self).__init__(root,
                  lake=lake, 
                 transform_function=transform_function,
                 corrupted=corrupted,
                 split=split)

        
def find_corrupted(imgList):
  corrupted = []
  n= len(imgList)
  for j, img_path in enumerate(imgList):
    print(j, n)
    try:
      _img = Image.open(img_path).convert('RGB')
      
    except:
      corrupted += [ut.extract_fname(img_path)]

  return corrupted

class YellowDocks(Fisheries):
    def __init__(self, root="", download=0, split=None, 
                 transform_function=None, 
                 ratio=0.2, density=0, sigma=8.0, Time=False):

        lake = "yellow_docks_1"
        self.ratio = ratio

        self.sigma = sigma
        self.density = density
        self.split = split

        #corrupted = ["MDGC4877.JPG"]
        '''
          names = glob.glob(tmp_path + "/*.JPG")
          corrupted = find_corrupted(names )
          ms.save_pkl(tmp_path + "/corrupted.pkl", corrupted)
        '''
        tmp_path = "/mnt/home/issam/DatasetsPrivate/fisheries/yellow_docks_1/"
        corrupted = ms.load_pkl(tmp_path + "/corrupted.pkl")

        super(YellowDocks, self).__init__(root,
                 split=split,
                  lake=lake, 
                 transform_function=transform_function,
                 corrupted=corrupted)

        # GET CORRUPTED


class YellowDocks25(YellowDocks):
    def __init__(self, root="", download=0, split=None, 
                 transform_function=None, 
                 ratio=0.2, density=0, sigma=8.0, Time=False):

        super().__init__(root,
                 split=split,
                  transform_function=transform_function,
                  Time=Time)
        if split == "train":
          self.meta = self.meta[:int(len(self.meta)*0.25)]



class YellowDocks50(YellowDocks):
    def __init__(self, root="", download=0, split=None, 
                 transform_function=None, 
                 ratio=0.2, density=0, sigma=8.0, Time=False):

        super().__init__(root,
                 split=split,
                  transform_function=transform_function,
                  Time=Time)
        if split == "train":
          self.meta = self.meta[:int(len(self.meta)*0.50)]


class YellowDocks75(YellowDocks):
    def __init__(self, root="", download=0, split=None, 
                 transform_function=None, 
                 ratio=0.2, density=0, sigma=8.0, Time=False):

        super().__init__(root,
                 split=split,
                  transform_function=transform_function,
                  Time=Time)
        if split == "train":
          self.meta = self.meta[:int(len(self.meta)*0.75)]


 
class Lafrage(Fisheries):
    def __init__(self, root="", download=0, split=None, 
                 transform_function=None, 
                 ratio=0.2, density=0, sigma=8.0, Time=False):
        self.split = split
        self.ratio = ratio
        lake = "lafrage"
        
        self.sigma = sigma
        self.density = density


        corrupted = []    
        super(Lafrage, self).__init__(root,
                 split=split,
                  lake=lake, 
                 transform_function=transform_function,
                 corrupted=corrupted,Time=Time)

class RiceLake(Fisheries):
    def __init__(self, root="", download=0, split=None, 
                 transform_function=None, 
                 ratio=0.3, density=0, sigma=8.0, Time=False):
        self.split = split
        self.ratio = ratio
        lake = "Rice_lake"
        
 
 
        self.sigma = sigma
        self.density = density

        corrupted = []    
        super(RiceLake, self).__init__(root,
                 split=split,
                  lake=lake, 
                 transform_function=transform_function,
                 corrupted=corrupted,Time=Time)


class GreenTimbers(Fisheries):
    def __init__(self, root="", download=0, split=None, 
                 transform_function=None, 
                 ratio=0.2, density=0, sigma=8.0, Time=False):
        self.split = split
        self.ratio = ratio
        lake = "Green_Timbers"
        

        self.sigma = sigma
        self.density = density

        corrupted = []    
        super(GreenTimbers, self).__init__(root,
                 split=split,
                  lake=lake, 
                 transform_function=transform_function,
                 corrupted=corrupted,Time=Time)


class Chimney(Fisheries):
    def __init__(self, root="", download=0, split=None, 
                 transform_function=None, 
                 ratio=0.5, density=0, sigma=8.0, Time=False):
        self.split = split
        self.ratio = ratio
        lake = "Chimney"
        


        self.sigma = sigma
        self.density = density

        corrupted = []    
        super(Chimney, self).__init__(root,
                  split=split,
                  lake=lake, 
                 transform_function=transform_function,
                 corrupted=corrupted,Time=Time)


class Hastings(Fisheries):
    def __init__(self, root="", download=0, split=None, 
                 transform_function=None, 
                 ratio=0.3, density=0, sigma=8.0):
        self.split = split
        self.ratio = ratio
        



        self.sigma = sigma
        self.density = density

        corrupted = []    
        super(Hastings, self).__init__(root,
                  split=split,
                  lake="Hastings", 
                 transform_function=transform_function,
                 corrupted=corrupted)



class Kentucky(Fisheries):
    def __init__(self, root="", download=0, split=None, 
                 transform_function=None, 
                 ratio=0.3, density=0, sigma=8.0, Time=False):
        self.split = split
        self.ratio = ratio
        lake = "kentucky"
        

        self.sigma = sigma
        self.density = density
        
        corrupted = []    
        super(Kentucky, self).__init__(root,
                  split=split,
                  lake=lake, 
                 transform_function=transform_function,
                 corrupted=corrupted,Time=Time)