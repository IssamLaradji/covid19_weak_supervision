
import matplotlib
matplotlib.use('Agg')
import json
import torch
import numpy as np
import subprocess

import torch
import pylab as plt
import numpy as np
from scipy import sparse
from tqdm import tqdm 
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
import PIL
from sklearn.feature_extraction.image import img_to_graph, _compute_gradient_3d, _make_edges_3d
import shutil
#UTILS
from distutils.dir_util import copy_tree

def pad_image(img):
        padded_size = (int(np.ceil(img.shape[2]/8)*8), int(np.ceil(img.shape[3]/8)*8))

        p2d = (0, padded_size[1] - img.shape[3], 0, padded_size[0] - img.shape[2])
        img = F.pad(img, p2d)
        return img 

def assert_no_nans(A):
    assert torch.isnan(A).sum().item() == 0

def copy_code(main_dict, epoch=""):
    dest = main_dict["path_save"]+"/"+main_dict["project_name"] + "{}".format(epoch)
    result = copy_tree(main_dict["code_path"], dest) 
    print("Code copied in {}".format(dest))


def batch2image(batch):
    return f2l(t2n((denormalize(batch["images"])))).squeeze()

def copy_code_best(main_dict):
    dest = main_dict["path_save"]+"/"+main_dict["project_name"] + "_best"
    result = copy_tree(main_dict["code_path"], dest) 
    print("Code copied in {}".format(dest))

def get_module_classes(module_name):
  import inspect
  

  mod_dict = {}
  
  modList = import_module("{}.__init__".format(module_name)).__all__


  for module in modList:
    funcs = get_functions(module)
    for name in funcs:
      val = funcs[name]

      if not inspect.isclass(val):
        continue


      if (name in mod_dict and 
         module_name in str(val.__module__)):
         if name != "Pascal2012":
            raise ValueError("repeated %s" % name)
         print("Repeated:", name)
      mod_dict[name] = val

  return mod_dict

def get_batch(datasets, indices):
  return default_collate([datasets[i] for i in indices])

def argmax_mask(X, mask):
    ind_local = np.argmax(X[mask])

    G = np.ravel_multi_index(np.where(mask), mask.shape)
    Gi = np.unravel_index(G[ind_local], mask.shape)
    
    return Gi 

# def argmax_mask(X, mask):
#     ind = np.meshgrid(np.where(mask))
#     return np.argmax(X[ind])

# def up():
#     globals().update(locals())

def resizeMax(A, max_size=500):
    scale = float(max_size) / max(A.shape[-2:])
    if scale >= 1:
        return A
    return t2n(F.interpolate(torch.FloatTensor(A), size=tuple(map(int, np.array(A.shape[-2:])*scale)),
                        mode="bilinear",
                        align_corners=True))


def resizeTo(A, B):
    return F.interpolate(A, size=B.shape[-2:],
                        mode="bilinear",
                        align_corners=True)

def imsave(fname, arr):
    arr = f2l(t2n(arr)).squeeze()
    create_dirs(fname + "tmp")
    #print(arr.shape)
    scipy.misc.imsave(fname, arr)

def t2f(X):
    return Variable(torch.FloatTensor(X).cuda())

def t2l(X):
    return Variable(torch.LongTensor(X).cuda())
def get_size(model):
    total_size = 0
    for tensor in model.state_dict().values():
        total_size += tensor.numel() * tensor.element_size()
    return total_size / (1024.**3)

def ToPil(inputList):
    result = []

    for i in inputList:
        result += [transforms.functional.to_pil_image(i)]

    return result 

def point2mask(pointList, image, n_classes=None, return_count=False):
    h, w = np.asarray(image).shape[:2]
    points = np.zeros((h, w, 1), np.uint8)
    if return_count:
        counts = np.zeros(n_classes)

    for p in pointList: 
        if  int(p["x"]) > w or int(p["y"]) > h:
            continue
        else:
            points[int(p["y"]), int(p["x"])] = p["cls"]
            if return_count:
                counts[p["cls"]-1] += 1

    if return_count:
        return points, counts

    return points

def load_cp_val():
    from pycocotools.coco import COCO
    path_base = "/mnt/datasets/public/issam/Cityscapes/annList/"
    fname = "{}/val.json".format(path_base)
    
    cocoGt = COCO(fname)
    return cocoGt

def load_voc2012_val():
    from pycocotools.coco import COCO
    path_base = "/mnt/datasets/public/issam/VOCdevkit/annotations/"
    fname = "{}/instances_val2012.json".format(path_base)
    
    cocoGt = COCO(fname)
    return cocoGt

def load_annList(main_dict, predict_proposal="BestObjectness"):
    print("Get predicted proposals for {}".format(predict_proposal))
    path_base = "/mnt/datasets/public/issam/VOCdevkit/annotations/"
    fname = (path_base + "/results/"+ main_dict["exp_name"] 
          +"_"+predict_proposal+".json")

    return load_json(fname)

# def mask2
def eval_cocoDt(main_dict, predict_proposal="BestObjectness"):
    from pycocotools.cocoeval import COCOeval
    cocoGt = load_voc2012_val()
    print("Get predicted proposals for {}".format(predict_proposal))
    path_base = "/mnt/datasets/public/issam/VOCdevkit/annotations/"
    fname = (path_base + "/results/"+ main_dict["exp_name"] 
          +"_"+str(predict_proposal)+".json")
    
    cocoDt = cocoGt.loadRes(fname)
    cocoEval = COCOeval(cocoGt, cocoDt, "segm")
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    print("Images:", len(cocoEval.params.imgIds))
    print("Model: {}, Loss: {}, Pred: {}".format(main_dict["model_name"],
      main_dict["loss_name"], predict_proposal))


def dict2frame(myDict):
  if len(myDict) == 0:
    return None 

  df=pd.DataFrame()

  for key in myDict:
    row = key[0]
    col = key[1]

    df.loc[row, col] = myDict[key]

  return df



def mask2hot(mask, n_classes):
    mask = t2n(mask)
    n, h, w = mask.shape
    Hot = np.zeros((n_classes, n, h, w))

    for c in np.unique(mask):
        if c == 0:
            continue
        Hot[c, mask==c] = 1

    return Hot.transpose(1,0,2,3)


def label2hot(y, n_classes):
    n = y.shape[0]
    Y = np.zeros((n, n_classes))
    Y[np.arange(n), y] = 1

    return Y
    
def get_exp_name(dataset_name, config_name, main_dict, return_dict=False):
    name2id = {"metricList":"m"}

    keys2override = ["model_name","sampler_name",
                     "batch_size","opt_name","learning_rate","loss_name","weight_decay","epoch2val",
                     "iter2val", "epochs",
                     "dataset_options","metricList","model_options",
                     "trainTransformer","testTransformer",
                     "val_batchsize"]

    config = jload("configs.json")[config_name]
    config_args = parser_config.parse_config(config)
    config_dict = vars(config_args)

    exp_name = config_name + "-d:%s" % dataset_name

    value_dict = {}
    for key in keys2override:        
        if key in main_dict and main_dict[key] != None and main_dict[key] != config_dict[key]:
            value = main_dict[key]

            if isinstance(value, list):
                exp_name += "-%s:%s" % (name2id[key], value[0])
            elif key in ["epochs"]:
                pass
            else:
                exp_name += "-%s:%s" % (name2id[key], value)

        elif key in config_dict:
            value = config_dict[key]

        else:
            raise ValueError("%s does not exist..." % key) 

        value_dict[key] = value

    if return_dict:
        return exp_name, value_dict

    return exp_name
# import types
# def get_modules(module):
#     modules = {}

#     for name, val in module.__dict__.items():
#         if name in modules:
#           raise ValueError("Repeated module %s" % name) 
     
#         if isinstance(val, types.ModuleType):
#           modules[name] = val

    
#     return modules

    

def get_functions(module):
    if isinstance(module, str):
        spec = importlib.util.spec_from_file_location("module.name", module)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

    funcs = {}
    for name, val in module.__dict__.items():
      if name in funcs:
        raise ValueError("Repeated func %s" % name) 
     

      if callable(val):
         funcs[name] = val

      
    return funcs

    
def old2new(path):
    return path.replace("/mnt/AIDATA/home/issam.laradji", 
                        "/mnt/home/issam")
def logsumexp(vals, dim=None):
    m = torch.max(vals, dim)[0]

    if dim is None:
        return m + torch.log(torch.sum(torch.exp(vals - m), dim))
    else:
        return m + torch.log(torch.sum(torch.exp(vals - m.unsqueeze(dim)), dim))
        
def count2weight(counts):
    uni, freq = np.unique(counts, return_counts=True)
    myDict = {i:j for i,j in zip(uni, freq)}
    freq = np.vectorize(myDict.get)(counts)

    return 1./freq
    
def time_elapsed(s_time):
    return (time.time() - s_time) / 60
    
def get_longest_list(listOfLists):
    LL = listOfLists
    longest_list = []

    if LL is None:
        return longest_list

    for L in LL:
        if not isinstance(L, list):
            continue

        if not isinstance(L[0], list):
            L = [L]
        
        if len(L) > len(longest_list):
            longest_list = L

    #print(longest_list)
    return longest_list

def n2l(A):
    return Variable(torch.LongTensor(A).cuda())
    
def get_median_list(listOfLists):
    LL = listOfLists
    pointList = []
    lenList = []

    if LL is None:
        return pointList

    for L in LL:
        if not isinstance(L, list):
            continue

        if not isinstance(L[0], list):
            L = [L]
        
        
        pointList += [L]
        lenList += [len(L)]
    if len(pointList) == 0:
        return pointList
        
    i = np.argsort(lenList)[len(lenList)//2]
    return pointList[i]



def get_histogram(dataset):
    n = len(dataset)
    n_classes = t2n(dataset[0]["counts"]).size
    
    counts = np.zeros((n, n_classes))
    pbar = tqdm.tqdm(total=len(dataset), leave=False)
    for i in range(len(dataset)):
        counts[i] = t2n(dataset[i]["counts"])

        pbar.update(1)
    pbar.close()
    return counts

def count2stats(countStats):
    pass

def shrink2roi(img, roi):
    ind = np.where(roi != 0)

    y_min = min(ind[0])
    y_max = max(ind[0])

    x_min = min(ind[1])
    x_max = max(ind[1])

    return img[y_min:y_max, x_min:x_max]

def read_xml(fname):
    with open(fname) as f:
        xml = f.readlines()
        xml = ''.join([line.strip('\t') for line in xml])
        
        xml = BeautifulSoup(xml, "lxml")

    return xml

def getFileFunctions(fname):
    name_func_tuples = inspect.getmembers(fname, inspect.isfunction)
    name_func_tuples = [t for t in name_func_tuples if inspect.getmodule(t[1]) == fname]
    functions = dict(name_func_tuples)

    return functions 

def add2diag(A, eps=1e-6):
    n = A.size(0)
    if A.is_cuda:
        return A + Variable(torch.eye(n).cuda()*eps)
    else:
        return A + Variable(torch.eye(n)*eps)

def batch_tril(A):
    B = A.clone()
    ii,jj = np.triu_indices(B.size(-2), k=1, m=B.size(-1))
    B[...,ii,jj] = 0
    return B
    
def batch_diag(A):
    ii,jj = np.diag_indices(min(A.size(-2),A.size(-1)))
    return A[...,ii,jj]


def gray2rgb(x):
    x = t2n(x)
    if x.ndim == 2:
        x = x[:,:,None]

        x = x.repeat(3, 2)

    if x.ndim == 3:
        x = x[:,:,:,None]

        x = x.repeat(3, 3)

    return x

def unique(tensor, return_counts=0):
    return np.unique(t2n(tensor), return_counts=return_counts)

def read_text(fname):
    # READS LINES
    with open(fname, "r") as f:
        lines = f.readlines()
    return lines

def read_textraw(fname):
    with open(fname, "r") as f:
        lines = f.read()
    return lines   

def parse_command(command, parser):    
    if isinstance(command, list):
      command = " ".join(command)
      
    io_args = parser.parse_args(shlex.split(command))

    return io_args

def dict2dataframe(dicts, on):
    names = list(dicts.keys()) 
    trh = pd.DataFrame(dicts[names[0]])
    teh = pd.DataFrame(dicts[names[1]])
    df = pd.merge(trh, teh, on=on, how="outer", sort=on, suffixes=("_%s" % names[0],
                                             "_%s" % names[1]))

    return df
def extract_fname(directory):
    import ntpath
    return ntpath.basename(directory)


def dict2name(my_dict):
    new_dict = collections.OrderedDict(sorted(my_dict.items()))

    name = "_".join(map(str, list(new_dict.values())))

    return name


def gray2cmap(gray, cmap="jet", thresh=0):
    # Gray has values between 0 and 255 or 0 and 1
    gray = t2n(gray)
    gray = gray / gray.max()
    gray = np.maximum(gray - thresh, 0)
    gray = gray / gray.max()
    gray = gray * 255

    gray = gray.astype(int)
    #print(gray)
   
    from pylab import get_cmap
    cmap = get_cmap(cmap)

    output = np.zeros(gray.shape + (3,), dtype=np.float64)

    for c in np.unique(gray):
        output[(gray==c).nonzero()] = cmap(c)[:3]

    return l2f(output)


import PIL

def n2p(img):
    im = PIL.Image.fromarray(np.uint8(img*255))

    return im

def get_counts():
    pass

def create_dirs(fname):
    if "/" not in fname:
        return
        
    if not os.path.exists(os.path.dirname(fname)):
        try:
            os.makedirs(os.path.dirname(fname))
        except OSError:
            pass 
            
            
def save_pkl(fname, dict):
    create_dirs(fname)
    with open(fname, "wb") as f: 
        pickle.dump(dict, f)

def jload(fname):
    with open(fname) as data_file:
        return json.loads(data_file.read())

def load_pkl(fname):
    with open(fname, "rb") as f:        
        return pickle.load(f)


def label2Image(imgs):
    imgs = t2n(imgs).copy()

    if imgs.ndim == 3:
        imgs = imgs[:, np.newaxis]

    imgs = l2f(imgs)

    if imgs.ndim == 4 and imgs.shape[1] != 1:
        imgs = np.argmax(imgs, 1)

    imgs = label2rgb(imgs)

    if imgs.ndim == 3:
        imgs = imgs[np.newaxis]
    return imgs

def run_bash_command(command, noSplit=True):
    if noSplit:
        command = command.split()
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    output, error = process.communicate()

    return str(output)

def run_bash(fname, arg1):
    return subprocess.check_call([fname, arg1])

# def label2Image(imgs, win="8888", nrow=4):
#     # If given a single image
#     imgs = t2n(imgs).copy()

#     # Label image case
#     if imgs.ndim == 2:
#         imgs = mask2label(imgs)
#         imgs = l2f(imgs)

#     # Prediction output case
#     if imgs.ndim == 4:
#         imgs = np.argmax(imgs, 1)
    
#     imgs = label2rgb(imgs, np.max(np.unique(imgs)) + 1)


#     return imgs

def create_dirs(fname):
    if "/" not in fname:
        return
        
    if not os.path.exists(os.path.dirname(fname)):
        try:
            os.makedirs(os.path.dirname(fname))
        except OSError:
            pass 
            
def stack(imgList):
    imgListNumpy = []
    for img in imgList:
        new_img = l2f(t2n(img)).copy()
        if new_img.max() > 1:
            new_img = new_img / 255.

        imgListNumpy += [new_img]

    return np.vstack(imgListNumpy)

def maskOnImage(imgs, mask, enlarge=0):
    imgs = l2f(t2n(imgs)).copy()
    mask = label2Image(mask)
    
    if enlarge:
        mask = zoom(mask, 11)

    if mask.max() > 1:
        mask = mask / 255.

    if imgs.max() > 1:
        imgs = imgs / 255.
    
    nz = mask.squeeze() != 0 
    imgs = imgs*0.5 + mask * 0.5
    imgs /= imgs.max()
    # print(mask.max(), imgs.shape, mask.shape)
    # ind = np.where(nz)
   
    # if len(ind) == 3:
    #     k, r, c = ind
    #     imgs[:,k,r,c] = imgs[:,k,r,c]*0.5 + mask[:,k,r,c] * 0.5
    #     imgs[:,k,r,c]  = imgs[:,k,r,c]/imgs[:,k,r,c].max()

    # if len(ind) == 2:
    #     r, c = ind
    #     imgs[:,:,r,c] = imgs[:,:,r,c]*0.5 + mask[:,:,r,c] * 0.5
    #     imgs[:,:,r,c]  = imgs[:,:,r,c]/imgs[:,:,r,c].max()

    #print(imgs[nz])
    #print(imgs.shape)
    #print(mask.shape)
    if mask.ndim == 4:
        mask = mask.sum(1)

    nz = mask != 0
    mask[nz] = 1

    mask = mask.astype(int)

    #imgs = imgs*0.5 + mask[:, :, :, np.newaxis] * 0.5

    segList = []
    for i in range(imgs.shape[0]):
        segList += [l2f(mark_boundaries(f2l(imgs[i]).copy(), f2l(mask[i]),mode="inner"))]
        # segList += [l2f(imgs[i]).copy()]
    imgs = np.stack(segList)

    return l2f(imgs)

def labelrgb2label(labels):
    gray_label = np.zeros((labels.shape[0], labels.shape[1]), dtype=np.uint8)

    rgbs = {(0,0,0):0}
    c_id = 1
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            c = tuple(labels[i,j])
            if c not in rgbs:
                rgbs[c] = c_id
                c_id += 1
            

    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            c = tuple(labels[i,j])
            gray_label[i, j] = rgbs[c]
    

    return gray_label



def rgb2label(img, n_classes, void_class=-1):
    rgb = img.copy()

    label = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint8)

    classes = np.arange(n_classes).tolist()

    # if void is not None:
    #     N = max(n_classes, void) + 1
    #     classes += [void]
    # else:
    N = n_classes + 1

    colors = color_map(N=N)

    for c in classes:
        label[np.where(np.all(rgb == colors[c], axis=-1))[:2]] = c

    # label[np.where(np.all(rgb == colors[c], axis=-1))[:2]] = c
    
    return label

def label2rgb(labels, bglabel=None, bg_color=(0., 0., 0.)):
    labels = np.squeeze(labels)
    colors = color_map(np.max(np.unique(labels)) + 1)
    output = np.zeros(labels.shape + (3,), dtype=np.float64)

    for i in range(len(colors)):
        if i != bglabel:
            output[(labels == i).nonzero()] = colors[i]

    if bglabel is not None:
        output[(labels == bglabel).nonzero()] = bg_color

    return l2f(output)

def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

def zoom(img,kernel_size=3):
    img = n2t(img)
    if img.dim() == 4:
        img = img.sum(1).unsqueeze(1)
    img = Variable(n2t(img)).float()
    img = F.max_pool2d(img, kernel_size=kernel_size, stride=1, 
                       padding=get_padding(kernel_size))
    return t2n(img)

def numpy2seq(Z, val=-1):
    """Appends the minimal required amount of zeroes at the end of each 
     array in the jagged array `M`, such that `M` looses its jagedness."""
    seq = []
    for z in t2n(Z).astype(int):
        i = np.where(z==val)[0]
        if i.size == 0:
            seq += [z.tolist()]
        else:
            seq += [z[:min(i)].tolist()]
        
    return seq

def seq2numpy(M, val=-1, maxlen=None):
    """Appends the minimal required amount of zeroes at the end of each 
     array in the jagged array `M`, such that `M` looses its jagedness."""
    if maxlen is None:
        maxlen = max(len(r) for r in M)

    Z = np.ones((len(M), maxlen)) * val
    for i, row in enumerate(M):
        Z[i, :len(row)] = row 
        
    return Z

def get_padding(kernel_size=1):
    return int((kernel_size - 1) / 2)

# MISC
def remove_dir(dir_name):
    import shutil
    shutil.rmtree(dir_name)

def dict2str(score):
    string = ""
    for k in score:
        string += "- %s - %.3f" % (k, score[k])
    return string[2:]

def save_csv(fname, df):
    create_dirs(fname)
    df.to_csv(fname, index=False)

def save_json(fname, data):
    create_dirs(fname)
    with open(fname, "w") as json_file:
        json.dump(data, json_file, indent=4, sort_keys=True)
    
def load_json(fname, decode=None):

    with open(fname, "r") as json_file:
        d = json.load(json_file)
        
    return d

def print_box(*strings):
    string_format = ["{%d:10s}" % i for i in range(len(strings))]
    

    sizes = [len(i) for i in strings]
    bars = ["-"*s for s in sizes]
    print("\n")
    print(" ".join(string_format).format(*bars))
    print(" ".join(string_format).format(*strings))
    print(" ".join(string_format).format(*bars))

def print_header(*strings):
    string_format = ["{%d:10s}" % i for i in range(len(strings))]
    print("\n"+" ".join(string_format).format(*strings))

    sizes = [len(i) for i in strings]
    bars = ["-"*s for s in sizes]
    print(" ".join(string_format).format(*bars))

def pprint2(*strings):
    string_format = ["{%d:10s}" % i for i in range(len(strings))]
    #string_format[0] = "{%d:5s}"
    strings = [str(s) for s in strings]
    print(" ".join(string_format).format(*strings))


def f2l(X):
    if X.ndim == 3 and (X.shape[2] == 3 or X.shape[2] == 1):
        return X
    if X.ndim == 4 and (X.shape[3] == 3 or X.shape[3] == 1):
        return X

    # CHANNELS FIRST
    if X.ndim == 3:
        return np.transpose(X, (1,2,0))
    if X.ndim == 4:
        return np.transpose(X, (0,2,3,1))

    return X

def l2f(X):
    if X.ndim == 3 and (X.shape[0] == 3 or X.shape[0] == 1):
        return X
    if X.ndim == 4 and (X.shape[1] == 3 or X.shape[1] == 1):
        return X

    if X.ndim == 4 and (X.shape[1] < X.shape[3]):
        return X

    # CHANNELS LAST
    if X.ndim == 3:
        return np.transpose(X, (2,0,1))
    if X.ndim == 4:
        return np.transpose(X, (0,3,1,2))

    return X


def stack_images(images):
    for img in images:
        import ipdb; ipdb.set_trace()  # breakpoint f1a9702d //
        

def t2n(x):
    if isinstance(x, (int, float)):
        return x
    if isinstance(x, torch.autograd.Variable):
        x = x.cpu().data.numpy()
        
    if isinstance(x, (torch.cuda.FloatTensor, torch.cuda.IntTensor, torch.cuda.LongTensor, torch.cuda.DoubleTensor )):
        x = x.cpu().numpy()

    if isinstance(x, (torch.FloatTensor, torch.IntTensor, torch.LongTensor, torch.DoubleTensor )):
        x = x.numpy()

    return x

def n2t(x, dtype="float"):
    if isinstance(x, (int, np.int64, float)):
        x = np.array([x])

    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x

def n2v(x, dtype="float", cuda=True):
    if isinstance(x, (int, np.int64, float)):
        x = np.array([x])

    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
      

    if isinstance(x, Variable):
        return x 

    if cuda:
        x = x.cuda()

    return Variable(x).float()


def set_gpu(gpu_id):
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"]="%d" % gpu_id


def print_config(configs):
    print("\n")
    pprint2("dataset: %s" % configs["dataset"], "model: %s" % configs["model"], 
              "optimizer: %s" % configs["opt"])
    print("\n")


def zscale(X, mu=None, var=None, with_stats=False):
    if mu is None:
        mu = X.mean(0)
    if var is None:
        var = X.var(0)
    Xs =  (X - mu) / var
  
    if with_stats:
        return Xs, mu, var

    else:
        return Xs
#### TRAINERS



import scipy.misc
import scipy.io as io
import os 

def imread(fname):
    return scipy.misc.imread(fname)


def loadmat(fname):
    return io.loadmat(fname)

def count_files(dir):
    list = os.listdir(dir) 
    return len(list)

def f2n( fig ):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()
      
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return data

def create_latex(fname, title, desc, sectionList, figList):
    template=("\documentclass[12pt,a4paper]{article} % din a4",
        ", 11 pt, one sided\n\n",
    "\begin{document}\n",
    "\VAR{sec}\n",
    "\VAR{fig}\n")

    for i in range(len(sectionList)):
        template += "\n%s\n" % sectionList[i]
        template += "\n%s\n" % create_latex_fig(figList[i])

    template += "\end{document}"

    save_txt(fname, template)

def save_txt(fname, string):
    with open(fname, "w") as f:
        f.write(string)


def create_latex_fig(fname, img):
   
    imsave(fname, img)

    fig = ("\begin{figure}\n",
    "\centering\n", 
    "\includegraphics[width=4in]{%s}\n",
    "\end{figure}\n" % (fname))

    return fig


def create_latex_table(fname, df):

    fig = ("\begin{figure}\n",
    "\centering\n", 
    "\includegraphics[width=4in]{%s}\n",
    "\end{figure}\n" % (fname))

    return fig



# VALIDATE
@torch.no_grad()
def valBatch(model, batch, metric_class=None):    
    model.eval()
    # with torch.no_grad():
    metricObject = metric_class()
    score_dict = metricObject.scoreBatch(model, batch)

    return score_dict["score"]


@torch.no_grad()
def validate(model, dataset,  
             metric_class,
             batch_size=1, epoch=0, 
             verbose=1,
             num_workers=1,
             sampler=None):
    batch_size = min(batch_size, len(dataset))

    if sampler is None:
      loader = data.DataLoader(dataset, 
                               batch_size=batch_size, 
                               num_workers=num_workers, 
                               drop_last=False)
    else:
      loader = data.DataLoader(dataset, batch_size=batch_size, 
                               num_workers=num_workers, 
                               drop_last=False,
                               sampler=sampler)

    return val(model, loader, metric_class, epoch=epoch, 
             verbose=verbose)

def val(model, loader,  metric_class, epoch=0, 
              verbose=1):
  metric_name = metric_class.__name__

  model.eval()

  split_name = loader.dataset.split
  
  n_batches = len(loader)

  if verbose==2:
    pbar = tqdm(desc="Validating %s set (%d samples)" % 
                (split_name, n_batches), total=n_batches, leave=False)
  elif verbose==1:
    print("Validating... %d" % len(loader.dataset))

  metricObject = metric_class()
  
  #iter2dis = n_batches // min(10, n_batches)
  iter2dis = 1
  for i, batch in enumerate(loader):
    # print(i)
    metricObject.update_running_average(model, batch)

    #######    
    progress = ("%d - %d/%d - Validating %s set - %s: %.3f" % 
               (epoch, i, n_batches, split_name, metric_name, 
                metricObject.get_running_average()))

    if verbose==2:
      pbar.set_description(progress)
      pbar.update(1)

    elif verbose==1 and i % iter2dis == 0:
      print(progress)

  if verbose==2:
    pbar.close()
  
  score = metricObject.get_running_average()

  score_dict = {}
  score_dict[metric_name] = score
  score_dict["n_samples"] = len(loader.dataset)
  score_dict["epoch"] = epoch


  # Print to screen
  if verbose:
    pprint2("%d - %s" % (epoch, split_name), dict2str(score_dict))
  
  score_dict["split_name"] = split_name
  return score_dict











def get_preds(model, dataset,  
             batch_size=1, epoch=0, 
             verbose=1,
             num_workers=1,
             sampler_name=None):
            
    model.eval()

    split_name = dataset.split
    batch_size = min(batch_size, len(dataset))
    
    if sampler_name is None:
      loader = data.DataLoader(dataset, batch_size=batch_size, 
                               num_workers=num_workers, 
                               drop_last=False)
    else:
      sampler = SAMPLER_DICT[sampler_name](dataset)
      loader = data.DataLoader(dataset, batch_size=batch_size, 
                               num_workers=num_workers, 
                               drop_last=False,
                               sampler=sampler)

    n_batches = len(loader)

    if verbose==1:
      pbar = tqdm(desc="Validating %s set (%d samples)" % 
                  (split_name, n_batches), total=n_batches, leave=False)
    else:
      print("Validating... %d" % len(dataset))

    
    iter2dis = n_batches // min(10, n_batches)
    preds = np.ones(len(dataset))*-1
    counts = np.ones(len(dataset))*-1

    for i, batch in enumerate(loader):
      preds[i*batch_size:(i+1)*batch_size] = t2n(model.predict(batch, "counts")).ravel()
      counts[i*batch_size:(i+1)*batch_size] = t2n(batch["counts"]).ravel()

      #######
      progress = ("%d - %d/%d - Validating %s set" % 
                 (epoch, i, n_batches, split_name))

      if verbose==1:
        pbar.set_description(progress)
        pbar.update(1)

      elif i % iter2dis == 0:
        print(progress)

    if verbose==1:
      pbar.close()
    

    score_dict = {}
    score_dict["preds"] = preds
    score_dict["counts"] = counts
    score_dict["n_samples"] = len(dataset)
    score_dict["epoch"] = epoch


    # Print to screen
    pprint2("%d - %s" % (epoch, split_name), dict2str(score_dict))
    
    score_dict["split_name"] = split_name
    return score_dict
from torch.utils.data.sampler import SubsetRandomSampler

def validate_stats(model, dataset, verbose=1, metric_class=None, predictFunc=None):
    model.eval()
    
    loader = data.DataLoader(dataset, batch_size=1, 
                             num_workers=1, drop_last=False)

    n_batches = len(loader)

    if verbose==1:
      pbar = tqdm(desc="Validating Test set (%d samples)" % 
                  (n_batches), total=n_batches, leave=False)

    metricObject = metric_class()
    metric_name = metric_class.__name__

    Corrects = []
    Wrongs = []
    scoreList = []
    for i, batch in enumerate(loader):
      
      score_dict = metricObject.update_running_average(model, batch, predictFunc)
      score = score_dict
      scoreList += [score]
      if score == 0:
        Corrects += [i]

      else:
        Wrongs += [i]

      progress = ("%d/%d - Validating Test set - %s: %.3f" % 
                 (i, n_batches, metric_name, 
                  metricObject.get_running_average()))

      if verbose==1:
        pbar.set_description(progress)
        pbar.update(1)

      elif verbose == 2:
        print(progress)

    if verbose==1:
      pbar.close()

    scores = np.array(scoreList)

    return {"score":metricObject.get_running_average(),
            "metric_name":metric_name, 
            "Corrects":Corrects, "Wrongs":Wrongs,
            "max_score":scores.max(), "min_score":scores.min(), 
            "mean_score":scores.mean(),
            "n_corrects":len(Corrects), "n_wrongs":len(Wrongs)}




# SCORERS
class AvgMeter:
    def __init__(self):
        self.dict = {}

    def __repr__(self):
        return self.get_string()

    def update(self, name, score, batch_size=None):
        if name not in self.dict:
          self.dict[name] = 0
          self.dict[name + "_n"] = 0

        if batch_size is None:
          batch_size = 1

        self.dict[name] += score
        self.dict[name + "_n"] += batch_size

    def get_dict(self):

      metricList = [m for m in self.dict if m[-2:] != "_n"]

      score = {}
      for m in metricList:
        num = self.dict[m]
        denom = self.dict[m + "_n"]

        if isinstance(num, np.ndarray):
          nz = denom != 0
          mscore = nastype(np.float)

          mscore[nz] = mscore[nz] / denom[nz].astype(float) 

          score[m] = (mscore[nz].sum() / nz.sum())
          
        else:
          score[m] = num / denom

      return score

    def get_string(self):
        score_dict = self.get_dict()

        return dict2str(score_dict)

def dict2str(score_dict):
  string = ""
  for s in score_dict:
      string += " - %s: %.3f" % (s, score_dict[s])

  return string[3:]


# TRAINERS

def fit(model, trainloader, opt, loss_function, 
        iter2dis=None, verbose=1, epoch=0, val_batch=True):
    
    n_samples = len(trainloader.dataset)
    n_batches = len(trainloader) 

    if iter2dis is None:
      iter2dis = n_batches // min(10, n_batches)
  
    elif verbose==1:
      print("Training Epoch {} .... {} batches".format(epoch, n_batches))

    assert trainloader.dataset.split == "train"
    # %%%%%%%%%%% 1. Train Phase %%%%%%%%%%%%"
    s_time = time.time()

    avg_meter = AvgMeter()
    example = None
    for i, batch in enumerate(trainloader):
        model.trained_batch_names.add(batch["name"][0])
        # if "maskObjects" in batch and type(trainloader.dataset).__name__ != "PascalSmall":

        #     assert batch["maskObjects"].sum().item() == 0
        #     assert batch["maskClasses"].sum().item() == 0


        # 1. Update
        opt.zero_grad()
        

        loss = loss_function(model, batch)
        if loss != 0.:
            loss.backward()
            opt.step()

            loss_example = {batch["index"][0]: loss.item()}
            
            # 3. Details
            avg_meter.update(name=loss_function.__name__, score=loss.item())

        if verbose==1 and (i % iter2dis) == 0:
            elapsed = ((time.time() - s_time) / 60)
            print("{} - ({}/{}) - {} - {} - elapsed: {:.3f}".format(epoch,  i, n_batches, 
                trainloader.dataset.split, avg_meter, elapsed))


    if verbose:
      pprint2("{}. train".format(epoch), avg_meter, 
            "n_samples: {}".format(n_samples), 
              "n_batches: {}".format(n_batches))

    

    # train: save history
    train_dict = avg_meter.get_dict()
    train_dict["epoch"] = epoch
    train_dict["n_samples"] = n_samples
    train_dict["time (min)"] = elapsed
    train_dict["iterations"] = n_batches
    train_dict["loss_example"] = loss_example
 
    return train_dict
    
def fitQuick(model, train_set, loss_name, 
        metric_class, opt=None, num_workers=1, batch_size=1, 
        verbose=1, epochs=10, n_samples=1000):

  if opt is None:
      opt = torch.optim.Adam(model.parameters(), lr=1e-3)

  ind = np.random.randint(0, len(train_set), min(n_samples, len(train_set)))
  trainloader = data.DataLoader(train_set, 
                              num_workers=num_workers,
                              batch_size=batch_size, 
                              sampler=SubsetRandomSampler(ind))
  for e in range(epochs):
    fit(model, trainloader, opt, loss_name, 
            metric_class, verbose=verbose, epoch=e)



def fitIndices(model, train_set, opt, loss_name, 
        metric_name, num_workers, batch_size, 
        verbose=1, epoch=0, ind=None):
  trainloader = data.DataLoader(train_set, 
                              num_workers=num_workers,
                              batch_size=batch_size, 
                              sampler=SubsetRandomSampler(ind))
  
  return fit(model, trainloader, opt, loss_name, 
          metric_name, verbose=verbose, epoch=epoch)
# def fitIndices(model, dataset, loss_function, indices, opt=None, epochs=10,  
#                verbose=1):
#     if opt is None:
#       opt = torch.optim.Adam(model.parameters(), lr=1e-5)

#     for epoch in range(epochs):
#       if verbose == 1:
#         pbar = tqdm(total=len(indices), leave=True)

#       lossSum = 0.
#       for i, ind in enumerate(indices):
#         batch = get_batch(dataset, [ind])

#         opt.zero_grad()
#         loss = loss_function(model, batch)       
#         loss.backward()
#         opt.step()

#         lossSum += float(loss)
#         lossMean = lossSum / (i + 1)

#         if verbose == 1:
#           pbar.set_description("{} - loss: {:.3f}".format(epoch, lossMean))
#           pbar.update(1)

#         elif verbose == 2:
#           print("{} - ind:{} - loss: {:.3f}".format(epoch, ind, lossMean))

#       if verbose == 1:
#         pbar.close()
import math
def fitBatch(model, batch, loss_function=None, opt=None, 
             loss_scale="linear", epochs=10, verbose=2, visualize=False):

  loss_name = loss_function.__name__
  model_name = type(model).__name__
  if verbose == 1:
    pbar = tqdm.tqdm(total=epochs, leave=False)
  if opt is None:
      opt = torch.optim.Adam(model.parameters(), lr=1e-5)

  for i in range(epochs):           
      #train_set.evaluate_count(model, batch)
      # 1. UPDATE MODEL
      opt.zero_grad()

      loss = loss_function(model, batch, visualize=visualize) 
      if loss != 0.:

        loss.backward()
        opt.step()
      

      loss_value = float(loss)
      if loss_scale == "log":
        loss_value = math.log(loss_value)

      if verbose == 1:
        pbar.set_description("{}: {:.3f}".format(loss_name, loss_value))
        pbar.update(1)


      elif verbose == 2:
          print("{} - {} - {}: {:.3f}".
            format(i, model_name, loss_name, loss_value))

  if verbose == 1:
    pbar.close()
  print("{} - {} - {}: {:.3f}".format(i, 
    model_name, loss_name, loss_value))

def fitBatchList(model, batchList, opt, name="", 
    verbose=True):
  
  lossSum = 0.

  if verbose:
    pbar = tqdm(total=len(batchList), leave=False)

  for i in range(len(batchList)):  
      batch = batchList[i]
      #train_set.evaluate_count(model, batch)
      # 1. UPDATE MODEL
      opt.zero_grad()
      loss = model.compute_loss(batch)                
      loss.backward()
      opt.step()

      lossSum += float(loss)
      lossMean = lossSum / (i+1)
      if verbose:
        if name != "":
          pbar.set_description("{} - loss: {:.3f}".format(name, lossMean))
        else:
          pbar.set_description("loss: {:.3f}".format(lossMean))

        pbar.update(1)
      #print("{} - loss: {:.3f}".format(i, float(loss)))

  if verbose:
    pbar.close()

    if len(batchList) > 0:
      if name != "":
        print("{} - loss: {:.3f}".format(name, lossMean))
      else:
        print("loss: {:.3f}".format(lossMean))

      

    else:
      print("{} batch is empty...".format(name))

  if len(batchList) > 0:
    return lossMean
  

def fitData(model, dataset, opt=None, loss_function=None, epochs=10, batch_size=1):
    loader = data.DataLoader(dataset, batch_size=batch_size, 
                             num_workers=min(batch_size, 3), 
                             shuffle=True, drop_last=True)

    n_batches = len(loader)

    for epoch in range(epochs):
      pbar = tqdm.tqdm(total=n_batches, leave=False)

      lossSum = 0.
      for i, batch in enumerate(loader):
        opt.zero_grad()
        loss = loss_function(model, batch)
        loss.backward()
        opt.step()

        lossSum += float(loss)
        lossMean = lossSum / (i + 1)

        pbar.set_description("{} - loss: {:.3f}".format(epoch, lossMean))
        pbar.update(1)

      pbar.close()

      print("{} - loss: {:.3f}".format(epoch, lossMean))



# Visualize

def visBoxes(image, boxes, filter_size=10, select_label=None):

    points = np.zeros(image.shape[:2]).astype(int)

    label = 0
    for i, b in enumerate(boxes):

        # excluding regions smaller than 2000 pixels
        if b['size'] < filter_size:
            continue
        # distorted rects
        x, y, w, h = b['rect']
        if h == 0 or w == 0:
            continue
        if (w // h) > 1.2 or (h // w) > 1.2:
            continue

        bb = b["rect"]

        x, y, w, h = bb[0], bb[1], bb[2], bb[3]

        if ((select_label is None) or 
            (select_label is not None and label == select_label)): 

            points[y, x:x + w] = label
            points[y:y+h, x] = label

            points[y+h, x:x + w] = label
            points[y:y+h, x + w] = label

        label += 1

    images(image, points)

def visBlobsQ(model, dataset, ind=None):
    if ind is None:
        ind = [np.random.randint(0, len(dataset))]
    else:
        ind = [ind]
    batch = get_batch(dataset, ind)
    visBlobs(model, batch)

    return batch



def visDensity(img, density, p=0.5, win="9999"):
    img = t2n(denormalize(img))
    density = t2n(density)
    images(p*img + (1-p) * gray2cmap(density), win=win+"_true")




def visDensityBatch(model, batch, img_index=0, p=0.5, win="9999", fname=None):
    img = t2n(batch["images"][img_index])

    density = model.predict(batch, "density")[img_index,0]
    images(p*img + (1-p) * gray2cmap(density), denorm=1, win=win+"_pred")

    density = batch["density"][img_index]
    images(p*img + (1-p) * gray2cmap(density), denorm=1, win=win+"_true")

def density(img, sigma=0.8):
    return gray2cmap(gaussian_filter(t2n(img).astype(float), sigma))

def toDensity(img, win="tmp"):
    # side = np.linspace(-2,2,15)
    # X,Y = np.meshgrid(side,side)
    # Z = np.exp(-((X-1)**2+Y**2))
    images(gray2cmap(img), win=win, env='main', title=win) 

def visFigure(fig, win="tmp"):
    import visdom
    
    fig.savefig("tmp.jpg")
    img = l2f(imread("tmp.jpg"))
    print(img.shape)
    vis = visdom.Visdom(port=1111)
    options = dict(title=win)
    images(img, win=win, env='main', title=win) 
    plt.close()

def visBlobsPdf(model, stats=None, dataset=None,
            metric_class=None, fname=""):
    #for img in 
    for cat in ["Corrects", "Wrongs"]:
        count = 0
        for i in stats[cat]: 
            if count > 5:
                break
            count += 1

            batch = get_batch(dataset, indices=[i])
            image = batch["images"].clone()
            pred = model.predict(batch, metric="blobs")

            sec ="%s_%d" % (cat, i)
            fig_dict = visBlobs(model, batch, win=sec)
            figList +=[fig_dict["fig"]]
            sectionList += [sec + " - %s" % fig_dict["de s"] ]

    create_latex(fname, title="", desc="", sectionList=sectionList, figList=figList)
    
import copy
# VISUALIZE

def get_tp_fp_blobs(image, blobs, points):
    tps, tp_counts = np.unique(t2n(points) * blobs, 
                     return_counts=True)
    ind = tps!=0
    tps = tps[ind]
    tp_counts = tp_counts[ind]

    tps_1 = tps[tp_counts==1]
    tps_more = tps[tp_counts!=1]

    fps = np.setdiff1d(np.unique(blobs[blobs!=0]), tps)
    fps = fps[fps!=0]

    if tps_1.size > 0:
        tp_func = np.vectorize(lambda t: t in tps_1)
        tp_blobs = tp_func(blobs).astype(int)
        tp_blobs[tp_blobs!=0] = 2
    else:
        tp_blobs = np.zeros(blobs.shape).astype(int)

    fp_func = np.vectorize(lambda t: t in fps)
    fp_blobs = fp_func(blobs).astype(int)
    fp_blobs[fp_blobs!=0] = 1

    if tps_more.size > 0:
        tp2_func = np.vectorize(lambda t: t in tps_more)
        tp2_blobs = tp2_func(blobs).astype(int)
        tp2_blobs[tp2_blobs!=0] = 3
    else:
        tp2_blobs = np.zeros(fp_blobs.shape).astype(int)

    tp_fp = get_image(image, 
                mask=tp_blobs + fp_blobs + tp2_blobs)

    return tp_fp


def visHeat(model, batch, win="9999", label=0,  
             enlarge=0, return_dict=False):

    batch = copy.deepcopy(batch)
    image = batch["images"]
    pred = t2n(model.predict(batch, metric="labels"))


    # for blobcounter
    blobs = get_blobs(pred == 1)
    points = (batch["points"] == 1).long()
    probs = model.predict(batch, metric="probs")

    img = t2n(denormalize(image))
    density = t2n(probs[0, 1]) 
    density /= density.max()
    p = 0.5
    images(p*img*255 + (1-p) * gray2cmap(density)*255, win=win+"_true")







def print_deviation(model, batch):
    print("true unique %s" % (unique(batch["points"] - 1)))
    print("pred unique %s" % (unique(pred_mask) - 1))
    if "index" in batch:
        print("win: %s - pred: %s - true: %s  diff: %.3f Index: %s" %
                 (win,pc,tc, abs(tc-pc).sum(), str(batch["index"][0]) ))
    else:
        print("win: %s - pred: %s - true: %s  diff: %.3f" %
                 (win,pc,tc, abs(tc-pc).sum() ))

    print ("MAE: %.3f" % ( abs(t2n(model.predict(batch, metric="count")) - 
                           t2n(batch["counts"]))).sum())

def save_images(fname, imgs):
    create_dirs(fname)
    images = f2l(t2n(imgs))
    for i, img in enumerate(images):
        imsave(fname.replace(".png","") + "_%d.png" % i, img)

def visBatchPoints(batch, win="9999", fname=None):
    image = batch["images"].clone()
    org = denormalize(image)

    org = get_image(org, mask=batch["points"], label=False, 
                    enlarge=1, win=win)

    if fname is not None:
        save_images(fname, org)
    else:
        import visdom
        vis = visdom.Visdom(port=1111)

        options = dict(title=win, xtick=True, ytick=True)
        vis.images(org, opts=options, win=win, env='main')

def visBatchLabels(batch, win="9999", fname=None):
    image = batch["images"].clone()
    org = denormalize(image)

    org = get_image(org, mask=batch["labels"], win=win)

    if fname is not None:
        save_images(fname, org)
    else:
        import visdom
        vis = visdom.Visdom(port=1111)

        options = dict(title=win, xtick=True, ytick=True)
        vis.images(org, opts=options, win=win, env='main')


def denormalize(img):
    _img = t2n(img)
    _img = _img.copy()
    # (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    if _img.ndim == 3:
        _img[0] = _img[0] * 0.229 + 0.485
        _img[1] = _img[1] * 0.224 + 0.456
        _img[2] = _img[2] * 0.225 + 0.406
    else:
        _img[:,0] = _img[:,0] * 0.229 + 0.485
        _img[:,1] = _img[:,1] * 0.224 + 0.456
        _img[:,2] = _img[:,2] * 0.225 + 0.406   

    return _img

# def visBatch(model, ref_batch, win="batch", env="maine"):
#     original = ref_batch["images_original"][:6]
#     preds = label2Image(model.predict(ref_batch))[:6]
#     GT = label2Image(ref_batch["labels"])[:6]

#     stacked = stack([original, preds, GT])
#     images(stacked, nrow=original.size(0), win=win, env=env)


def saveVisBatch(fname, model, ref_batch, nrows=3):
    original = ref_batch["images_original"][:6]
    preds = label2Image(pf.predict(model, ref_batch, rule="image2mask"))[:6]
    GT = label2Image(ref_batch["labels"])[:6]
    
    ncols = original.shape[0]
    stacked = stack([original, preds, GT])

    i = 0
    fig, axList = plt.subplots(nrows=nrows, ncols=ncols,
                   figsize=(12,9))
    fig.suptitle(extract_fname(fname).replace(".png", ""))
    for r in range(nrows):
        for c in range(ncols):
            axList[i].imshow(f2l(stacked[r*ncols + c]))

            i += 1

    fig.savefig(fname)
    plt.close()
    
def plotHistory(history, title="", line_name="", win="main",
                env="main"):

    import visdom
    vis = visdom.Visdom(port=1111)
    
    df = pd.DataFrame(history)
    epochs = np.array(df["epoch"])

    del df["epoch"]
    for c in df.columns:
        Y = np.array(df[c])
        plot(Y=Y, X=epochs, line_name=line_name, ylabel=c, xlabel="epochs", 
             title=title, win="%s_%s" % (c, win), env=env, vis=vis)


def close(win="main", env="main"):
    import visdom

    vis = visdom.Visdom(port=1111)
    vis.close(win, env=env)

def text(text, win="main", env="main"):
    import visdom

    vis = visdom.Visdom(port=1111)
    vis.text(text=text, win=win, env=env) 


def plot(Y, X, line_name="", ylabel="", xlabel="", title="", 
         win="main", env="main"):
    import visdom

    vis = visdom.Visdom(port=1111)
    if not isinstance(Y, (list, np.ndarray)):
        Y = [Y]

    if not isinstance(X, (list, np.ndarray)):
        X = [X]

    if isinstance(Y, list):
        Y = np.array(Y)
    if isinstance(X, list):
        X = np.array(X)   

    msg = vis.updateTrace(Y=Y, X=X, name=line_name, env=env, win=win, 
                          append=True)

    if msg == 'win does not exist':
       options = dict(title=title , xlabel=xlabel, 
                      ylabel=ylabel, legend=[line_name])

       vis.line(X=X, Y=Y , opts=options, win=win, env=env) 


def visInd(dataset, ind):
    for i in ind:
        batch = get_batch(dataset, [i])
        images(batch["images"], win="%d" % i, denorm=1)


def images(imgs, mask=None, heatmap=None, label=False, enlarge=0, 
    win="9999", nrow=4, gray=False, env="main", denorm=0,
    title=None, resize=True):

    import visdom
    vis = visdom.Visdom(port=1111)

    """
    Display images into the Visdom server
    """
    # Break dict into key -> image list
    if isinstance(imgs, dict):
        for k, img in zip(imgs.keys(), imgs.values()):
            image(img, mask, label, enlarge, str(k), nrow, env,
                  vis=vis, title=title,resize=resize)

    # Break list into set of images
    elif isinstance(imgs, list):
        for k, img in enumerate(imgs):
            image(img, mask, label, enlarge, "%s-%d"%(win,k), 
                  nrow, env, vis=vis, title=title,resize=resize)

    elif isinstance(imgs, plt.Figure):
        image(f2n(imgs), mask, label, enlarge, win, nrow, env, 
             gray=gray, vis=vis, denorm=denorm, title=title,resize=resize)

    else:
        if heatmap is not None:
            imgs = t2n(imgs)*0.4 + 0.6*t2n(gray2cmap(heatmap))

        image(imgs, mask, label, enlarge, win, nrow, env, 
             gray=gray, vis=vis, denorm=denorm, title=title,resize=resize)


def image(imgs, mask, label, enlarge, win, nrow, env="main",
          vis=None, gray=False, denorm=0, title=None,resize=True):
    
    if title is None:
        title = win

    if isinstance(mask, list):
        imgs = pretty_vis(imgs, mask, alpha=0.0, dpi=100)
    else:
        imgs = get_image(imgs, mask, label, enlarge, gray,denorm)
        if resize:
            imgs = resizeMax(imgs, max_size=500)
    options = dict(title=title, xtick=True, ytick=True)
    
    vis.images(imgs, opts=options, nrow=nrow, win=win, 
               env=env)


def get_image(imgs, mask=None, label=False, enlarge=0, gray=False,
    denorm=0):
    if denorm:
        imgs = denormalize(imgs)
    if isinstance(imgs, PIL.Image.Image):
        imgs = np.array(imgs)
    if isinstance(mask, PIL.Image.Image):
        mask = np.array(mask)

    imgs = t2n(imgs).copy()
    imgs = l2f(imgs)

    if mask is not None and mask.sum()!=0:
        imgs = maskOnImage(imgs, mask, enlarge)

    # LABEL
    elif (not gray) and (label or 
        imgs.ndim == 2 or 
        (imgs.ndim == 3 and imgs.shape[0] != 3) or
        (imgs.ndim == 4 and imgs.shape[1] != 3)):
        
        imgs = label2Image(imgs)

        if enlarge:
            imgs = zoom(imgs, 11)

    
    # Make sure it is 4-dimensional
    if imgs.ndim == 3:
        imgs = imgs[np.newaxis]

    return imgs



######### GET FP, TP, FN, TN
def save_qualitative_mse(model, dataset, fname=None):
    score, scoreList = val.validate_mse(model, dataset, return_scoreList=1) 

    #print("baseline: %.3f - val score: %s" % ((test_set.n_fishers**2).sum(), str(score)))

    TP = np.where((dataset.n_objects > 0) * (scoreList == 0))[0][:8]
    TN = np.where((dataset.n_objects == 0) * (scoreList == 0))[0][:8]
    FP = np.where((dataset.n_objects == 0) * (scoreList > 0))[0][:8]
    FN = np.where((dataset.n_objects > 0) * (scoreList > 0))[0][:8]

    for ind, win in zip([TP,TN,FP,FN], ["TP","TN","FP","FN"]):
        if ind.size == 0:
            continue
        batch = get_batch(dataset,  indices=ind)
        # images(batch["images_original"], mask=model.predict(batch), win=win,
        #  env="fisheries")


def addTitle(img, text="something"):
    source_img = n2p(f2l(img.copy()))

    draw = PIL.ImageDraw.Draw(source_img)

    draw.rectangle(((0, 0), (img.shape[1], 20)),
        fill="white")
    font = PIL.ImageFont.truetype("DejaVuSerif.ttf", 18)
    draw.text((0, 0), "  %s" % text, fill="black", font=font)

    return l2f(np.array(source_img))


from torch import optim


import contextlib

@contextlib.contextmanager
def empty_context():
    yield None

def get_metric_func(main_dict=None, metric_name=None):
  if metric_name is not None:
    return main_dict["metric_dict"][metric_name]
  return main_dict["metric_dict"]["metric_name"]


def val_test(main_dict, metric_name=None, n_workers=1):
  test_set = load_test(main_dict)
  model = load_best_model(main_dict)

  if metric_name is None:
    metric_name=main_dict["metric_name"]
  
  score = validate(model, test_set,
                   metric_class=main_dict["metric_dict"][metric_name],
                   num_workers=n_workers)
  return score 


def prettyprint(main_dict):
      pprint.PrettyPrinter(depth=6).pprint(
            {k:main_dict[k] for k in main_dict 
                            if main_dict[k] is not None and k.find("_dict")==-1})

def print_welcome(main_dict):
    pprint.PrettyPrinter(depth=6).pprint(
            {k:main_dict[k] for k in main_dict 
                            if main_dict[k] is not None and k.find("_dict")==-1})

    print_header("EXP: %s,  Reset: %s" % 
                   (main_dict["exp_name"], 
                    main_dict["reset"]))



#### DATASET

def get_trainloader(main_dict):
  train_set = load_trainval(main_dict, train_only=True)
  sampler_name = main_dict["sampler_name"]
  dataloader = get_dataloader(
                              train_set, 
                              batch_size=main_dict["batch_size"], 
                              sampler=main_dict["sampler_dict"][sampler_name])
  return dataloader


def get_testloader(main_dict):
  test_set = load_test(main_dict)
  
  dataloader = get_dataloader(test_set, 
                              batch_size=main_dict["val_batchsize"], 
                              sampler=None)
  return dataloader

def load_test_dict(main_dict):
  return load_pkl(main_dict["path_save"] + "/test.pkl")

def save_test_dict(main_dict, test_dict):
  return save_pkl(main_dict["path_save"] + "/test.pkl", test_dict)

def load_history(main_dict):
  if not os.path.exists(main_dict["path_save"] + "/history.pkl"):
    return None
  
  return load_pkl(main_dict["path_save"] + "/history.pkl")

def history_exists(main_dict):
    if not os.path.exists(main_dict["path_save"] + "/history.pkl"):
        return False
    else:
        return True

def model_exists(main_dict):
    
    if not os.path.exists(main_dict["path_save"] + "/history.pkl"):
        return False
    else:
        history = load_pkl(main_dict["path_save"] + "/history.pkl")
        
        if not os.path.exists(main_dict["path_train_model"]):
            return False
        else:
            return True

def get_dataloader(dataset, batch_size, sampler_class=None):
  if sampler_class is None:
    trainloader = data.DataLoader(dataset, batch_size=batch_size, 
                                shuffle=True, num_workers=0, 
                                drop_last=False)
  else:
    trainloader = data.DataLoader(dataset, batch_size=batch_size, 
                                sampler=sampler_class(dataset), 
                                num_workers=0, 
                                drop_last=False)

  return trainloader


def subsetloader(dataset, batch_size, ind, num_workers=1):
  sampler = SubsetRandomSampler(ind)
  loader = data.DataLoader(dataset, batch_size=batch_size, 
                              sampler=sampler, 
                              num_workers=min(batch_size,2), 
                              drop_last=False)
  return loader


def load_trainval(main_dict, train_only=False):  
  
  path_datasets = main_dict["path_datasets"]
  dataset_name = main_dict["dataset_name"]
  trainTransformer = main_dict["trainTransformer"]
  testTransformer = main_dict["testTransformer"]
  dataset_options = main_dict["dataset_options"]

  
  train_set = main_dict["dataset_dict"][dataset_name](root=path_datasets, 
                                         split="train", 
                                         transform_function=main_dict["transform_dict"][trainTransformer],
                                         **dataset_options)
  if train_only:
    return train_set

  val_set = main_dict["dataset_dict"][dataset_name](root=path_datasets, 
                                         split="val", 
                                         transform_function=main_dict["transform_dict"][testTransformer],
                                         **dataset_options)

  stats = [{"dataset":dataset_name, 
            "n_train": len(train_set), 
            "n_val":len(val_set)}]
            
  print(pd.DataFrame(stats))

  return train_set, val_set


def load_test(main_dict):

  path_datasets = main_dict["path_datasets"]
  dataset_name = main_dict["dataset_name"]
  testTransformer = main_dict["testTransformer"]
  dataset_options = main_dict["dataset_options"]

  test_set = main_dict["dataset_dict"][dataset_name](root=path_datasets,  
                                             split="test", 
                                             transform_function=main_dict["transform_dict"][testTransformer],
                                             **dataset_options)

  return test_set


#### MODEL INIT
def create_model(main_dict, train_set=None):
  # LOAD MODELS
  model_name = main_dict["model_name"]
  model_options = main_dict["model_options"]
  model_options_tmp = copy.deepcopy(model_options)
  model_options_tmp["main_dict"] = copy.deepcopy(main_dict)
  if train_set is None:
    train_set = load_trainval(main_dict, train_only=True)

  
  model = main_dict["model_dict"][model_name](train_set=train_set, 
                                      **model_options_tmp).cuda()
  return model


def create_model_and_opt(main_dict, train_set=None):
  # LOAD MODELS
  model = create_model(main_dict, train_set=train_set)

  opt_name = main_dict["opt_name"]
  opt_options = main_dict["opt_options"]

  opt = main_dict["opt_dict"][opt_name](filter(lambda p: p.requires_grad, model.parameters()), 
                                **opt_options)
  return model, opt 


def create_opt(model, main_dict, train_set=None):
  # LOAD MODELS
  opt_name = main_dict["opt_name"]
  opt_options = main_dict["opt_options"]

  opt = main_dict["opt_dict"][opt_name](filter(lambda p: p.requires_grad, model.parameters()), 
                                **opt_options)
  return opt 


def init_model_and_opt(main_dict, train_set=None):
  # SET TIME
  start_time = dt.datetime.now(dt.timezone(dt.timedelta(hours=-8.0)))
  start_time = start_time.strftime("%I:%M%p %a, %d-%b-%y")
  
  exp_name = main_dict["exp_name"]
  metric_name = main_dict["metric_name"]
  path_save = main_dict["path_save"]

  # LOAD HISTORY
  history = {"start_time": start_time,
             "exp_name":exp_name,
             "epoch":0,
             "metric_name":metric_name,
             "main_dict": {k:main_dict[k] for k in main_dict if k.find("_dict") == -1},
             "train": [],
             "val": [],
             "best_model":{},
             "trained_batch_names":[]}

  model, opt = create_model_and_opt(main_dict, train_set)

  print("Initializing model from scratch...")

  return model, opt, history


# LOADING AND SAVING MODELS

def load_latest_model(main_dict, train_set=None):
  model = create_model(main_dict, 
                      train_set=train_set)
  history = load_pkl(main_dict["path_save"] + "/history.pkl")
  name = type(model).__name__
  if len(history["train"]) == 0:
    print("No model saved - initailizing...{}".format(name))
    return model 
  model.load_state_dict(torch.load(main_dict["path_train_model"]), strict=False)    
  print("Load latest model for {} ... epoch {}".format(name,
            history["train"][-1]["epoch"]))
  return model 


def load_lcfcn(train_set, mode="lcfcn"):
    from models.lcfcn import Res50FCN
    model = Res50FCN(train_set).cuda()
    if mode=="prm":
        model.load_state_dict(torch.load(main_dict["path_train_model"]))
    if mode=="lcfcn":
        name = "pascal_ResFCN"
        path = "/mnt/home/issam/Research_Ground/LCFCN/checkpoints/best_model_{}.pth".format(name)
    model.load_state_dict(torch.load(path))   
    return model

def load_model_epoch(main_dict, epoch, train_set=None):
  model = create_model(main_dict, train_set=train_set)
  
  model.load_state_dict(torch.load(main_dict["path_train_model"].replace(".pth","_{}.pth".format(epoch))), strict=False)    
  print("Load model at epoch {}".format(epoch))
  return model 


def load_latest_model_and_opt(main_dict, train_set=None):
  model, opt = create_model_and_opt(main_dict, 
                                    train_set=train_set)

  history = load_pkl(main_dict["path_save"] + "/history.pkl")
  
  model.load_state_dict(torch.load(main_dict["path_train_model"]))
  opt.load_state_dict(torch.load(main_dict["path_train_opt"]))

  return model, opt, history


def save_latest_model_and_opt(main_dict, model, opt, history):
  pbar = tqdm.tqdm(desc="Saving Model...Don't Exit...       ", leave=False)
  
  create_dirs(main_dict["path_train_model"])
  torch.save(model.state_dict(), main_dict["path_train_model"])
  torch.save(opt.state_dict(), main_dict["path_train_opt"])

  save_pkl(main_dict["path_history"], history)
  
  pbar.close()


#######################################
def load_best_annList(main_dict):
    return load_pkl(main_dict["path_best_annList"])

def load_best_model(main_dict, train_set=None):
  model = create_model(main_dict, train_set=train_set)
  history = load_history(main_dict)

  if os.path.exists(main_dict["path_best_model"]):

    model.load_state_dict(torch.load(main_dict["path_best_model"]))
    print("Loaded best model...epoch {}".format(history["best_model"]["epoch"]))

  else:
    assert history is None
    print("Loaded model from scratch...")

  return model


def save_test_model(main_dict, model, fname):
  pbar = tqdm.tqdm(desc="Saving Model...Don't Exit...       ", leave=False)
  
  create_dirs(fname)
  torch.save(model.state_dict(), fname)

  pbar.close()

  print("New best model...")


def save_best_model(main_dict, model):
  pbar = tqdm.tqdm(desc="Saving Model...Don't Exit...       ", leave=False)
  
  create_dirs(main_dict["path_best_model"])
  torch.save(model.state_dict(), main_dict["path_best_model"])

  pbar.close()

  print("New best model...")



def save_model(path, model):
  pbar = tqdm.tqdm(desc="Saving Model...Don't Exit...       ", leave=False)
  create_dirs(path)
  torch.save(model.state_dict(), path)
  pbar.close()



### SUMMARY
def summary(main_dict, which):
  history = load_history(main_dict)
  if history is None:
    return "None"
  metric_name = main_dict["metric_name"]
  
  if "epoch" not in history["best_model"]:
    return "Not Yet"


  best_epoch = history["best_model"]["epoch"]
  epoch = history["epoch"]

  if which == "train":
    try:
      loss = history["train"][-1][main_dict["loss_name"]]
    except:
      loss = "Not Found"
    #loss = 1
    best_score =  history["best_model"][metric_name]
    score = ("loss: {:.3} | ({}/{}) {:.3f}".format 
             ( loss, best_epoch, epoch, best_score))

  if which == "test_count":
    fname = main_dict["path_save"] + "/test_count.pkl"

    records = load_pkl(fname)
    if best_epoch != records["best_epoch"]:
      state = "* "
    else:
      state = " "
    score = "({}/{}){}{}".format(best_epoch, epoch, state,
                              records[metric_name])

  if which == "fscore":
    fname = main_dict["path_save"] + "/test_fscore.pkl"

    records = load_pkl(fname)
    if best_epoch != records["best_epoch"]:
      state = "* "
    else:
      state = " "
    score = "({}/{}){}{}".format(best_epoch, epoch, state,
                              records["fscore"])

  return score

def get_summary(main_dict):
  if os.path.exists(main_dict["path_save"] + "/history.pkl"):
    history = load_pkl(main_dict["path_save"] + "/history.pkl")

    loss_name = main_dict["loss_name"]
    metric_name = main_dict["metric_name"] 
    dataset_name = main_dict["dataset_name"] 
    config_name = main_dict["config_name"] 
    
    summary = {}
    summary["config"] = config_name
    summary["dataset"] = dataset_name
    summary["metric_name"] = metric_name

    # train
    try:
      summary["_train_%s"% metric_name] = history["train"][-1][metric_name]
      summary["train_epoch"] = history["train"][-1]["epoch"]
      summary[loss_name] = "%.3f" % history["train"][-1][loss_name]
    except:
      pass

    # val
    try:
      epoch = history["val"][-1]["epoch"]
      score = history["val"][-1][metric_name]
      summary["val"] = ("%d-%.3f" %
                        (epoch, score))

      epoch = history["best_model"]["epoch"]
      score = history["best_model"][metric_name]
      summary["val_best"] = ("%d-%.3f" %
                             (epoch, score))
    except:
      pass

    return summary

  else:
    return {}



# Main dict
class MainClass:
    def __init__(self, path_datasets, path_models, path_samplers, path_transforms,
                       path_metrics, path_losses, path_saves, project):
        self.path_datasets = path_datasets
        self.path_saves = path_saves
        self.dataset_dict = get_module_classes(path_datasets)
        self.model_dict = get_module_classes(path_models)

        self.loss_dict = get_functions(path_losses)
        self.metric_dict = get_functions(path_metrics)
        self.sampler_dict = get_functions(path_samplers)
        self.transform_dict = get_functions(path_transforms)
        self.project = project

        

        self.opt_dict = {"adam":optim.Adam, 
                         "adamFast":lambda params, lr, 
                          weight_decay:optim.Adam(params, lr=lr, betas=(0.9999,0.9999999), weight_decay=weight_decay),

                          "sgd":lambda params, lr, 
                           weight_decay:optim.SGD(params, lr=lr, weight_decay=weight_decay,
                                                                momentum=0.9)}
        # DATASETS

        
        
    def get_main_dict(self, mode, dataset_name, model_name, config_name, config, reset, 
                            epochs, metric_name, loss_name, 
                            gpu=None):
        main_dict = config
        main_dict["config_name"] = config_name
        main_dict["model_name"] = model_name
        main_dict["loss_name"] = loss_name
        main_dict["metric_name"] = metric_name
        main_dict["dataset_name"] = dataset_name
        main_dict["epochs"] = epochs
        main_dict["reset"] = reset
        main_dict["project_name"] = self.project
        main_dict["code_path"] = "/mnt/home/issam/Research_Ground/{}".format(self.project)
        # GET GPU
        set_gpu(gpu)

        main_dict["path_datasets"] = self.path_datasets
        main_dict["exp_name"] = ("dataset:{}_model:{}_metric:{}_loss:{}_config:{}".format 
                                (dataset_name, model_name, 
                                 metric_name, loss_name,config_name))


        # SAVE
        main_dict["path_save"] = "{}/{}/".format(self.path_saves, 
                                                 main_dict["exp_name"])

        path_save = main_dict["path_save"]


        main_dict["path_summary"] = main_dict["path_save"].replace("Saves", "Summaries")


        main_dict["metric_dict"] = self.metric_dict
        main_dict["sampler_dict"] = self.sampler_dict
        main_dict["loss_dict"] = self.loss_dict
        main_dict["model_dict"] = self.model_dict
        main_dict["dataset_dict"] = self.dataset_dict
        main_dict["opt_dict"] = self.opt_dict

        main_dict["transform_dict"] = self.transform_dict

        main_dict["path_history"]= path_save + "/history.pkl"
        main_dict["path_train_opt"]= path_save + "/State_Dicts/opt.pth"
        
        main_dict["path_train_model"]= path_save + "/State_Dicts/model.pth"
        main_dict["path_baselines"]= path_save + "/baselines.pkl"
        main_dict["path_best_model"]= path_save + "/State_Dicts/best_model.pth"
        main_dict["path_best_annList"]= path_save + "/State_Dicts/best_annList.pkl"

        assert_exist(main_dict["model_name"], self.model_dict)
        assert_exist(main_dict["loss_name"], self.loss_dict)
        assert_exist(main_dict["metric_name"], self.metric_dict)
        assert_exist(main_dict["dataset_name"], self.dataset_dict)

        return main_dict



def assert_exist(key, dict):
    if key not in dict:
        raise ValueError("{} does not exist...".format(key))



def compute_gradient_2d(edges, img):
    h, w, _ = img.shape
    A = img[edges[0] // w, (edges[0] % w)]
    B = img[edges[1] // w, (edges[1] % w)]

    gradient = np.abs(A - B).max(1)
    return gradient

def get_affinity(img):
    img = t2n(img).squeeze().transpose(1,2,0)
    dtype = img.dtype

    h, w, c = img.shape

    D = np.arange(h*w)

    E = _make_edges_3d(h, w)
    W = compute_gradient_2d(E, img)

    n_voxels = D.size
    diag_idx = np.arange(n_voxels)
    i_idx = np.hstack((E[0], E[1]))
    j_idx = np.hstack((E[1], E[0]))


    G = sparse.coo_matrix((np.hstack((W, W, D)),
                              (np.hstack((i_idx, diag_idx)),
                               np.hstack((j_idx, diag_idx)))),
                              (n_voxels, n_voxels),
                              dtype=dtype)

    return G

def sparse_c2t(A):
    values = A.data
    indices = np.vstack((A.row, A.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = A.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


def dataset2cocoformat(dataset_name):
    path_datasets = "datasets"
    path_transforms = 'addons/transforms.py'
    dataset_dict = get_module_classes(path_datasets)
    transform_dict = get_functions(path_transforms)
    _,val_set = load_trainval({"dataset_name":dataset_name,
                               "path_datasets":path_datasets,
                               "trainTransformer":"Tr_WTP_NoFlip",
                               "testTransformer":"Te_WTP",
                               "dataset_options":{},
                               "dataset_dict":dataset_dict,
                               "transform_dict":transform_dict})
    annList_path = val_set.annList_path
    
        
    import ipdb; ipdb.set_trace()  # breakpoint a06159cc //
    
    ann_json = {}
    ann_json["categories"] = val_set.categories
    ann_json["type"] = "instances"


    # Images
    imageList = []
    annList = []
    # id = 1

    for i in range(len(val_set)):
        print("{}/{}".format(i, len(val_set)))
        batch = val_set[i]

        image_id = batch["name"]

        height, width = batch["images"].shape[-2:]
        imageList += [{"file_name":batch["name"],
                      "height":height,
                      "width":width,
                      "id":batch["name"]}]

        maskObjects = batch["maskObjects"]
        maskClasses = batch["maskClasses"]
        n_objects = maskObjects[maskObjects!=255].max().item()
        for obj_id in range(1, n_objects+1):
            if obj_id == 0:
                continue

            binmask = (maskObjects == obj_id)
            segmentation = maskUtils.encode(np.asfortranarray(ms.t2n(binmask))) 
            segmentation["counts"] = segmentation["counts"].decode("utf-8")
            uniques = (binmask.long()*maskClasses).unique()
            uniques = uniques[uniques!=0]
            assert len(uniques) == 1

            category_id = uniques[0].item()
            
            annList += [{"segmentation":segmentation,
                          "iscrowd":0,
                          # "bbox":maskUtils.toBbox(segmentation).tolist(),
                          "area":int(maskUtils.area(segmentation)),
                         # "id":id,
                         "image_id":image_id,
                         "category_id":category_id}]
            # id += 1

    ann_json["annotations"] = annList
    ann_json["images"] = imageList

    import ipdb; ipdb.set_trace()  # breakpoint a5259132 //

    

    save_json(annList_path, ann_json)

    anns = load_json(annList_path)
    fname_dummy = annList_path.replace(".json","_best.json")
    annList = anns["annotations"]
    for a in annList:
        a["score"] = 1

    save_json(fname_dummy, annList)

    # Test should be 100
    cocoGt = COCO(annList_path)

    imgIds=sorted(cocoGt.getImgIds())
    imgIds = imgIds[0:100]
    imgIds = imgIds[np.random.randint(100)]
    cocoDt=cocoGt.loadRes(fname_dummy)

    cocoEval = COCOeval(cocoGt, cocoDt, "segm")
    cocoEval.params.imgIds  = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    assert cocoEval.stats[0] == 1
    assert cocoEval.stats[1] == 1
    assert cocoEval.stats[2] == 1



def visGT(batch, win="1", return_image=False, 
                     alpha=0.0):
    import ann_utils as au
    gt_annList = au.batch2annList(batch)
    # # Counts
    # count_diff(pred_dict_dice, batch)

    # img_points = get_image_points(batch)
    # dice_dict = au.annList2mask(pred_dict_dice["annList"], color=1)
        
    # image = pretty_vis(batch["images"], pred_dict_dice["annList"])
    image = pretty_vis(batch["images"], gt_annList, alpha=alpha, dpi=100)
    if return_image:
        return image
    else:
        images(image, resize=False, win=win)


def visPoints(model, batch, win="1", return_image=False, 
                     alpha=0.0):
    import ann_utils as au
    images(batch["images"], au.annList2mask(model.predict(batch, 
      predict_method="loc")["annList"])["mask"], enlarge=1, denorm=1)


def visBestObjectness(batch, win="1", return_image=False, 
                     alpha=0.0):
    import ann_utils as au
    points = batch["points"].cuda()
    pointList = au.mask2pointList(points)["pointList"]
    batch["maskVoid"] = None
    gt_annList = au.pointList2BestObjectness(pointList, batch, proposal_type="sharp")["annList"]

    # # Counts
    # count_diff(pred_dict_dice, batch)

    # img_points = get_image_points(batch)
    # dice_dict = au.annList2mask(pred_dict_dice["annList"], color=1)
        
    # image = pretty_vis(batch["images"], pred_dict_dice["annList"])
    image = pretty_vis(batch["images"], gt_annList, alpha=alpha, dpi=100)
    if return_image:
        return image
    else:
        images(image, resize=False, win=win)

def visBlobs(model, batch, win="1", 
                predict_method="BestDice", return_image=False, 
                with_void=False,alpha=0.0):

    if not with_void:
        batch["maskVoid"] = None
    pred_dict_dice = model.predict(batch, predict_method=predict_method)
    # # Counts
    # count_diff(pred_dict_dice, batch)

    # img_points = get_image_points(batch)
    # dice_dict = au.annList2mask(pred_dict_dice["annList"], color=1)
        
    # image = pretty_vis(batch["images"], pred_dict_dice["annList"])
    image = pretty_vis(batch["images"], pred_dict_dice["annList"], alpha=alpha, dpi=100)
    if return_image:
        return image
    else:
        images(image, resize=False, win=win)


def visLoc(model, batch, win="1", 
                predict_method="BestDice", return_image=False, 
                with_void=False,alpha=0.0):

    import ipdb; ipdb.set_trace()  # breakpoint 90ae8003 //
    

def visEmbed(model, batch, win="1", 
                predict_method="BestDice", return_image=False, 
                with_void=False,alpha=0.0):

    if not with_void:
        batch["maskVoid"] = None
    pred_dict_dice = model.predict(batch, predict_method=predict_method)
    
    image = pretty_vis(batch["images"], pred_dict_dice["annList"], alpha=alpha, dpi=100)
    if return_image:
        return image
    else:
        images(image, resize=False, win=win)

def pretty_vis(image, annList, show_class=False,alpha=0.0, dpi=200):
    import cv2
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.patches import Polygon
    from matplotlib.figure import Figure
    import ann_utils as au
    image = im = f2l(t2n(denormalize(image))).squeeze()
    image = image*0.7
    # im = im*0.7
    # box_alpha = 0.5
    color_list = colormap(rgb=True) / 255
    # fig = Figure()
    fig = plt.figure(frameon=False)
    canvas = FigureCanvas(fig)
    fig.set_size_inches(im.shape[1] / dpi, im.shape[0] / dpi)
    # ax = fig.gca()

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.axis('off')
    fig.add_axes(ax)
    ax.imshow(im)

    # Display in largest to smallest order to reduce occlusion
    # areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    # sorted_inds = np.argsort(-areas)

    mask_color_id = 0
    for i in range(len(annList)):
        ann = annList[i]

        # bbox = boxes[i, :4]
        # score = boxes[i, -1]

        # bbox = au.ann2bbox(ann)["shape"]
        # score = ann["score"]
        mask = au.ann2mask(ann)["mask"]
        # if score < thresh:
        #     continue

        # show box (off by default, box_alpha=0.0)
        # ax.add_patch(
        #     plt.Rectangle((bbox[0], bbox[1]),
        #                   bbox[2] - bbox[0],
        #                   bbox[3] - bbox[1],
        #                   fill=False, edgecolor='g',
        #                   linewidth=1.0, alpha=alpha))

        # if show_class:
        # ax.text(
        #     bbox[0], bbox[1] - 2,
        #     "Class: {}".format(ann["category_id"]),
        #     fontsize=5,
        #     family='serif',
        #     bbox=dict(facecolor='g', alpha=0.4, pad=0, edgecolor='none'),
        #     color='white')

        # show mask
        img = np.ones(im.shape)
        color_mask = color_list[mask_color_id % len(color_list), 0:3]
        mask_color_id += 1

        w_ratio = .4
        for c in range(3):
            color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio
        for c in range(3):
            img[:, :, c] = color_mask[c]
        e = mask

        _, contour, hier = cv2.findContours(
            e.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

        for c in contour:
            polygon = Polygon(
                c.reshape((-1, 2)),
                fill=True, facecolor=color_mask,
                edgecolor=color_mask, linewidth=3.0,
                alpha=0.5)
            ax.add_patch(polygon)

    
    canvas.draw()       # draw the canvas, cache the renderer
    width, height = fig.get_size_inches() * fig.get_dpi()
    # image = np.fromstring(canvas.tostring_rgb(), dtype='uint8')

    fig_image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    plt.close()
    return fig_image

    



def colormap(rgb=False):
    color_list = np.array(
        [
            0.000, 0.447, 0.741,
            0.850, 0.325, 0.098,
            0.929, 0.694, 0.125,
            0.494, 0.184, 0.556,
            0.466, 0.674, 0.188,
            0.301, 0.745, 0.933,
            0.635, 0.078, 0.184,
            0.300, 0.300, 0.300,
            0.600, 0.600, 0.600,
            1.000, 0.000, 0.000,
            1.000, 0.500, 0.000,
            0.749, 0.749, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 1.000,
            0.667, 0.000, 1.000,
            0.333, 0.333, 0.000,
            0.333, 0.667, 0.000,
            0.333, 1.000, 0.000,
            0.667, 0.333, 0.000,
            0.667, 0.667, 0.000,
            0.667, 1.000, 0.000,
            1.000, 0.333, 0.000,
            1.000, 0.667, 0.000,
            1.000, 1.000, 0.000,
            0.000, 0.333, 0.500,
            0.000, 0.667, 0.500,
            0.000, 1.000, 0.500,
            0.333, 0.000, 0.500,
            0.333, 0.333, 0.500,
            0.333, 0.667, 0.500,
            0.333, 1.000, 0.500,
            0.667, 0.000, 0.500,
            0.667, 0.333, 0.500,
            0.667, 0.667, 0.500,
            0.667, 1.000, 0.500,
            1.000, 0.000, 0.500,
            1.000, 0.333, 0.500,
            1.000, 0.667, 0.500,
            1.000, 1.000, 0.500,
            0.000, 0.333, 1.000,
            0.000, 0.667, 1.000,
            0.000, 1.000, 1.000,
            0.333, 0.000, 1.000,
            0.333, 0.333, 1.000,
            0.333, 0.667, 1.000,
            0.333, 1.000, 1.000,
            0.667, 0.000, 1.000,
            0.667, 0.333, 1.000,
            0.667, 0.667, 1.000,
            0.667, 1.000, 1.000,
            1.000, 0.000, 1.000,
            1.000, 0.333, 1.000,
            1.000, 0.667, 1.000,
            0.167, 0.000, 0.000,
            0.333, 0.000, 0.000,
            0.500, 0.000, 0.000,
            0.667, 0.000, 0.000,
            0.833, 0.000, 0.000,
            1.000, 0.000, 0.000,
            0.000, 0.167, 0.000,
            0.000, 0.333, 0.000,
            0.000, 0.500, 0.000,
            0.000, 0.667, 0.000,
            0.000, 0.833, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 0.167,
            0.000, 0.000, 0.333,
            0.000, 0.000, 0.500,
            0.000, 0.000, 0.667,
            0.000, 0.000, 0.833,
            0.000, 0.000, 1.000,
            0.000, 0.000, 0.000,
            0.143, 0.143, 0.143,
            0.286, 0.286, 0.286,
            0.429, 0.429, 0.429,
            0.571, 0.571, 0.571,
            0.714, 0.714, 0.714,
            0.857, 0.857, 0.857,
            1.000, 1.000, 1.000
        ]
    ).astype(np.float32)
    color_list = color_list.reshape((-1, 3)) * 255
    if not rgb:
        color_list = color_list[:, ::-1]
    return color_list