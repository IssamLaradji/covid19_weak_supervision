import json
import torch
import numpy as np
import subprocess
import json
import torch
import pylab as plt
import numpy as np
import cv2
from PIL import ImageFont, ImageDraw, Image
from tqdm import tqdm 
from pycocotools.coco import COCO
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
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
from pycocotools import mask as maskUtils
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
import misc as ms 
from losses import splits
from core import response_map as rm
from losses import helpers as l_helpers
# from models.helpers import helpers as m_helpers

def visBlobDict(model, batch):
    blob_dict = l_helpers.get_blob_dict(model, batch, training=True)
    return blob_dict

def visSplit(model, batch, boundary=False, split_mode="water",add_bg=False):
    images = batch["images"].cuda()
    points = batch["points"].cuda()
    counts = batch["counts"].cuda()

    O = model(images)
    S = F.softmax(O, 1)
    S_log = F.log_softmax(O, 1)

    if boundary:
        mask = l_helpers.compute_boundary_loss(S_log, S, points, counts, add_bg=add_bg, return_mask=True)
    else:
        blob_dict = l_helpers.get_blob_dict(model, batch, training=True)
        mask = l_helpers.compute_split_loss(S_log, S, points, blob_dict, split_mode=split_mode, return_mask=True)

    ms.images(get_image_points(batch), mask.astype(int), win="split")

def get_image_points(batch):
    img = ms.denormalize(batch["images"])
    img_points = ms.get_image(img, mask=batch["points"], enlarge=1)
    return img_points

def visSp_points(batch):
    ms.images(batch["images"], l_helpers.get_sp_points(batch).long(), denorm=1, win="sp_points")

def visSp(batch):
    ms.images(batch["images"], l_helpers.get_sp(batch), denorm=1, win="superpixel")

from core import blobs_utils as bu


def count_diff(pred_dict, batch):
    for i, c in enumerate(pred_dict["counts"].ravel()):
        if c == 0:
            continue
        gt_count = batch["counts"].squeeze()[i]
        print("# blobs class {}: P: {} - GT:{}".format(i+1, int(c),gt_count))
    print("True Classes:", batch["points"].unique())

from skimage.transform import resize, rescale
def visGt(batch):
    gtAnnDict = au.load_gtAnnDict(main_dict)
    annList = [a for a in gtAnnDict["annotations"] if a["image_id"]==batch["name"][0]]
    ms.images(batch["images"], 
              au.annList2mask(annList)["mask"], 
              denorm=1, win="2")

import ann_utils as au
def visBaselines(batch, win="", return_image=False):
    if 1:
        # Preds
        pred_dict_obj = au.pointList2BestObjectness(batch["lcfcn_pointList"], batch)
        pred_dict_uB = au.pointList2UpperBound(batch["lcfcn_pointList"], batch)
        
        img_points = get_image_points(batch)

        img_obj =  ms.get_image(img_points, au.annList2mask(pred_dict_obj["annList"], color=1)["mask"])
        img_ub =  ms.get_image(img_points, au.annList2mask(pred_dict_uB["annList"], color=1)["mask"])
        
        img_points = rescale_image(img_points, "Original")
        if img_points.shape[-1] > img_points.shape[-2]:
            axis = 1
        else:
            axis = 0

        imgAll = np.concatenate([img_points, 
                                 rescale_image(img_obj, "BestObjectness", len(pred_dict_obj["annList"])), 
                                 rescale_image(img_ub, "UpperBound",len(pred_dict_uB["annList"]))], axis=axis)

        if return_image:
            return imgAll
        else:
            ms.images(imgAll, win=("image: {} - counts: {} (1) GT"
                                " - (2) BestObjectness"
                                " - (3) UpperBound".format(batch["index"][0], 
                                    len(batch["lcfcn_pointList"]))))

def rescale_image(img, text="Image", n_preds=None):
    scale = min(1, 400./ max(img.shape))   
    img = rescale(ms.f2l(img).squeeze(),anti_aliasing=True,
     scale=scale, mode="constant",multichannel=True, preserve_range=True)

    pil_im = Image.fromarray((img*255).astype("uint8"))

    draw = ImageDraw.Draw(pil_im)
    # font = ImageFont.truetype("Roboto-Regular.ttf", 50)
    # Draw the text
    if n_preds is None:
        draw.text((0, 0), text)
    else:
        draw.text((0, 0), text + " - n_preds: {}".format(n_preds))
    img = np.array(pil_im)

    return img

def visQuantitative(model, batch, win="", return_image=False):
    if 1:
        # Preds
        pred_dict_dice = model.predict(batch, predict_method="BestDice")
        pred_dict_blobs = model.predict(batch, predict_method="blobs")
        pred_dict_OB = au.pointList2BestObjectness(batch["lcfcn_pointList"], batch)
        # Counts
        count_diff(pred_dict_dice, batch)

        img_points = get_image_points(batch)
        blob_dict = au.annList2mask(pred_dict_blobs["annList"], color=1)
        dice_dict = au.annList2mask(pred_dict_dice["annList"], color=1)
        OB_dict = au.annList2mask(pred_dict_OB["annList"], color=1)
        
        img_blobs =  ms.get_image(img_points, blob_dict["mask"])
        img_dice =  ms.get_image(img_points, dice_dict["mask"])
        img_OB =  ms.get_image(img_points, OB_dict["mask"])
        
        img_points = rescale_image(img_points, "Original", n_preds=batch["counts"].sum().item())

        if img_points.shape[-1] > img_points.shape[-2]:
            axis = 1
        else:
            axis = 0

        imgAll = np.concatenate([img_points, 
                                rescale_image(img_OB, "BestObjectness", len(pred_dict_OB["annList"])),
                                 rescale_image(img_blobs, "Blobs", len(pred_dict_blobs["annList"])), 
                                 rescale_image(img_dice, "BestDice", len(pred_dict_dice["annList"]))], axis=axis)
        if return_image:
            return imgAll
        else:
            ms.images(imgAll, win="image: {}  (1) GT - (2) Blobs - (3) BestDice".format(batch["index"][0]))

    # if return_image:
    #     return ms.get_image(img_points, pred_dict["blobs"])
    # else:
    #     ms.images(img_points, pred_dict["blobs"], win="blobs {}".format(win))
    


def visBlobs(model, batch, win="", predict_method="BestDice", return_image=False):
    if 1:
        # Preds
        pred_dict_dice = model.predict(batch, predict_method=predict_method)
       
        # Counts
        count_diff(pred_dict_dice, batch)

        img_points = get_image_points(batch)
        dice_dict = au.annList2mask(pred_dict_dice["annList"], color=1)
        
        img_dice =  ms.get_image(img_points, dice_dict["mask"])
        
        img_points = rescale_image(img_points, "Original", n_preds=batch["counts"].sum().item())



        imgAll = rescale_image(img_dice, "BestDice", len(pred_dict_dice["annList"]))
        if return_image:
            return imgAll
        else:
            ms.images(imgAll, win="image: {}  (1) GT - (2) Blobs - (3) BestDice".format(batch["index"][0]))

    # if return_image:
    #     return ms.get_image(img_points, pred_dict["blobs"])
    # else:
    #     ms.images(img_points, pred_dict["blobs"], win="blobs {}".format(win))
    


def visBlobs_old(model, batch, win="", predict_method="blobs", 
             cocoGt=None, return_image=False, split=False):
    
    pred_dict = model.predict(batch, predict_method=predict_method)

    for i, c in enumerate(pred_dict["counts"].ravel()):
        if c == 0:
            continue
        gt_count = batch["counts"].squeeze()[i]
        print("# blobs class {}: P: {} - GT:{}".format(i+1, int(c),gt_count))

    print("True Classes:", batch["points"].unique())

    img_points = get_image_points(batch)
    if cocoGt is not None:
        # print(batch["image_id"][0])
        try:
            annList = cocoGt.imgToAnns[batch["image_id"][0].item()]
        except:
            annList = cocoGt.imgToAnns[batch["image_id"][0]]
        # print(annList)
        # print(au.annList2mask(cocoGt.imgToAnns[batch["image_id"][0].item()])["mask"])
        ms.images(img_points, au.annList2mask(annList)["mask"], 
                  win="gt {}".format(win))
    if return_image:
        return ms.get_image(img_points, pred_dict["blobs"])
    else:
        ms.images(img_points, pred_dict["blobs"], win="blobs {}".format(win))
    

    if predict_method not in [None, "blobs"] and not return_image: 
        pred_dict = model.predict(batch, predict_method='blobs')
        ms.images(img_points, pred_dict["blobs"], win="pure blobs {}".format(win))

    if split:
        print(pred_dict.keys())
        ms.images(img_points, 
                  1-au.probs2splitMask_all(pred_dict["probs"], pointList=au.points2pointList(batch["points"])["pointList"])["maskList"][0]["mask"],
                  win="split")

def visBlobList(model, dataset, indices, predict_method=None):
    for i in indices:
        visBlobs(model, ms.get_batch(dataset, [i]), 
             win=str(i),predict_method=predict_method)

def saveBlobList(model, dataset, indices, predict_method=None):
    for i in indices:
        img = visBlobs(model, ms.get_batch(dataset, [i]), 
                        win=str(i),predict_method=predict_method, return_image=True)
        ms.imsave(fname="/mnt/home/issam/Summaries/pascal/{}_{}.png".format(i,predict_method), arr=img)
        if predict_method is not None:
            img = visBlobs(model, ms.get_batch(dataset, [i]), 
                            win=str(i),predict_method="blobs", return_image=True)
            ms.imsave(fname="/mnt/home/issam/Summaries/pascal/{}_blobs.png".format(i), arr=img)


def visAnns(model, batch, cocoGt, win="", predict_proposal=None):
    #blobs = model.predict(batch, "blobs")
    # probs = F.softmax(model(batch["images"].cuda()),dim=1).data
    # blobs, counts = m_helpers.get_blobs(probs, return_counts=True)
    if predict_proposal is not None:
        model.set_proposal(predict_proposal)
    pred_annList, counts = model.predict(batch, "blobs_counts",
                                        return_annList=True)
    for i, ann in enumerate(pred_annList):
        print("object {} score: {:3f}".format(i, ann["score"]))
    image_id = int(batch["name"][0])
    annList = cocoGt.imgToAnns[image_id]
    mask = annList2mask(annList)

    dt_mask = annList2mask(pred_annList)
    ms.images(batch["images"], mask, denorm=1, win=win+"_true")
    ms.images(batch["images"], dt_mask, denorm=1, win=win+"_pred_{}".format(predict_proposal))

    for k, ann in enumerate(pred_annList):
        mask = (ann2mask(ann) !=0).astype(float)
        print(batch["images"].shape)
        print(mask.shape)
        print(ann["category_id"])
        blobs = torch.zeros(1, 21, mask.shape[0], mask.shape[1])
        blobs[0, ann["category_id"]] = torch.FloatTensor(mask)
        # asa
        # print("uniques", blobs.unique())
        # blobs = (blobs == blobs.max()[0]).float()

        # print(blobs.sum())
        excited_mask = rm.guided_backprop(model, 
                                          batch["images"].clone(), 
                                          gradient=(blobs.cuda()))
        excited_mask = excited_mask.mean(1)
        excited_mask = np.abs(excited_mask)

        scale = np.linalg.norm(mask) * np.linalg.norm(excited_mask)
        excited_score = excited_mask.ravel().dot(mask.ravel()) / scale
        print("Excited Score: {}".format(excited_score))
        print("Outside Values: {}".format(excited_mask[0][mask==0].mean()))
        excited_mask = ms.gray2cmap(excited_mask)
        ms.images(excited_mask, win=win+"_pred_{}_ann_".format(predict_proposal,k))


        
    model.set_proposal(model.proposal_name_default)
    return image_id, pred_annList


def visAnnList(model, dataset, indices, cocoGt, predict_proposal=None):
    image_id_list = []
    annList_list = []

    annList_dict = {}
    for i in indices:
        image_id, annList = visAnns(model, ms.get_batch(dataset, [i]), 
                                    cocoGt=cocoGt, win=str(i),
                                    predict_proposal=predict_proposal)
        image_id_list += [image_id]
        annList_list += annList

        annList_dict[i] = annList

    # ms.save_json("tmp.json", annList_list)
    # cocoDt = cocoGt.loadRes("tmp.json")
    # # print(ms.load_json("tmp.json")[0])

    # cocoEval = COCOeval(cocoGt, cocoDt, "segm")

    # # cocoEval = COCOeval(cocoGt, cocoDt, annType)
    # #cocoEval.params.imgIds  = list(set([v["image_id"] for v in cocoDt.anns.values()]))
   
    # cocoEval.params.imgIds = image_id_list
    # cocoEval.evaluate()
    # cocoEval.accumulate()
    # cocoEval.summarize()

    return annList_dict

def ann2mask(ann):
        mask =  maskUtils.decode(ann["segmentation"])
        mask[mask==1] = ann["category_id"]

        return mask

def annList2mask(annList):
    
    
    mask = None
    for ann in annList:
        if mask is None:
            mask = ann2mask(ann)
        else:
            mask += ann2mask(ann)

    return mask

def visPoints(model, batch):
    ms.images(batch["images"], batch["points"], denorm=1, enlarge=1, win="points")





def old_visSplit(model, batch, split_mode="water", win="split"):
    image = ms.denormalize(batch["images"])
    image = ms.f2l(ms.t2n(image)).squeeze()

    probs = ms.t2n(model.predict(batch, "probs").squeeze()[1])
    points =ms.t2n(batch["points"].squeeze())

    
    # if "blobs" in split_mode:
    #     # probs[probs < 0.5] = 0
    #     points[probs < 0.5] = 0

    if split_mode == "water" or split_mode == "water_blobs":
        split_matrix = splits.watersplit(probs, points)

    elif split_mode == "line":
        split_matrix = 1 - splits.line_splits(probs, points)

    elif split_mode == "line_blobs":
        # eqwe
        blob_dict = l_helpers.get_blob_dict(model, batch)
        split_matrix = np.zeros(points.shape, int)
        for b in blob_dict["blobList"]:
            if b["n_points"] < 2:
                continue
            
            points_class = (points==(b["class"] + 1)).astype("int")
            blob_ind = blob_dict["blobs"][b["class"] ] == b["label"]
            # print((blob_ind).sum())
            splitss = 1 - splits.line_splits(probs*blob_ind, 
                                               points_class*blob_ind)
            split_matrix += splitss*blob_ind
        
    # if "blobs" in split_mode:
    #     split_matrix[probs < 0.5] = 0

    # print(split_matrix.shape)
    print(split_matrix.max())
    split_img = ms.get_image(ms.l2f(image)[None], split_matrix.squeeze()[None].astype(int))
    #print(image.shape)

    ms.images(split_img, points, enlarge=True, win=win)
def old_visBlobs(model, batch, win="9999", label=0,  
             enlarge=0, return_dict=False, 
             training=False,
             split=None,
             color_types=False,
             which=0):
    batch = ms.copy.deepcopy(batch)
    image = batch["images"]
    points = batch["points"][0]
    counts = ms.t2n(batch["counts"][0])

    denormed_img = ms.denormalize(image)

    p_probs = ms.t2n(model.predict(batch, metric="probs"))[0]
    p_blobs = model.predict(batch, "blobs", training)

    p_blobs = p_blobs[0]        
    import ipdb; ipdb.set_trace()  # breakpoint c749dcc0 //

    p_counts = p_blobs.max((1,2))

    if p_blobs.shape[0] > 1:
       p_blobs = p_blobs[p_counts!=0]
       p_blobs = p_blobs[which]
    #     p_blobs = p_blobs[p_counts!=0]

    if p_counts.size > counts.size:
        counts = counts.repeat(p_counts.size)

    for i in range(p_counts.size):
        if int(counts[i]) == 0:
            continue
        sting = "class: %d: True-Pred: %d-%d" % (i, int(counts[i]), int(p_counts[i]))
        if not return_dict:
            print(sting)
    
    if split is not None:
        from addons import vis
        vis.visSplit(model, batch, split_mode=split)

    if color_types:
        p_blobs = colorBlobs(points, p_blobs)

    img_points = ms.get_image(denormed_img, mask=points, enlarge=1)
    
    img_blobs = ms.get_image(denormed_img, mask=p_blobs)
    img_combined = ms.get_image(img_points, mask=p_blobs)

    if return_dict:
        return {"images":denormed_img, 
                "with_blobs":img_blobs, 
                "with_points":img_points,
                "combined":img_combined,
                "p_counts":p_counts,
                "counts":counts}
    else:
        title = ""
        print(title)
        ms.images(denormed_img, title=title+"Images", win=win+"0")
        ms.images(img_points, title=title+"Points", win=win+"1")
        ms.images(img_blobs, title=title+"Blobs", win=win+"2")

def visTrain(main_dict, win="train"):
    pass

def colorBlobs(points, p_blobs):
    output = ms.t2n(points)*p_blobs
    all_colors = np.unique(p_blobs)

    point_colors, n_colors = np.unique(output, return_counts=True)

    new_blobs = np.zeros(p_blobs.shape)
    
    for c in all_colors:
        if c == 0:
            continue
        
        if c not in point_colors:
            new_blobs[p_blobs==c] = 1

        elif n_colors[point_colors==c] > 1:
            new_blobs[p_blobs==c] = 3

        else:
            new_blobs[p_blobs==c] = 2


    return new_blobs.astype(int)

def save_images(model, dataset, indices, path):

    for i in indices:
        print(i)
        batch = ms.get_batch(dataset, [i])
        try:
            if batch["counts"].sum().item() > 0:
                blob_dict = visBlobs(model, batch, return_dict=1)
                ms.create_dirs(path + "/{}.png".format(i))

                ms.save_images(path + "/{}.png".format(i), 
                               blob_dict["with_blobs"])
        except:
            pass