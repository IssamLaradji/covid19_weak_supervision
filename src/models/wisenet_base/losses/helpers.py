import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch.autograd import Variable
import numpy as np 
from .. import misc as ms
from skimage import morphology as morph
from torch.autograd import Function
from . import splits as sp
from skimage.segmentation import felzenszwalb, slic
# from core import proposals as prp
from skimage.segmentation import find_boundaries
from scipy.ndimage import distance_transform_cdt



    

        # n, c, h, w = O.shape
        # # seeds = torch.cat([fg_seeds, bg_seeds], 2)[:,:,None]
        # fA = O.view(1,c,-1)
        # fS = O[:,:, fg_bg_seeds["yList"], fg_bg_seeds["xList"]]

        # n_pixels = h*w
        # blobs = torch.zeros(h*w)

        # n_seeds =  fS.shape[-1]

        # maximum = 5000000
        # n_loops = int(np.ceil((n_pixels * n_seeds) / maximum))
        
        # for (s,e) in get_batches(n_pixels, size=n_pixels//n_loops):
        #     # s,e = map(int, (s,e))
        #     diff = au.log_pairwise_sum(fS[:,:,None], fA[:,:,s:e,None]) 
        #     blobs[s:e] = diff.max(2)[1] + 1 
        
        # bg_min_index = np.where(np.array(fg_bg_seeds["categoryList"])==0)[0].min()
        # assert len(fg_bg_seeds["yList"])//2 == bg_min_index
        # blobs[blobs > int(bg_min_index)] = 0
        # blobs = blobs.squeeze().reshape(h,w).long()

        # categoryDict = {}
        # for i, category_id in enumerate(fg_bg_seeds["categoryList"]):
        #     if category_id == 0:
        #          continue

        #     categoryDict[i+1] = category_id 

        # return {"blobs":ms.t2n(blobs), "categoryDict":categoryDict}

from skimage.measure import regionprops
def get_bg_dict(background):
    distance = distance_transform_cdt(background.squeeze())
    sp_mask = (slic(distance, n_segments=5, compactness=10)+1) * (distance!=0)
    
    mask_pos = np.where(background.squeeze())
    mask_neg = np.where(1 - background.squeeze())
    regionList = regionprops(sp_mask)
    # for l in np.unique(sp_mask):
    #     if l == 0:
    #         continue

    #     mask = sp_mask == l
    #     dis = distance_transform_cdt(mask)
    #     yx = np.where(dis==np.median(dis[dis>0]))

    yList = []
    xList = []
    for region in regionList:
        centroid = region.centroid

        y = int(centroid[0])
        x = int(centroid[1])

        yList += [y]
        xList += [x]

    return {"xList":xList, "yList":yList, "mask_pos":mask_pos, 
            "mask_neg":mask_neg}

def compute_gap_loss(probs_log, batch, thresh=0.5, with_bg=True):
    pointList = au.points2pointList(batch["points"])["pointList"]
    if len(pointList)==0:
        return 0.
        
        #annList = au.probs2GtAnnList(probs, points)
    
    propList = prp.Sharp_class(batch["name"])
    propDict = au.pointList2propDict(pointList, propList, thresh=0.5)

    loss = compute_gap(probs_log, propDict)
    if with_bg:
        loss += compute_gap_bg(probs_log, propDict)

    return loss


def compute_gap_per_instance_loss(probs_log, batch, thresh=0.5, with_bg=True):
    pointList = au.points2pointList(batch["points"])["pointList"]
    if len(pointList)==0:
        return 0.
        
        #annList = au.probs2GtAnnList(probs, points)
    
    with torch.no_grad():
        points_annList = au.points2annList(batch["points"])["annList"]

    # bg_mask = 
    loss = 0.
    for ann in points_annList:
        # if np.random.rand() < 0.5:
        #     matching_method = "dice"
        # else:
        #     matching_method = "objectness"
        fg_mask = au.ann2proposalRandom(ann, batch, matching_method="dice")
        # ms.images(fg_mask)
        if fg_mask is None:
            continue
       
        T = 1 - find_boundaries(fg_mask)
        T = torch.LongTensor(T).cuda()[None]

        loss += F.nll_loss(probs_log, T, ignore_index=1, reduction="elementwise_mean")


        fg_mask = fg_mask*ann["category_id"]
        fg_mask = torch.LongTensor(fg_mask).cuda()[None]
        if with_bg:
            loss += F.nll_loss(probs_log, fg_mask, ignore_index=0, reduction="elementwise_mean")
        # All proposals
        # fg_mask = au.ann2AllProposal(ann, batch, matching_method=matching_method)

    return loss

def compute_similarity_loss(probs, batch):
    n, k, h, w = probs.shape
    probs_flat = probs.view(n, k, h*w)[:,1]
    A = ms.get_affinity(batch["images"])
    G = ms.sparse_c2t(A).cuda()

    U = torch.mm(probs_flat, torch.mm(G, 1. - probs_flat.t())).sum()

    A_sum = torch.FloatTensor(np.asarray(A.sum(1))).cuda()
    D = torch.mm(probs_flat, A_sum).sum()

    loss = U / D

    return loss

from .. import ann_utils as au
def compute_fg_loss():
    propDict = au.pointList2propDict(pointList, batch, 
                                     single_point=single_point,
                                     thresh=0.5)
    background = propDict["background"]

    propDict = propDict["propDict"]

    yList = []
    xList = []
    for p in pointList:
        yList += [p["y"]]
        xList += [p["x"]]

    fg_seeds = O[:, :, yList, xList]
    n_seeds = fg_seeds.shape[-1]
    prop_mask = np.zeros((h, w))

    for i in range(n_seeds):
        annList = propDict[i]["annList"]

        if len(annList) == 0:
            mask = np.zeros(points.squeeze().shape)
            mask[propDict[i]["point"]["y"], propDict[i]["point"]["x"]] = 1
        else:

            if random_proposal:
                ann_i = np.random.randint(0, len(annList))
                mask = annList[ann_i]["mask"]
            else:
                mask = annList[0]["mask"]


       
        mask_ind = np.where(mask)
        prop_mask[mask!=0] = (i+1)

        
        f_A = fg_seeds[:,:,[i]]
        
        # Positive Embeddings
        n_pixels = mask_ind[0].shape[0]
        P_ind = np.random.randint(0, n_pixels, 100)
        yList = mask_ind[0][P_ind]
        xList = mask_ind[1][P_ind]
        fg_P = O[:,:,yList, xList]
        
        ap = - torch.log(similarity_function(f_A, fg_P)) 
        loss += ap.mean()

        # Get Negatives
        if n_seeds > 1:
            N_ind = [j for j in range(n_seeds) if j != i]
            f_N = fg_seeds[:,:,N_ind]
            an = - torch.log(1. - similarity_function(f_A, f_N)) 
            loss += an.mean()


def log_pairwise_l1_sum(fi, fj):
    diff = (fi - fj).abs().sum(1)
    return  (2./ (1+torch.exp(diff))).clamp(min=1e-6, max=(1.-1e-6))

def log_pairwise_rbf(fi, fj):
    diff = (fi - fj).pow(2).sum(1)

    # print(diff.max().item())
    return  torch.exp(-diff/64).clamp(min=1e-6, max=(1.-1e-6))

def log_pairwise_l1_rbf(fi, fj):
    diff = (fi - fj).abs().sum(1)
    # print(diff.max().item())
    return  torch.exp(-diff/64).clamp(min=1e-6, max=(1.-1e-6))

def compute_metric_loss_random(O, batch, random_proposal=False, similarity=""):

    n,c,h,w = O.shape
    
    points = batch["points"]
    batch["maskObjects"] = None 
    batch['maskClasses'] = None
    batch["maskVoid"] = None
    
    pointList = au.mask2pointList(points)["pointList"]

    loss = torch.tensor(0.).cuda()
    if len(pointList) == 0:
        return loss

    if "single_point" in batch:
        single_point = True
    else:
        single_point = False


    propDict = au.pointList2propDict(pointList, batch, 
                                     single_point=single_point,
                                     thresh=0.5)
    background = propDict["background"]
    
    propDict = propDict["propDict"]

    # a1 = propDict[0]["annList"][0]
    # a2 = propDict[1]["annList"][0]
    # img = ms.pretty_vis(batch["images"], [a1,a2],dpi=100)

    # ms.images(img) 
    # import ipdb; ipdb.set_trace()  # breakpoint 6758c0a1 //
    

    yList = []
    xList = []
    for p in pointList:
        yList += [p["y"]]
        xList += [p["x"]]

    fg_seeds = O[:, :, yList, xList]
    n_seeds = fg_seeds.shape[-1]
    prop_mask = np.zeros((h, w))

    for i in range(n_seeds):
        annList = propDict[i]["annList"]

        if len(annList) == 0:
            mask = np.zeros(points.squeeze().shape)
            mask[propDict[i]["point"]["y"], propDict[i]["point"]["x"]] = 1
        else:

            if random_proposal:
                ann_i = np.random.randint(0, len(annList))
                mask = annList[ann_i]["mask"]
            else:
                mask = annList[0]["mask"]


       
        mask_ind = np.where(mask)
        prop_mask[mask!=0] = (i+1)

        
        f_A = fg_seeds[:,:,[i]]
        
        # Positive Embeddings
        n_pixels = mask_ind[0].shape[0]
        P_ind = np.random.randint(0, n_pixels, 100)
        yList = mask_ind[0][P_ind]
        xList = mask_ind[1][P_ind]
        fg_P = O[:,:,yList, xList]
        
        ap = - torch.log(similarity_function(f_A, fg_P)) 
        loss += ap.mean()




        # Get Negatives
        if n_seeds > 1:
            N_ind = [j for j in range(n_seeds) if j != i]
            f_N = fg_seeds[:,:,N_ind]
            an = - torch.log(1. - similarity_function(f_A, f_N)) 
            loss += an.mean()

    # # Extract background seeds
    bg = np.where(background.squeeze())

    n_pixels = bg[0].shape[0]
    bg_ind = np.random.randint(0, n_pixels, n_seeds)
    yList = bg[0][bg_ind]
    xList = bg[1][bg_ind]
    f_A = O[:,:,yList, xList]


    bg_ind = np.random.randint(0, n_pixels, 100)
    yList = bg[0][bg_ind]
    xList = bg[1][bg_ind]
    f_P = O[:,:,yList, xList]


    # BG seeds towards BG pixels, BG seeds away from FG seeds
    ap = - torch.log(similarity_function(f_A[:,:,None], f_P[:,:,:,None])) 
    an = - torch.log(1. - similarity_function(f_A[:,:,None], fg_seeds[:,:,:,None])) 

    loss += ap.mean()
    loss += an.mean()

    if batch["dataset"][0] == "cityscapes" or batch["dataset"][0] == "coco2014":        
        n_max = 6
    else:
        n_max = 12

    if f_A.shape[2] < n_max:
        with torch.no_grad():
            diff = similarity_function(O.view(1,c,-1)[:,:,:,None], 
                                   torch.cat([fg_seeds, f_A], 2)[:,:,None])    
            labels = diff.max(2)[1] + 1 
            labels = labels <= n_seeds
            labels = labels.squeeze().reshape(h,w)
            bg = labels.cpu().long()*torch.from_numpy(background)        
            # ms.images(labels.cpu().long()*torch.from_numpy(background))


        # Extract false positive pixels
        bg_ind = np.where(bg.squeeze())
        n_P = bg_ind[0].shape[0]
        if n_P != 0:
            A_ind = np.random.randint(0, n_P, n_seeds)
            f_P = O[:,:, bg_ind[0][A_ind], bg_ind[1][A_ind]]

            ap = - torch.log(similarity_function(f_A[:,:,None], f_P[:,:,:,None])) 
            an = - torch.log(1. - similarity_function(f_P[:,:,None], fg_seeds[:,:,:,None])) 

            # if i < 3:
            loss += ap.mean()
            loss += an.mean()

    # if visualize:
    #     diff = log_func(O.view(1,64,-1)[:,:,:,None], torch.cat([se, f_A], 2)[:,:,None])
    #     labels = diff.max(2)[1] + 1
    #     labels[labels > n_se] = 0
    #     labels = labels.squeeze().reshape(h,w)

    #     ms.images(batch["images"], ms.t2n(labels),denorm=1, win="labels")
    #     ms.images(batch["images"], prop_mask.astype(int), denorm=1, win="true")
    #     ms.images(batch["images"], background.astype(int), denorm=1, win="bg")


    return loss / max(n_seeds, 1)

def compute_metric_loss_sum(O, batch, random_proposal=False, similarity=""):

    n,c,h,w = O.shape
    if similarity == "":
        similarity_function = au.pairwise_sum
    elif similarity == "L1":
        similarity_function = au.pairwise_l1_sum
    elif similarity == "L1_RBF":
        similarity_function = au.pairwise_l1_rbf
    elif similarity == "RBF":
        similarity_function = au.pairwise_rbf
    else:
        rferrwe
    points = batch["points"]
    batch["maskObjects"] = None 
    batch['maskClasses'] = None
    batch["maskVoid"] = None
    
    pointList = au.mask2pointList(points)["pointList"]

    loss = torch.tensor(0.).cuda()
    if len(pointList) == 0:
        return loss

    if "single_point" in batch:
        single_point = True
    else:
        single_point = False


    propDict = au.pointList2propDict(pointList, batch, 
                                     single_point=single_point,
                                     thresh=0.5)
    background = propDict["background"]
    
    propDict = propDict["propDict"]

    # a1 = propDict[0]["annList"][0]
    # a2 = propDict[1]["annList"][0]
    # img = ms.pretty_vis(batch["images"], [a1,a2],dpi=100)

    # ms.images(img) 
    # import ipdb; ipdb.set_trace()  # breakpoint 6758c0a1 //
    

    yList = []
    xList = []
    for p in pointList:
        yList += [p["y"]]
        xList += [p["x"]]

    fg_seeds = O[:, :, yList, xList]
    n_seeds = fg_seeds.shape[-1]
    prop_mask = np.zeros((h, w))

    for i in range(n_seeds):
        annList = propDict[i]["annList"]

        if len(annList) == 0:
            mask = np.zeros(points.squeeze().shape)
            mask[propDict[i]["point"]["y"], propDict[i]["point"]["x"]] = 1
        else:

            if random_proposal:
                ann_i = np.random.randint(0, len(annList))
                mask = annList[ann_i]["mask"]
            else:
                mask = annList[0]["mask"]


       
        mask_ind = np.where(mask)
        prop_mask[mask!=0] = (i+1)

        
        f_A = fg_seeds[:,:,[i]]
        
        # Positive Embeddings
        n_pixels = mask_ind[0].shape[0]
        P_ind = np.random.randint(0, n_pixels, 100)
        yList = mask_ind[0][P_ind]
        xList = mask_ind[1][P_ind]
        fg_P = O[:,:,yList, xList]
        
        ap = - torch.log(similarity_function(f_A, fg_P)) 
        loss += ap.mean()




        # Get Negatives
        if n_seeds > 1:
            N_ind = [j for j in range(n_seeds) if j != i]
            f_N = fg_seeds[:,:,N_ind]
            an = - torch.log(1. - similarity_function(f_A, f_N)) 
            loss += an.mean()

    # # Extract background seeds
    bg = np.where(background.squeeze())

    n_pixels = bg[0].shape[0]
    bg_ind = np.random.randint(0, n_pixels, n_seeds)
    yList = bg[0][bg_ind]
    xList = bg[1][bg_ind]
    f_A = O[:,:,yList, xList]


    bg_ind = np.random.randint(0, n_pixels, 100)
    yList = bg[0][bg_ind]
    xList = bg[1][bg_ind]
    f_P = O[:,:,yList, xList]


    # BG seeds towards BG pixels, BG seeds away from FG seeds
    ap = - torch.log(similarity_function(f_A[:,:,None], f_P[:,:,:,None])) 
    an = - torch.log(1. - similarity_function(f_A[:,:,None], fg_seeds[:,:,:,None])) 

    loss += ap.mean()
    loss += an.mean()

    if batch["dataset"][0] == "cityscapes" or batch["dataset"][0] == "coco2014":        
        n_max = 6
    else:
        n_max = 12

    if f_A.shape[2] < n_max:
        with torch.no_grad():
            diff = similarity_function(O.view(1,c,-1)[:,:,:,None], 
                                   torch.cat([fg_seeds, f_A], 2)[:,:,None])    
            labels = diff.max(2)[1] + 1 
            labels = labels <= n_seeds
            labels = labels.squeeze().reshape(h,w)
            bg = labels.cpu().long()*torch.from_numpy(background)        
            # ms.images(labels.cpu().long()*torch.from_numpy(background))


        # Extract false positive pixels
        bg_ind = np.where(bg.squeeze())
        n_P = bg_ind[0].shape[0]
        if n_P != 0:
            A_ind = np.random.randint(0, n_P, n_seeds)
            f_P = O[:,:, bg_ind[0][A_ind], bg_ind[1][A_ind]]

            ap = - torch.log(similarity_function(f_A[:,:,None], f_P[:,:,:,None])) 
            an = - torch.log(1. - similarity_function(f_P[:,:,None], fg_seeds[:,:,:,None])) 

            # if i < 3:
            loss += ap.mean()
            loss += an.mean()

    # if visualize:
    #     diff = log_func(O.view(1,64,-1)[:,:,:,None], torch.cat([se, f_A], 2)[:,:,None])
    #     labels = diff.max(2)[1] + 1
    #     labels[labels > n_se] = 0
    #     labels = labels.squeeze().reshape(h,w)

    #     ms.images(batch["images"], ms.t2n(labels),denorm=1, win="labels")
    #     ms.images(batch["images"], prop_mask.astype(int), denorm=1, win="true")
    #     ms.images(batch["images"], background.astype(int), denorm=1, win="bg")


    return loss / max(n_seeds, 1)



def compute_metric_loss_mean(O, batch, random_proposal=False):

    n,c,h,w = O.shape

    similarity_function = au.log_pairwise_mean

    points = batch["points"]
    batch["maskObjects"] = None 
    batch['maskClasses'] = None
    batch["maskVoid"] = None
    
    pointList = au.mask2pointList(points)["pointList"]

    loss = torch.tensor(0.).cuda()
    if len(pointList) == 0:
        return loss

    if "single_point" in batch:
        single_point = True
    else:
        single_point = False


    propDict = au.pointList2propDict(pointList, batch, 
                                     single_point=single_point,
                                     thresh=0.5)
    # img = ms.pretty_vis(batch["images"], propDict["propDict"][0]["annList"],dpi=100)

    # ms.images(img) 
    # import ipdb; ipdb.set_trace()  # breakpoint 6758c0a1 //

    background = propDict["background"]
    
    propDict = propDict["propDict"]

    yList = []
    xList = []
    for p in pointList:
        yList += [p["y"]]
        xList += [p["x"]]

    fg_seeds = O[:, :, yList, xList]
    n_seeds = fg_seeds.shape[-1]
    prop_mask = np.zeros((h, w))

    for i in range(n_seeds):
        annList = propDict[i]["annList"]

        if len(annList) == 0:
            mask = np.zeros(points.squeeze().shape)
            mask[propDict[i]["point"]["y"], propDict[i]["point"]["x"]] = 1
        else:

            if random_proposal:
                ann_i = np.random.randint(0, len(annList))
                mask = annList[ann_i]["mask"]
            else:
                mask = annList[0]["mask"]


       
        mask_ind = np.where(mask)
        prop_mask[mask!=0] = (i+1)

        
        f_A = fg_seeds[:,:,[i]]
        
        # Positive Embeddings
        n_pixels = mask_ind[0].shape[0]
        P_ind = np.random.randint(0, n_pixels, 100)
        yList = mask_ind[0][P_ind]
        xList = mask_ind[1][P_ind]
        fg_P = O[:,:,yList, xList]
        
        ap = - torch.log(similarity_function(f_A, fg_P)) 
        loss += ap.mean()




        # Get Negatives
        if n_seeds > 1:
            N_ind = [j for j in range(n_seeds) if j != i]
            f_N = fg_seeds[:,:,N_ind]
            an = - torch.log(1. - similarity_function(f_A, f_N)) 
            loss += an.mean()

    # # Extract background seeds
    bg = np.where(background.squeeze())

    n_pixels = bg[0].shape[0]
    bg_ind = np.random.randint(0, n_pixels, n_seeds)
    yList = bg[0][bg_ind]
    xList = bg[1][bg_ind]
    f_A = O[:,:,yList, xList]


    bg_ind = np.random.randint(0, n_pixels, 100)
    yList = bg[0][bg_ind]
    xList = bg[1][bg_ind]
    f_P = O[:,:,yList, xList]


    # BG seeds towards BG pixels, BG seeds away from FG seeds
    ap = - torch.log(similarity_function(f_A[:,:,None], f_P[:,:,:,None])) 
    an = - torch.log(1. - similarity_function(f_A[:,:,None], fg_seeds[:,:,:,None])) 

    loss += ap.mean()
    loss += an.mean()

    if batch["dataset"][0] == "cityscapes" or batch["dataset"][0] == "coco2014":        
        n_max = 6
    else:
        n_max = 12

    if f_A.shape[2] < n_max:
        with torch.no_grad():
            diff = similarity_function(O.view(1,c,-1)[:,:,:,None], 
                                   torch.cat([fg_seeds, f_A], 2)[:,:,None])    
            labels = diff.max(2)[1] + 1 
            labels = labels <= n_seeds
            labels = labels.squeeze().reshape(h,w)
            bg = labels.cpu().long()*torch.from_numpy(background)        
            # ms.images(labels.cpu().long()*torch.from_numpy(background))


        # Extract false positive pixels
        bg_ind = np.where(bg.squeeze())
        n_P = bg_ind[0].shape[0]
        if n_P != 0:
            A_ind = np.random.randint(0, n_P, n_seeds)
            f_P = O[:,:, bg_ind[0][A_ind], bg_ind[1][A_ind]]

            ap = - torch.log(similarity_function(f_A[:,:,None], f_P[:,:,:,None])) 
            an = - torch.log(1. - similarity_function(f_P[:,:,None], fg_seeds[:,:,:,None])) 

            # if i < 3:
            loss += ap.mean()
            loss += an.mean()

    # if visualize:
    #     diff = log_func(O.view(1,64,-1)[:,:,:,None], torch.cat([se, f_A], 2)[:,:,None])
    #     labels = diff.max(2)[1] + 1
    #     labels[labels > n_se] = 0
    #     labels = labels.squeeze().reshape(h,w)

    #     ms.images(batch["images"], ms.t2n(labels),denorm=1, win="labels")
    #     ms.images(batch["images"], prop_mask.astype(int), denorm=1, win="true")
    #     ms.images(batch["images"], background.astype(int), denorm=1, win="bg")


    return loss / max(n_seeds, 1)



def metric_base(O, batch, pointList=None):
    n,c,h,w = O.shape

    if pointList is None:
        points = batch["points"]
        batch["maskObjects"] = None 
        batch['maskClasses'] = None
        batch["maskVoid"] = None

        pointList = au.mask2pointList(points)["pointList"]

    
    if len(pointList) == 0:
        return None

    if "single_point" in batch:
        single_point = True
    else:
        single_point = False


    propDict = au.pointList2propDict(pointList, batch, 
                                     single_point=single_point,
                                     thresh=0.5)
    background = propDict["background"]

    propDict = propDict["propDict"]

    yList = []
    xList = []
    for p in pointList:
        yList += [p["y"]]
        xList += [p["x"]]

    return {"xList":xList, "yList":yList, "background":background, "propDict":propDict}



def metric_bg(O, metric_base_dict):
    import ipdb; ipdb.set_trace()  # breakpoint 92b524e9 //
    
    return {"xList":xList, "yList":yList, "background":background, "propDict":propDict}

    


def compute_gap(probs_log, propDict):
    loss = 0.
    for propList in propDict["propDict"]:
        ann_mask = au.annList2mask(propList["annList"])["mask"]
        if ann_mask is None:
            continue
        mask = (ann_mask[None]>1).astype(int)
        # ms.images(au.annList2mask(propDict["propDict"][0]["annList"])["mask"])
        category_id = propList["category_id"]

        # foreground = category_id*torch.LongTensor(mask).cuda()
        # foreground = foreground*split_background

        target = torch.LongTensor(mask*category_id).cuda()
        # import ipdb; ipdb.set_trace()  # breakpoint cc1bb723 //
        # ms.images(mask)

        loss +=  F.nll_loss(probs_log, target,
                            ignore_index=0, reduction="elementwise_mean")
    return loss

def compute_gap_bg(probs_log, propDict):
    background = 1 - propDict["background"]
    bg_target = torch.LongTensor(background).cuda()

    bg_loss = F.nll_loss(probs_log,  bg_target,
                       ignore_index=1, reduction="elementwise_mean")

    return bg_loss

def compute_image_loss(S, Counts):
    n,k,h,w = S.size()

    # GET TARGET
    ones = Variable(torch.ones(Counts.size(0), 1).long().cuda())
    BgFgCounts = torch.cat([ones, Counts], 1)
    Target = (BgFgCounts.view(n*k).view(-1) > 0).view(-1).float()

    # GET INPUT
    Smax = S.view(n, k, h*w).max(2)[0].view(-1)
    
    loss = F.binary_cross_entropy(Smax, Target, reduction="sum")
    
    return loss

def compute_bg_loss(S, Counts):
    n,k,h,w = S.size()

    # GET TARGET
    Target = torch.ones(Counts.size(0), 1).float().cuda()

    # GET INPUT

    Savg = S.view(n, k, h*w).mean(2)[:,:1].view(-1)
    
    loss = F.binary_cross_entropy(Savg, Target[0], 
                                   reduction="sum")
    
    return loss

def compute_fp_loss(S_log, blob_dict):
    blobs = blob_dict["blobs"]
    
    scale = 1.
    loss = 0.

    for b in blob_dict["blobList"]:
        if b["n_points"] != 0:
            continue

        T = np.ones(blobs.shape[-2:])
        T[blobs[b["class"]] == b["blob_id"]] = 0

        loss += scale * F.nll_loss(S_log, ms.n2l(T[None]),
                        ignore_index=1,  reduction="elementwise_mean")
    return loss 

def compute_split_loss(S_log, S, points, blob_dict, split_mode="line", return_mask=False):
    blobs = blob_dict["blobs"]
    S_numpy = ms.t2n(S[0])
    points_npy = ms.t2n(points).squeeze() 

    loss = 0.


    if return_mask:
        mask = np.ones(points_npy.shape)

    for b in blob_dict["blobList"]:
        if b["n_points"] < 2:
            continue

        c = b["class"] + 1
        probs = S_numpy[b["class"] + 1]

        points_class = (points_npy==c).astype("int")
        blob_ind = blobs[b["class"] ] == b["blob_id"]

        if split_mode == "line":
            T = sp.line_splits(probs*blob_ind, points_class*blob_ind)
        elif split_mode == "water":
            T = sp.watersplit(probs, points_class*blob_ind)*blob_ind
            T = 1 - T
        else:
            raise ValueError("%s LOL" % split_mode)

        if return_mask:
            mask[T==0] = 0

        # ms.images(T)
        scale = b["n_points"] + 1
        loss += float(scale) * F.nll_loss(S_log, ms.n2l(T)[None],
                        ignore_index=1,  reduction="elementwise_mean")

    if return_mask:
        return (mask == 0)

    return loss 


def compute_boundary_loss(S_log, S, points, counts, add_bg=False, return_mask=False):
    S_npy = ms.t2n(S[0])
    points_npy = ms.t2n(points).squeeze()
    loss = 0.

    if return_mask:
        mask = np.ones(points_npy.shape)

    for c in range(S.shape[1]):
        if c == 0:
            continue 

        points_class = (points_npy==c).astype(int)

        if add_bg:
            points_class[S_npy[c] == S_npy[c].min()] = 1

        if points_class.sum() <= 1:
            continue


        T = sp.watersplit(S_npy[c], points_class)
        T = 1 - T

        if return_mask:
            mask[T==0] = 0

        scale = float(counts.sum())
        loss += float(scale) * F.nll_loss(S_log, ms.n2l(T)[None],
                        ignore_index=1,  reduction="elementwise_mean")

    if return_mask:
        return (mask == 0)

    return loss


def compute_fixed_recursive_loss(model, batch, S_log):
    blob_dict = get_blob_dict(model.base_model, batch, training=True)

    blobs = blob_dict["blobs"]
    probs = ms.t2n(blob_dict["probs"])
    point_loss = 0.

    for b in blob_dict["blobList"]:
        if b["n_points"] != 1:
            continue

        T = np.zeros(blobs.shape[-2:])
        W = np.zeros(blobs.shape[-2:])

        ind = blobs[b["class"]] == b["blob_id"]
        T[ind] = (b["class"]+1)
        W[ind] = probs[b["class"]+1][ind]
        # ms.images(probs[b["class"]+1]>0.5)
        b_loss = F.nll_loss(S_log, ms.n2l(T[None]),
                        ignore_index=0, reduce=False) * torch.FloatTensor(W).cuda()

        point_loss += (b_loss.sum() / float(ind.sum()))

    return point_loss 


def compute_recursive_blob_loss(batch, S_log, blob_dict):
    blobs = blob_dict["blobs"]
    point_loss = 0.

    for b in blob_dict["blobList"]:
        if b["n_points"] != 1:
            continue

        T = np.zeros(blobs.shape[-2:])

        T[blobs[b["class"]] == b["blob_id"]] = (b["class"]+1)

        point_loss += F.nll_loss(S_log, ms.n2l(T[None]),
                        ignore_index=0,  reduction="elementwise_mean")
        
    return point_loss 


def compute_sp_loss(batch, S_log):
    points = batch["points"]
    point_loss = 0.

    segments = get_sp(batch)

    for c in np.unique(points):
        if c == 0:
            continue


        indices = ((ms.t2n(points)[0]!=0) * (segments))
        sp_ind = np.unique(indices[indices!=0])

        for sp_i in sp_ind:
            T = torch.from_numpy((segments == sp_i).astype(int))[None]
            T[T!=0] = int(c)
            point_loss +=  F.nll_loss(S_log, T.long().cuda(), 
                                 ignore_index=0,  reduction="elementwise_mean")

        # sp_ind_byte = torch.ByteTensor(np.isin(segments, sp_ind).astype(int))[None]

        # sp_points[sp_ind_byte] = float(c)
        
   
    return point_loss

def get_sp(batch):
    img = ms.t2n(ms.denormalize(batch["images"]))[0].transpose(1,2,0)
    segments = slic(img, n_segments=250, compactness=10, sigma=1)
    return segments

def get_sp_points(batch):
    points = batch["points"]
    
    segments = get_sp(batch)

    sp_points = torch.zeros(points.shape)

    for c in np.unique(points):
        if c == 0:
            continue
        indices = ((ms.t2n(points)[0]!=0) * (segments))
        sp_ind = np.unique(indices[indices!=0])

        sp_ind_byte = torch.ByteTensor(np.isin(segments, sp_ind).astype(int))[None]

        sp_points[sp_ind_byte] = float(c)

    return sp_points

from ..core import blobs_utils as bu
def compute_lc_local_loss(counts, points, probs, probs_log):
    n,k,h,w = probs.size()

    with torch.no_grad():
        annList = au.probs2GtAnnList(probs, points)

    # IMAGE LOSS
    probs_max = probs.view(n, k, h*w).max(2)[0].view(-1)
    
    loss = F.binary_cross_entropy(probs_max[1:], (counts.squeeze()!=0).float(), reduction="sum")
    loss += F.binary_cross_entropy(probs_max[:1], torch.ones(1).cuda(), reduction="sum")

    # Point Loss
    loss += F.nll_loss(probs_log, points, ignore_index=0, reduction="sum")

    for ann in annList:
        
        if ann["status"] == "SP":  
            scale = len(ann["gt_pointList"])          
            T = 1 - au.probs2splitMask_all(probs, ann["gt_pointList"])["background"]
            T = 1 - T*au.ann2mask(ann)["mask"]
            loss += scale * F.nll_loss(probs_log, torch.LongTensor(T).cuda(),
                        ignore_index=1,  reduction="elementwise_mean")

        if ann["status"] == "FP":
            T = 1 - au.ann2mask(ann)["mask"]            
            loss += F.nll_loss(probs_log, torch.LongTensor(T).cuda()[None],
                        ignore_index=1,  reduction="elementwise_mean")

    # print("n_points", len(annList), loss.item())
    return loss

def compute_lc_loss(counts, points, probs, probs_log):
    n,k,h,w = probs.size()

    with torch.no_grad():
        annList = au.probs2GtAnnList(probs, points)

    # IMAGE LOSS
    probs_max = probs.view(n, k, h*w).max(2)[0].view(-1)
    
    loss = F.binary_cross_entropy(probs_max[1:], (counts.squeeze()!=0).float(), reduction="sum")
    loss += F.binary_cross_entropy(probs_max[:1], torch.ones(1).cuda(), reduction="sum")

    # Point Loss
    loss += F.nll_loss(probs_log, points, ignore_index=0, reduction="sum")

    for ann in annList:
        
        if ann["status"] == "SP":  
            scale = len(ann["gt_pointList"])          
            T = 1 - au.probs2splitMask_all(probs, ann["gt_pointList"])["background"]
            T = 1 - T*au.ann2mask(ann)["mask"]
            loss += scale * F.nll_loss(probs_log, torch.LongTensor(T).cuda(),
                        ignore_index=1,  reduction="elementwise_mean")

        if ann["status"] == "FP":
            T = 1 - au.ann2mask(ann)["mask"]            
            loss += F.nll_loss(probs_log, torch.LongTensor(T).cuda()[None],
                        ignore_index=1,  reduction="elementwise_mean")

    # Global loss
    pointList = au.mask2pointList(points)["pointList"]
    if len(pointList) > 1:
        T = au.probs2splitMask_all(probs, pointList)["background"]
        loss += F.nll_loss(probs_log, torch.LongTensor(T).cuda(),
                            ignore_index=1,  reduction="elementwise_mean")
    # print("n_points", len(annList), loss.item())
    return loss

def get_blob_dict(model, batch, training=False): 
    blobs, probs = model.predict(batch, "blobs_probs", training)

    blobs = blobs.squeeze()
    probs = probs.squeeze()
    points = ms.t2n(batch["points"]).squeeze()

    if blobs.ndim == 2:
        blobs = blobs[None]


    blobList = []

    n_multi = 0
    n_single = 0
    n_fp = 0
    total_size = 0

    for l in range(blobs.shape[0]):
        class_blobs = blobs[l]
        points_mask = points == (l+1)
        # Intersecting
        blob_uniques, blob_counts = np.unique(class_blobs * (points_mask), return_counts=True)

        uniques = np.delete(np.unique(class_blobs), blob_uniques)

        


        for u in uniques:
            blobList += [{"class":l, "blob_id":u, "n_points":0, "size":0,
                         "pointsList":[]}]
            n_fp += 1

        for i, u in enumerate(blob_uniques):
            if u == 0:
                continue

            pointsList = []
            blob_ind = class_blobs==u

            locs = np.where(blob_ind * (points_mask))

            for j in range(locs[0].shape[0]):
                pointsList += [{"y":locs[0][j], "x":locs[1][j]}]
            
            assert len(pointsList) == blob_counts[i]

            if blob_counts[i] == 1:
                n_single += 1

            else:
                n_multi += 1

            size = blob_ind.sum()
            total_size += size

            blobList += [{"class":l, "size":size, 
                          "blob_id":u, "n_points":blob_counts[i],
                          "pointsList":pointsList}]

    blob_dict = {"blobs":blobs, "probs":probs, "blobList":blobList, 
                 "n_fp":n_fp, 
                 "n_single":n_single,
                 "n_multi":n_multi,
                 "total_size":total_size}

    return blob_dict


