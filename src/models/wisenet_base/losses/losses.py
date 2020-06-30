import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn

import numpy as np
from sklearn.feature_extraction.image import img_to_graph, _compute_gradient_3d, _make_edges_3d
from skimage.segmentation import find_boundaries
import pydensecrf.densecrf as dcrf
from losses import splits as sp
from skimage.segmentation import slic_superpixels
import misc as ms
from scipy import sparse
from skimage.segmentation import mark_boundaries
from sklearn.feature_extraction.image import img_to_graph
# from . import methods
from skimage.morphology import label
from losses import helpers
from core import blobs_utils as bu
from sklearn.feature_extraction.image import grid_to_graph
# from models.helpers import proposals as prp
from skimage.segmentation import slic
# from core import proposals as prp
from pycocotools import mask as maskUtils
import torchvision
from skimage.segmentation import boundaries
from scipy import ndimage
from skimage.segmentation import find_boundaries
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from scipy.ndimage import distance_transform_cdt

import models
from losses import affinity_losses

def AffinityLoss(model, batch, visualize=False):
    return affinity_losses.affinity_loss(model, batch)


def OneHeadLoss_debug(model, batch, visualize=False):
    return OneHeadLoss_tmp(model, batch, visualize=False)


# -------------------- MAIN LOSS

def lcfcnRegularizedLoss(model, batch, visualize=False):
    
    model.train()
    N =  batch["images"].size(0)
    assert N == 1
    
    blob_dict = helpers.get_blob_dict(model, batch, training=True)
    images = batch["images"].cuda()
    points = batch["points"].cuda()
    counts = batch["counts"].cuda()

    O = model(images)["cam"]
    pred_counts = model(images)["count"]
    new_counts =  torch.cat([torch.LongTensor([[1]]).cuda(), counts], 1).float()
    count_loss = F.l1_loss(pred_counts, new_counts)
    S = F.softmax(O, 1)
    S_log = F.log_softmax(O, 1)

    fp_loss = 0.
    split_loss = 0.
    point_loss = 0.
    
    # IMAGE LOSS
    image_loss = helpers.compute_image_loss(S, counts)
    image_loss += helpers.compute_bg_loss(S, counts)

    # POINT LOSS
    point_loss = F.nll_loss(S_log, points, 
                   ignore_index=0, size_average=False)
    
    # FP loss
    if blob_dict["n_fp"] > 0:
        fp_loss = helpers.compute_fp_loss(S_log, blob_dict)

    # Split loss
    if blob_dict["n_multi"] > 0:
        split_loss = helpers.compute_split_loss(S_log, S, points, blob_dict, split_mode="water")

    # Global loss
    # split_loss += helpers.compute_boundary_loss(S_log, S, points, counts, add_bg=True)

    return (count_loss + image_loss + point_loss + fp_loss + split_loss) / N


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

def lcfcnBLoss(model, batch, visualize=False):
    
    model.train()
    N =  batch["images"].size(0)
    assert N == 1

    blob_dict = helpers.get_blob_dict(model, batch, training=True)
    images = batch["images"].cuda()
    points = batch["points"].cuda()
    counts = batch["counts"].cuda()

    O = model(images)
    S = F.softmax(O, 1)
    S_log = F.log_softmax(O, 1)

    fp_loss = 0.
    split_loss = 0.
    point_loss = 0.
    
    # IMAGE LOSS
    image_loss = helpers.compute_image_loss(S, counts)
    image_loss += helpers.compute_bg_loss(S, counts)

    # POINT LOSS
    point_loss = F.nll_loss(S_log, points, 
                   ignore_index=0, size_average=False)
    
    # FP loss
    if blob_dict["n_fp"] > 0:
        fp_loss = helpers.compute_fp_loss(S_log, blob_dict)

    # Split loss
    if blob_dict["n_multi"] > 0:
        split_loss = helpers.compute_split_loss(S_log, S, points, blob_dict, split_mode="water")

    # Global loss
    split_loss += helpers.compute_boundary_loss(S_log, S, points, counts, add_bg=True)

    return (image_loss + point_loss + fp_loss + split_loss) / N

def lcfcnLoss(model, batch, visualize=False):
    
    model.train()
    N =  batch["images"].size(0)
    assert N == 1

    blob_dict = helpers.get_blob_dict(model, batch, training=True)
    images = batch["images"].cuda()
    points = batch["points"].cuda()
    counts = batch["counts"].cuda()

    O = model(images)
    S = F.softmax(O, 1)
    S_log = F.log_softmax(O, 1)

    fp_loss = 0.
    split_loss = 0.
    point_loss = 0.
    
    # IMAGE LOSS
    image_loss = helpers.compute_image_loss(S, counts)
    image_loss += helpers.compute_bg_loss(S, counts)

    # POINT LOSS
    point_loss = F.nll_loss(S_log, points, 
                   ignore_index=0, size_average=False)
    
    # FP loss
    if blob_dict["n_fp"] > 0:
        fp_loss = helpers.compute_fp_loss(S_log, blob_dict)

    # Split loss
    if blob_dict["n_multi"] > 0:
        split_loss = helpers.compute_split_loss(S_log, S, points, blob_dict, split_mode="water")

    # Global loss
    # split_loss += helpers.compute_boundary_loss(S_log, S, points, counts, add_bg=True)

    return (image_loss + point_loss + fp_loss + split_loss) / N

def blob_loss(model, batch, 
              enable_point_loss=True,
              enable_split_loss=True,
              enable_fp_loss=True,
              split_mode="line", 
              loss_list=(None,), visualize=False):
    
    model.train()
    N =  batch["images"].size(0)
    assert N == 1

    blob_dict = helpers.get_blob_dict(model, batch, training=True)
    # blobs = blob_dict["blobs"]
    # blob_list = blob_dict["blobList"]
    # put variables in cuda
    images = batch["images"].cuda()
    points = batch["points"].cuda()
    counts = batch["counts"].cuda()

    O = model(images)
    S = F.softmax(O, 1)
    S_log = F.log_softmax(O, 1)

    fp_loss = 0.
    split_loss = 0.
    point_loss = 0.
    
    # IMAGE LOSS

    image_loss = helpers.compute_image_loss(S, counts)
    image_loss += helpers.compute_bg_loss(S, counts)

    # POINT LOSS
    if "sp_points" in loss_list:
        point_loss = helpers.compute_sp_loss(batch, S_log)
        point_loss += F.nll_loss(S_log, points, 
                       ignore_index=0, size_average=False)


    elif "recursive_blob_points" in loss_list:
        point_loss = helpers.compute_recursive_blob_loss(batch, S_log, blob_dict)
        point_loss += F.nll_loss(S_log, points, 
                       ignore_index=0, size_average=False)


    elif "fixed_recursive" in loss_list:

        point_loss = helpers.compute_fixed_recursive_loss(model, batch, 
            S_log)
        point_loss += F.nll_loss( S_log, points, 
                       ignore_index=0, size_average=False)
    else:
        point_loss = F.nll_loss(S_log, points, 
                       ignore_index=0, size_average=False)
    # FP loss
    if blob_dict["n_fp"] > 0:
        fp_loss = helpers.compute_fp_loss(S_log, blob_dict)

    # Split loss
    if blob_dict["n_multi"] > 0:
        split_loss = helpers.compute_split_loss(S_log, S, points, blob_dict, split_mode=split_mode)

    # Global loss
    if "water_boundary" in loss_list:  
        split_loss += helpers.compute_boundary_loss(S_log, S, points, counts, add_bg=True)

    return (image_loss + point_loss + fp_loss + split_loss) / N


from sklearn.utils import shuffle

from skimage import transform
from torchvision import transforms
# from core import annList_methods as am
# from core import score_functions as sf
import ann_utils as au
import numpy as np
def GAMLoss(model, batch, **options):
    images = batch["images"].cuda()
    counts = batch["counts"].cuda()
    pred_dict = model(images)

    # criterion_count = nn.SmoothL1Loss().cuda();
    # criterion_cam = nn.L1Loss().cuda();
    loss = F.smooth_l1_loss(pred_dict["count"], counts.float())
    return loss

def GAMLoss2(model, batch, **options):
    images = batch["images"].cuda()
    counts = batch["counts"].cuda()
    pred_dict = model(images)

    # criterion_count = nn.SmoothL1Loss().cuda();
    # criterion_cam = nn.L1Loss().cuda();
    loss = F.smooth_l1_loss(pred_dict["count"], counts.float())

    loss += F.smooth_l1_loss(pred_dict["cam"].sum()[None], counts.float().squeeze())
    return loss


def GAMDiscOnly(model, batch, **options):
    images = batch["images"].cuda()
    counts = batch["counts"].cuda()
    pred_dict = model(images)
    loss = F.l1_loss(pred_dict["mask"].squeeze(), pred_dict["density"].cuda(), reduction="sum" )
    # loss += F.l1_loss(pred_dict["mask"].sum(), pred_dict["density"].sum().cuda() )
    # Sample noise and generate fake data

    # if np.random.rand() < 0.75:
    #     # Freeze disc
    #     for param in model.discriminator.parameters():
    #         param.requires_grad = False

    #     loss = F.smooth_l1_loss(pred_dict["count"], counts.float())

    #     loss += F.binary_cross_entropy(torch.sigmoid(model.discriminator(pred_dict["cam"][None,None])), 
    #                                   torch.FloatTensor([1]).cuda())

    #     # loss = F.l1_loss(pred_dict["cam"], pred_dict["density"].cuda() )
    #     # loss += F.smooth_l1_loss(pred_dict["cam"].sum()[None],counts.sum().float().squeeze())
    #     # loss += F.smooth_l1_loss(pred_dict["cam"].sum()[None], pred_dict["density"].sum().float().squeeze().cuda())
    #     print("gen loss:", loss.item())

    # else:
    #     # Unfreeze disc
    #     for param in model.discriminator.parameters():
    #         param.requires_grad = True

    #     prediction = model.discriminator(pred_dict["density"][None,None].cuda())
    #     loss = F.binary_cross_entropy(torch.sigmoid(prediction), torch.FloatTensor([1]).cuda())

    #     prediction = model.discriminator(pred_dict["cam"].detach()[None,None])
    #     loss += F.binary_cross_entropy(torch.sigmoid(prediction), torch.FloatTensor([0]).cuda())

    #     print("disc loss:", loss.item())

    return loss

from losses import mask_rcnn_losses

def MaskRCNNLoss_sm(model, batch, visualize=False):
    return mask_rcnn_losses.MaskRCNNLoss(model, batch)

def wiseaffinity_loss(model, batch):
    return affinity_losses.wiseaffinity_loss(model, batch)
def MaskRCNNLoss_gt(model, batch, visualize=False):
    return mask_rcnn_losses.MaskRCNNLoss(model, batch, true_annList=True)

def MaskRCNNLoss_prm(model, batch, visualize=False):
    return mask_rcnn_losses.MaskRCNNLoss(model, batch, prm_points=True)


def TripletLoss(model, batch, visualize=False, fp_loss=True, sim_loss=False):
    return pairwise_losses.compute_triplet_loss(model, batch)

def OneHeadLoss_tmp(model, batch, visualize=False):
    n,c,h,w = batch["images"].shape

    model.train()
    O_dict = model(batch["images"].cuda())
    embedding_mask = O_dict["embedding_mask"]

    loss = helpers.compute_metric_loss_mean(embedding_mask, batch)

    return loss

from losses import pairwise_losses

def OneHeadL1RBFLoss(model, batch, visualize=False):
    return OneHeadLoss(model, batch, visualize)


def OneHeadRBFLoss(model, batch, visualize=False):
    return OneHeadLoss(model, batch, visualize)

def OneHeadRBFLoss_noFP(model, batch, visualize=False):
    return OneHeadLoss(model, batch, visualize, fp_loss=False)

def OneHeadRBFLoss_withSim_noFP(model, batch, visualize=False):
    return OneHeadLoss(model, batch, visualize, sim_loss=True, fp_loss=False)

def OneHeadRBFLoss_multiproposals_noFP(model, batch, visualize=False):
    return OneHeadLoss(model, batch, visualize, multi_proposals=True, fp_loss=False)

def OneHeadRBFLoss_multiproposals(model, batch, visualize=False):
    return OneHeadLoss(model, batch, visualize, multi_proposals=True)

def OneHeadRBFLoss_withSim(model, batch, visualize=False):
    return OneHeadLoss(model, batch, visualize, sim_loss=True)

def OneHeadRBFLoss_random(model, batch, visualize=False):
    n,c,h,w = batch["images"].shape

    model.train()
    O_dict = model(batch["images"].cuda())
    embedding_mask = O_dict["embedding_mask"]

    loss = pairwise_losses.compute_pairwise_random_loss(embedding_mask, batch, similarity_function=model.similarity_function)

    return loss

def OneHeadL1SumLoss(model, batch, visualize=False):
    return OneHeadLoss(model, batch, visualize)

def OneHeadSumLoss(model, batch, visualize=False):
    return OneHeadLoss(model, batch, visualize)

def OneHeadLoss(model, batch, visualize=False, fp_loss=True, sim_loss=False, multi_proposals=False):
    n,c,h,w = batch["images"].shape

    model.train()
    
    O_dict = model(batch["images"].cuda())
    # model.PRM.classifier[0].weight.grad
    embedding_mask = O_dict["embedding_mask"]

    loss = pairwise_losses.compute_pairwise_loss(embedding_mask, batch, similarity_function=model.similarity_function,
        fp_loss=fp_loss, sim_loss=sim_loss, multi_proposals=multi_proposals)

    return loss





def PRMLoss(model, batch, visualize=False):
    n,c,h,w = batch["images"].shape

    model.train()
    O = model(batch["images"].cuda())
    loss = F.multilabel_soft_margin_loss(O, (batch["counts"].cuda()>0).float())

    return loss

def CAMLoss(model, batch, visualize=False):
    n,c,h,w = batch["images"].shape

    model.train()

    O = model(batch["images"].cuda())
    T = (batch["counts"]>0).float().cuda()
    loss = F.multilabel_soft_margin_loss(O, T)

    return loss

def AFFLoss(model, batch, visualize=False):
    import ipdb; ipdb.set_trace()  # breakpoint 914c5be2 //
    model.train()
    aff = model.forward(batch["images"].cuda())

    bg_label = pack[1][0].cuda(non_blocking=True)
    fg_label = pack[1][1].cuda(non_blocking=True)
    neg_label = pack[1][2].cuda(non_blocking=True)

    bg_count = torch.sum(bg_label) + 1e-5
    fg_count = torch.sum(fg_label) + 1e-5
    neg_count = torch.sum(neg_label) + 1e-5

    bg_loss = torch.sum(- bg_label * torch.log(aff + 1e-5)) / bg_count
    fg_loss = torch.sum(- fg_label * torch.log(aff + 1e-5)) / fg_count
    
    neg_loss = torch.sum(- neg_label * torch.log(1. + 1e-5 - aff)) / neg_count

    loss = bg_loss/4 + fg_loss/4 + neg_loss/2

    return loss 

def OneHeadRandomLoss(model, batch, visualize=False):
    n,c,h,w = batch["images"].shape

    model.train()
    O_dict = model(batch["images"].cuda())
    embedding_mask = O_dict["embedding_mask"]

    loss = helpers.compute_metric_loss_mean(embedding_mask, batch, random_proposal=True)

    return loss

def OneHeadLocLoss(model, batch, visualize=False):
    n,c,h,w = batch["images"].shape

    model.train()
    O_dict = model(batch["images"].cuda())
    O = O_dict["embedding_mask"]
    n,c,h,w = O.shape

    G = np.mgrid[:h,:w].astype(float)
    G[0] = G[0] / float(h)
    G[1] = G[1] / float(w)
    G = torch.FloatTensor(G).cuda()

    O = torch.cat([O, G[None]], dim=1)
    
    loss = helpers.compute_metric_loss_mean(O, batch)

    return loss


def WeaklyLCFCN_Loss(model, batch, visualize=False):
    n,c,h,w = batch["images"].shape
    fname = "{}/gam_{}.pkl".format(batch["path"][0], batch["index"].item()) 
    
    points = torch.LongTensor(ms.load_pkl(fname)).cuda()
    counts = batch["counts"].cuda()

    model.train()
    blob_mask = model(batch["images"].cuda())
    
    probs = F.softmax(blob_mask, 1)
    probs_log = F.log_softmax(blob_mask, 1)

  
    loss = helpers.compute_lc_loss(counts, points[None], probs, probs_log)

    return loss



def TwoHeadLoss_freeze(model, batch, visualize=False):
    n,c,h,w = batch["images"].shape
    import ipdb; ipdb.set_trace()  # breakpoint c0cecefc //
    
    points = batch["points"].cuda()
    counts = batch["counts"].cuda()

    model.train()
    O_dict = model(batch["images"].cuda())
    embedding_mask = O_dict["embedding_mask"]
    blob_mask = O_dict["blob_mask"]
    
    probs = F.softmax(blob_mask, 1)
    probs_log = F.log_softmax(blob_mask, 1)

    em = embedding_mask.cpu()
    pc = probs.cpu()
    loss = scale1 * helpers.compute_metric_loss_mean(embedding_mask, batch)

    return loss

def TwoHeadLoss_base(model, batch, scale1, scale2, visualize=False):
    n,c,h,w = batch["images"].shape
    points = batch["points"].cuda()
    counts = batch["counts"].cuda()

    model.train()
    O_dict = model(batch["images"].cuda())
    embedding_mask = O_dict["embedding_mask"]
    blob_mask = O_dict["blob_mask"]
    
    probs = F.softmax(blob_mask, 1)
    probs_log = F.log_softmax(blob_mask, 1)

    em = embedding_mask.cpu()
    pc = probs.cpu()
    loss = scale1 * helpers.compute_metric_loss_mean(embedding_mask, batch)
    loss += scale2 * helpers.compute_lc_loss(counts, points, probs, probs_log)

    return loss


def TwoHeadLoss_sum(model, batch, scale1=0.1, scale2=0.9, visualize=False):
    n,c,h,w = batch["images"].shape
    points = batch["points"].cuda()
    counts = batch["counts"].cuda()

    model.train()
    O_dict = model(batch["images"].cuda())
    embedding_mask = O_dict["embedding_mask"]
    blob_mask = O_dict["blob_mask"]
    
    probs = F.softmax(blob_mask, 1)
    probs_log = F.log_softmax(blob_mask, 1)

    loss = scale1 * helpers.compute_metric_loss_sum(embedding_mask, batch)
    loss += scale2 * helpers.compute_lc_loss(counts, points, probs, probs_log)

    return loss


def TwoHeadLoss(model, batch, visualize=False):

    n,c,h,w = batch["images"].shape
    points = batch["points"].cuda()
    counts = batch["counts"].cuda()

    model.train()
    O_dict = model(batch["images"].cuda())
    embedding_mask = O_dict["embedding_mask"]
    blob_mask = O_dict["blob_mask"]
    
    probs = F.softmax(blob_mask, 1)
    probs_log = F.log_softmax(blob_mask, 1)

        
    loss = model.options["scale1"] * helpers.compute_metric_loss_mean(embedding_mask, batch)
    loss += model.options["scale2"] * helpers.compute_lc_loss(counts, points, probs, probs_log)

    return loss

def TwoHeadLoss_1_9(model, batch, visualize=False):
    return TwoHeadLoss_base(model, batch, 0.1, 0.9, visualize=False)

def TwoHeadLoss_2_8(model, batch, visualize=False):
    return TwoHeadLoss_base(model, batch, 0.2, 0.8, visualize=False)

def TwoHeadLoss_3_7(model, batch, visualize=False):
    return TwoHeadLoss_base(model, batch, 0.3, 0.7, visualize=False)

def TwoHeadLoss_4_6(model, batch, visualize=False):
    return TwoHeadLoss_base(model, batch, 0.4, 0.6, visualize=False)

def TwoHeadLoss_5_5(model, batch, visualize=False):
    return TwoHeadLoss_base(model, batch, 0.5, 0.5, visualize=False)

def TwoHeadLoss_6_4(model, batch, visualize=False):
    return TwoHeadLoss_base(model, batch, 0.6, 0.4, visualize=False)

def TwoHeadLoss_7_3(model, batch, visualize=False):
    return TwoHeadLoss_base(model, batch, 0.7, 0.3, visualize=False)

def TwoHeadLoss_8_2(model, batch, visualize=False):
    return TwoHeadLoss_base(model, batch, 0.8, 0.2, visualize=False)

def TwoHeadLoss_9_1(model, batch, visualize=False):
    return TwoHeadLoss_base(model, batch, 0.9, 0.1, visualize=False)

def similarity_loss_old(model, batch, visualize=False):

    n,c,h,w = batch["images"].shape
    model.train()

    images = batch["images"].cuda()
    points = batch["points"].cuda()
    # counts = batch["counts"].cuda()
    
    pointList = bu.mask2pointList(points)["pointList"]

    loss = torch.tensor(0.).cuda()
    if len(pointList) == 0:
        return loss
    

    with torch.no_grad():
        propDict = bu.pointList2propDict(pointList, batch, thresh=0.5)
        background = propDict["background"]

    propDict = propDict["propDict"]

    # Segmenter
    O = model(images)

    n,c,h,w = O.shape

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
        
        ap = - torch.log(bu.log_pairwise(f_A, fg_P)) 
        loss += ap.mean()

        # Get Negatives
        if n_seeds > 1:
            N_ind = [j for j in range(n_seeds) if j != i]
            f_N = fg_seeds[:,:,N_ind]
            an = - torch.log(1. - bu.log_pairwise(f_A, f_N)) 
            loss += an.mean()

    # Extract background seeds
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
    ap = - torch.log(bu.log_pairwise(f_A[:,:,None], f_P[:,:,:,None])) 
    an = - torch.log(1. - bu.log_pairwise(f_A[:,:,None], fg_seeds[:,:,:,None])) 

    loss += ap.mean()
    loss += an.mean()

    if batch["dataset"][0] == "cityscapes":        
        n_max = 6
    else:
        n_max = 6

    if f_A.shape[2] < n_max:
        with torch.no_grad():
            diff = bu.log_pairwise(O.view(1,64,-1)[:,:,:,None], 
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

            ap = - torch.log(bu.log_pairwise(f_A[:,:,None], f_P[:,:,:,None])) 
            an = - torch.log(1. - bu.log_pairwise(f_P[:,:,None], fg_seeds[:,:,:,None])) 

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



def matching_proposal6(model, batch, visualize=False, matching_method='dice'):
    model.train()

    images = batch["images"].cuda()
    points = batch["points"].cuda()
    counts = batch["counts"].cuda()
    
    # Segmenter
    O = model(images)
    probs = F.softmax(O, 1)
    probs_log = F.log_softmax(O, 1)

    loss = torch.tensor(0.).cuda()
    
    loss += helpers.compute_lc_loss(counts, points, probs, probs_log)
    loss += helpers.compute_gap_loss(probs_log, batch)
    loss += helpers.compute_similarity_loss(probs_log, batch)

    # Counter
    O = model.counter(images)
    probs = F.softmax(O, 1)
    probs_log = F.log_softmax(O, 1)
    
    loss += helpers.compute_lc_loss(counts, points, probs, probs_log)

    return loss / 2

def matching_proposal(model, batch, visualize=False, matching_method='dice'):
    model.train()

    images = batch["images"].cuda()
    points = batch["points"].cuda()
    counts = batch["counts"].cuda()
    if 1:
        O = model(images)
        probs = F.softmax(O, 1)
        probs_log = F.log_softmax(O, 1)

    loss = torch.tensor(0.).cuda()
    
    loss += helpers.compute_lc_loss(counts, points, probs, probs_log)

    with torch.no_grad():
        # points_annList = bu.points2annList(points)["annList"]
        annList = bu.probs2GtAnnList(probs, points)


    # bg_mask = 
    for ann in annList:
        if ann["status"] != "TP":
            continue

        fg_mask = bu.ann2proposal(ann, batch, matching_method=matching_method)

        if fg_mask is None:
            continue
        if visualize:
            ms.images(fg_mask)
        T = 1 - find_boundaries(fg_mask)
        T = torch.LongTensor(T).cuda()[None]

        loss += F.nll_loss(probs_log, T, ignore_index=1, reduction="elementwise_mean")


        fg_mask = fg_mask*ann["category_id"]
        fg_mask = torch.LongTensor(fg_mask).cuda()[None]
        loss += F.nll_loss(probs_log, fg_mask, ignore_index=0, reduction="elementwise_mean")
    
    return loss / 3

def region_growing_loss(model, batch, visualize=False):
    model.train()

    images = batch["images"].cuda()
    # points = batch["points"].cuda()
    counts = batch["counts"].cuda()

    O = model(images)
    probs = F.softmax(O, 1)
    probs_log = F.log_softmax(O, 1)


    loss = torch.tensor(0.).cuda()
    loss += helpers.compute_image_loss(probs, counts)

    # Get pointList
    pointList = bu.mask2pointList(batch["points"])["pointList"]
    if len(pointList) == 0:
        return loss 

    propList = prp.Sharp_class(batch["name"])

    propDict = bu.pointList2propDict(pointList, propList, 0.5)

    # ms.images(propDict["foreground"])

    # Get split mask


    

    splitList = bu.probs2splitMask_all(probs, pointList)

    # ms.images(splitList["background"])

    split_background = splitList["background"]
    split_background = torch.LongTensor(split_background).cuda()

    loss += F.nll_loss(probs_log, split_background,
                        ignore_index=1, size_average=True) * len(pointList)

    for p in pointList:
        point_target =  torch.ones(1).long().cuda()*int(p["category_id"])
        loss += F.nll_loss(probs_log[:, :, p["y"],
                                     p["x"]], 
                            point_target, 
                            size_average=False)

    for propList in propDict["propDict"]:
        ann_mask = bu.annList2mask(propList["annList"])["mask"]
        if ann_mask is None:
            continue
        mask = (ann_mask[None]>0).astype(int)
        # ms.images(bu.annList2mask(propDict["propDict"][0]["annList"])["mask"])
        category_id = propList["category_id"]

        foreground = category_id*torch.LongTensor(mask).cuda()
        foreground = foreground*split_background

        target = torch.LongTensor(mask*category_id).cuda()
        # 
        if visualize:
            ms.images(batch["images"], foreground, win="foreground", denorm=1)
        # annLoss = F.nll_loss(probs_log, target,
                            # ignore_index=0,reduce=False)*probs[:,category_id]
        # probs_sum = probs[:,category_id][target!=0].sum().detach()
        # loss += annLoss.sum()/ probs_sum

        loss +=  F.nll_loss(probs_log, target,
                            ignore_index=0,size_average=True)
    background = 1 - propDict["background"]
    if visualize:
            ms.images(batch["images"], background, win="background", denorm=1)

    bg_target = torch.LongTensor(background).cuda()
    # bg_loss = F.nll_loss(probs_log, 
    #                    bg_target,
    #                    ignore_index=1, reduce=False)*probs[:,0]
    # probs_sum = probs[:,0][bg_target!=1].sum().detach()
    # loss += bg_loss.sum()/ probs_sum

    bg_loss = F.nll_loss(probs_log, 
                       bg_target,
                       ignore_index=1, size_average=True)
    loss += bg_loss


    # For all proposals do average pooling
    # for p in pointList:
    #     point_propList = bu.point2propList(p, propList)
    #     ms.images(bu.annList2mask(propDict[0]["annList"]))

    # Background - untouched proposals
    # ms.images(propDict["foreground"].astype(int))


    return loss / (10+1+1+1)

def glance_loss(model, batch, visualize=False):
    model.train()

    points = batch["points"]

    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    transformer = transforms.Compose([
        transforms.ToPILImage(),
             transforms.Resize((224,224)),
             transforms.ToTensor(),
             transforms.Normalize(*mean_std)])



    N =  batch["images"].size(0)
    assert N == 1
    pointList = bu.mask2pointList(points)["pointList"]

    annList = prp.Sharp_class(batch["name"])
    images = ms.f2l(ms.denormalize(batch["images"])).squeeze()
    images = (images*255).astype("uint8")
    # point_mask = bu.pointList2mask(pointList[:1])

    loss = torch.tensor(0.).cuda()
    ann_idList = []

    for p in pointList:
        point_annList = []
        cropList = []
        category_id = p["category_id"]

        ann_count = 0
        for ai, ann in enumerate(shuffle(annList)):
            if ann_count == 16:
                break
            if ann["score"] < 0.5:
                continue


            # if bu.isPointInBBox(p, ann):

            #     bbox = bu.ann2bbox(ann) !=0 
            if ann["mask"][p["y"], p["x"]] !=0: 
                
                bbox = bu.ann2bbox(ann)
                # point_mask = bu.pointList2mask([p])
                # ms.images(batch["images"], point_mask,enlarge=1,denorm=1,win="points")
                # ms.images(batch["images"], bbox["mask"],denorm=1)
                
                x, y, xe, ye = map(int, bbox["shape"])
                cropped = transformer(images[y:ye, x:xe])
                cropList += [cropped]
                point_annList += [ai]

                ann_count += 1
               
                # print("good")
            else:
                ann_idList += [ai]

        if len(cropList) != 0:
            img_batch = torch.stack(cropList)
            with torch.no_grad():
                O = model(img_batch.cuda())
                O_cat = O[:, category_id]
            ind_max = O_cat.argmax()

            S_log = F.log_softmax(model(img_batch[[ind_max]].cuda()[None]), dim=1)
            loss += F.nll_loss(S_log, 
                              torch.ones(1).long().cuda()*int(category_id))
            #                        size_average=True)
        # ann = annList[point_annList[S_log_cat.argmin().item()]] 


        # bbox = bu.ann2bbox(ann) 
        # ms.images(batch["images"], bbox["mask"],denorm=1)

    cropList = []
    for i, ai in enumerate(shuffle(ann_idList)):
        if i == 16:
            break
        ann = annList[ai]
        bbox = bu.ann2bbox(ann) 
        x, y, xe, ye = map(int, bbox["shape"])

        sub_img = images[y:ye, x:xe]
        if sub_img.size == 0:
            continue
        cropped = transformer(sub_img)
        cropList += [cropped]

    if len(cropList) != 0:
        img_batch = torch.stack(cropList)
        S_log = F.log_softmax(model(img_batch.cuda()), dim=1)
        
        loss += F.nll_loss(S_log, 
                          torch.zeros(S_log.shape[0]).long().cuda()) / 32.


    return loss



def proposal_pretrained_loss(model, batch, visualize=False):
    model.train()

    N =  batch["images"].size(0)
    assert N == 1
    blob_dict = helpers.get_blob_dict(model, batch, training=True)
    sharp_proposals = prp.Sharp_class(batch["name"])
    # probs = ms.t2n(F.softmax(O, dim=1))
    

    
    images = batch["images"].cuda()
    # points = batch["points"].cuda()

    with torch.no_grad():
        O = model.base_model(images)
        S = F.softmax(O, 1)
        _, counts = bu.get_blobs(S, return_counts=True)
        points = bu.blobs2points(ms.t2n(S))[None]

    O = model(images)
    S = F.softmax(O, 1)
    S_log = F.log_softmax(O, 1)


    pointInd = np.where(points.squeeze())
    n_points = pointInd[0].size

    loss = helpers.compute_image_loss(S, 
         torch.LongTensor(counts).cuda())

    for p in range(n_points):
        p_y, p_x = pointInd[0][p], pointInd[1][p]
        point_category = points[0, p_y,p_x].item()
        point_mask = np.zeros(points.shape, int)
        point_mask[:, p_y, p_x] = 1

        min_score = 10000
        best_k = None

        assert point_category != 0
        for k in range(len(sharp_proposals)):
            proposal_dict = sharp_proposals[k]
            if proposal_dict["score"] < 0.5:
                continue
            proposal_mask =  proposal_dict["mask"]

            if proposal_mask[p_y,p_x] == 0:
                continue

            if (proposal_mask * 
                (points==point_category)).sum().item() > 1:
                continue

            # if (np.unique(label(proposal_mask))).size > 2:
            #     continue

            # box = maskUtils.toBbox(proposal_dict["segmentation"])
            # x, y, w, h = list(map(int, box))
            

            boundary = 1 - boundaries.find_boundaries(proposal_dict["mask"])
            boundary = torch.LongTensor(boundary).cuda()
            with torch.no_grad():
                proposal_mask[proposal_mask!=0] = point_category
                proposal_mask = torch.LongTensor(proposal_mask).cuda()

                score = F.nll_loss(S_log, boundary[None], ignore_index=1,
                        size_average=True).item()
                # score += F.nll_loss(S_log, proposal_mask[None], 
                #          ignore_index=0,
                #         size_average=True).item()
                # score = F.nll_loss(S_log, boundary[None], ignore_index=1,
                #         reduce=False).max().item()
                # score += F.nll_loss(S_log, proposal_mask[None], 
                #          ignore_index=0,
                #         reduce=False).max().item()
            
            if score < min_score:
                min_score = score
                best_k = k

            # print("objectness:", proposal_dict["score"])
            # ms.images(batch["original"][:,:,y:y+h,x:x+w],denorm=1, win="box")
            # ms.images(batch["original"], proposal_dict["mask"], win="proposal")
            # ms.images(batch["original"], boundary, win="boundary")
            # ms.images(batch["original"], point_mask, win="points")
        if best_k is not None:
            proposal_dict = sharp_proposals[best_k]
            proposal_mask = proposal_dict["mask"]
            ms.images(batch["images"], 
                      proposal_dict["mask"], denorm=1,
                      win="proposal_{}".format(p))
            # print("objectness:", proposal_dict["score"])
            # import ipdb; ipdb.set_trace()  # breakpoint e229c2d5 //

            boundary = 1 - boundaries.find_boundaries(proposal_dict["mask"])
            boundary = torch.LongTensor(boundary).cuda()

            proposal_mask[proposal_mask!=0] = point_category
            proposal_mask = torch.LongTensor(proposal_mask).cuda()

            loss += F.nll_loss(S_log, boundary[None], ignore_index=1,
                               size_average=True)
            loss += F.nll_loss(S_log, proposal_mask[None], ignore_index=0,
                               size_average=True)


    # Points
    loss += F.nll_loss(S_log, torch.LongTensor(points).cuda(), 
                   ignore_index=0, size_average=False)

    # Split loss
    if blob_dict["n_multi"] > 0:
        loss += helpers.compute_split_loss(S_log, S, points,
         blob_dict, split_mode="water")


    # FP loss
    if blob_dict["n_fp"] > 0:
        loss += helpers.compute_fp_loss(S_log, blob_dict)

    return loss

def proposal_seg_loss(model, batch, visualize=False):
    model.train()

    N =  batch["images"].size(0)
    assert N == 1
    blob_dict = helpers.get_blob_dict(model, batch, training=True)
    sharp_proposals = prp.Sharp_class(batch["name"])
    points = batch["points"]

    pointInd = np.where(points.squeeze())
    n_points = pointInd[0].size

    images = batch["images"].cuda()
    points = batch["points"].cuda()
    counts = batch["counts"].cuda()

    O = model(images)
    S = F.softmax(O, 1)
    S_log = F.log_softmax(O, 1)
    from skimage.morphology import label
    loss = helpers.compute_image_loss(S, counts)
    for p in range(n_points):
        p_y, p_x = pointInd[0][p], pointInd[1][p]
        point_category = batch["points"][0, p_y,p_x].item()
        point_mask = np.zeros(batch["points"].shape, int)
        point_mask[:, p_y, p_x] = 1

        min_score = 10000
        best_k = None
        for k in range(len(sharp_proposals)):
            proposal_dict = sharp_proposals[k]
            if proposal_dict["score"] < 0.5:
                continue
            proposal_mask =  proposal_dict["mask"]

            if proposal_mask[p_y,p_x] == 0:
                continue

            if (proposal_mask * (batch["points"]==point_category)).sum().item() > 1:
                continue

            if (np.unique(label(proposal_mask))).size > 2:
                continue

            box = maskUtils.toBbox(proposal_dict["segmentation"])
            x, y, w, h = list(map(int, box))
            

            boundary = 1 - boundaries.find_boundaries(proposal_dict["mask"])
            boundary = torch.LongTensor(boundary).cuda()
            with torch.no_grad():
                score = F.nll_loss(S_log, boundary[None], ignore_index=1,
                        size_average=True).item()
                # score = F.nll_loss(S_log, boundary[None], ignore_index=1,
                #         reduce=False).max().item()
            
            if score < min_score:
                min_score = score
                best_k = k

            # print("objectness:", proposal_dict["score"])
            # ms.images(batch["original"][:,:,y:y+h,x:x+w],denorm=1, win="box")
            # ms.images(batch["original"], proposal_dict["mask"], win="proposal")
            # ms.images(batch["original"], boundary, win="boundary")
            # ms.images(batch["original"], point_mask, win="points")
        proposal_dict = sharp_proposals[best_k]
        proposal_mask = proposal_dict["mask"]
        ms.images(batch["images"], 
                  proposal_dict["mask"], denorm=1,
                  win="proposal_{}".format(p))
        # print("objectness:", proposal_dict["score"])
        # import ipdb; ipdb.set_trace()  # breakpoint e229c2d5 //

        boundary = 1 - boundaries.find_boundaries(proposal_dict["mask"])
        boundary = torch.LongTensor(boundary).cuda()

        proposal_mask[proposal_mask!=0] = point_category
        proposal_mask = torch.LongTensor(proposal_mask).cuda()

        loss += F.nll_loss(S_log, boundary[None], ignore_index=1,
                           size_average=True)
        loss += F.nll_loss(S_log, proposal_mask[None], ignore_index=0,
                           size_average=True)

    # Points
    loss += F.nll_loss(S_log, points, 
                   ignore_index=0, size_average=False)

    # Split loss
    if blob_dict["n_multi"] > 0:
        loss += helpers.compute_split_loss(S_log, S, points,
         blob_dict, split_mode="water")


    # FP loss
    if blob_dict["n_fp"] > 0:
        loss += helpers.compute_fp_loss(S_log, blob_dict)

    return loss

def proposal_loss(model, batch, visualize=False):
    model.train()
    N =  batch["images"].size(0)
    assert N == 1
    sharp_proposals = prp.Sharp_class(batch["name"])
    points = batch["points"]

    pointInd = np.where(points.squeeze())
    n_points = pointInd[0].size
    clf = torchvision.models.resnet50(pretrained=True).cuda()
    clf.eval()
    tr = torchvision.transforms.Compose([
           torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((224,224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
         ])

    lbl_dict =  ms.load_json("notes/imagenet_labels.json")

    for p in range(n_points):
        p_y, p_x = pointInd[0][p], pointInd[1][p]
        point_category = batch["points"][0, p_y,p_x].item()
        point_mask = np.zeros(batch["points"].shape, int)
        point_mask[:, p_y, p_x] = 1
        for k in range(len(sharp_proposals)):
            proposal_dict = sharp_proposals[k]
            if proposal_dict["score"] < 0.5:
                continue
            proposal_mask =  proposal_dict["mask"]

            if proposal_mask[p_y,p_x] == 0:
                continue

            box = maskUtils.toBbox(proposal_dict["segmentation"])
            x, y, w, h = list(map(int, box))
            
            
            org = proposal_mask*batch["original"]
            X = org[:,:,y:y+h,x:x+w]
            out = clf(tr(X[0])[None].cuda())
            scores = out.max(1)
            print("{1}: {0:.2f}".format(scores[0].item(),
                              lbl_dict[str(scores[1].item())][1]))

            
            # ms.images(batch["original"][:,:,y:y+h,x:x+w],denorm=1, win="box")
            ms.images(org, proposal_dict["mask"], win="proposal")
            # ms.images(batch["original"], point_mask, win="points")
            import ipdb; ipdb.set_trace()  # breakpoint 1556430e //

        import ipdb; ipdb.set_trace()  # breakpoint e229c2d5 //


    images = batch["images"].cuda()
    points = batch["points"].cuda()
    counts = batch["counts"].cuda()


    O = model(images.cuda())
    S = F.softmax(O, 1)
    
    # IMAGE LOSS

    image_loss = helpers.compute_image_loss(S, counts)
    return image_loss
    # ms.images(batch["images"], denorm=1)

def recursive_blob_loss(model, batch):
    return blob_loss(model, batch, split_mode="water", 
            loss_list=("recursive_blob_points", "water_boundary"))

def fixed_recursive_loss(model, batch):
    return blob_loss(model, batch, split_mode="water", 
            loss_list=("fixed_recursive", "water_boundary"))



def match_sharp_loss(model, batch):
    pass

def water_loss(model, batch):
    return blob_loss(model, batch, split_mode="water")


def sp_water_loss(model, batch):
    return blob_loss(model, batch, split_mode="water", loss_list=("sp_points", "water_boundary"))


def point_loss(model, batch, 
              split_mode="line", 
              loss_list=(None,)):
    
    return blob_loss(model, batch, 
              enable_point_loss=True,
              enable_split_loss=False,
              enable_fp_loss=False,
              split_mode="line", 
              loss_list=(None,))


def affinity_water_loss(model, batch):
    return blob_loss(model, batch, split_mode="water", loss_list=("affinity"))


def image_count(model, batch):
    model.train()
    N =  batch["images"].size(0)
    #assert N == 1

    # put variables in cuda
    images = batch["images"].cuda()
    counts = batch["counts"].cuda()

    O = model(images)
    S = F.softmax(O, 1)

    # IMAGE LOSS
    n,k,h,w = S.size()

    # GET TARGET
    ones = torch.ones(counts.size(0), 1).long().cuda()
    BgFgCounts = torch.cat([ones, counts], 1)
    Target = (BgFgCounts.view(n*k).view(-1) > 0).view(-1).float()

    # GET INPUT
    Smax = S.view(n, k, h*w).max(2)[0].view(-1)
    
    loss = F.binary_cross_entropy(Smax, Target, size_average=False)

    if batch["counts"].item() > 0:
        points = methods.watershed_clusters(model, batch).squeeze()
        batch["points"] = torch.from_numpy(points)[None]
        loss = water_loss(model, batch)
    else:
        loss = image_loss(model, batch)

    return loss / N
    


def blobcounter(model, batch):
    model.train()
    N =  batch["images"].size(0)
    #assert N == 1

    # put variables in cuda
    images = batch["images"].cuda()
    counts = batch["counts"].cuda()

    S = F.softmax(model(images), 1)

    # IMAGE LOSS
    n,k,h,w = S.size()

    # GET TARGET
    ones = torch.ones(counts.size(0), 1).long().cuda()
    BgFgCounts = torch.cat([ones, counts], 1)
    Target = (BgFgCounts.view(n*k).view(-1) > 0).view(-1).float()

    # GET INPUT
    Smax = S.view(n, k, h*w).max(2)[0].view(-1)
    
    loss = F.binary_cross_entropy(Smax, Target, size_average=False)

    blob_dict = methods.get_blob_dict(model, batch)
    blobs = blob_dict["blobs"]
    blob_list = blob_dict["blobList"]

    c_pred = model.counter(S.detach())
    c_truth = torch.FloatTensor([len(blob_list)]).cuda()

    loss += ((c_pred.view(-1) - c_truth)**2).sum()
    # n_trails = 2
    # for i in range(n_trails):
    #     cc = np.random.randint(1, 10)
    #     im, count = methods.generate_blobs(n=cc, l=images.shape[2:])
    #     loss += ((model.counter(torch.FloatTensor(im).cuda()[None,None]).view(-1) - 
    #                 count)**2).sum() / float(n_trails)

    for p in model.counter.parameters():
        p.requires_grad = False
    
    loss += ((model.counter(S).view(-1)  - counts.view(-1).float())**2).sum()
    
    for p in model.counter.parameters():
        p.requires_grad = True

    return loss / N


def expand_salient(model, batch):
    return blob_loss(model, batch, split_mode="water", 
                     loss_list=["expand_salient"])

def expand_constrain_sp(model, batch):
    return blob_loss(model, batch, split_mode="water", 
                     loss_list=["expand_sp", 
                     "constrain_sp"])

def expand_sp(model, batch):
    return blob_loss(model, batch, split_mode="water", 
                     loss_list=["expand_sp", 
                     "crf_constrain"])

def expand_sp_M(model, batch):
    return blob_loss(model, batch, split_mode="water", 
                     loss_list=["expand_sp_M", 
                     "crf_constrain"])

def expand_nei_constrain(model, batch):
    return blob_loss(model, batch, split_mode="water", 
                     loss_list=["expand_nei_fg", 
                     "crf_constrain"])

def expand_nei_constrain_B(model, batch):
    return blob_loss(model, batch, split_mode="water", 
                     loss_list=["expand_nei_fg", 
                     "crf_constrain",
                     "water_boundary"])

def expand_nei_loss(model, batch):
    return blob_loss(model, batch, split_mode="water", 
                     loss_list=["expand_nei_fg", 
                     "expand_nei_bg"])

def expand_loss(model, batch):
    return blob_loss(model, batch, split_mode="water", 
                     loss_list=["expand_fg","expand_bg"])

def least_squares(model, batch):
    model.train()
    
    # put variables in cuda
    images = batch["images"].cuda()
    counts = batch["counts"].cuda()
    pred = model(images)

    
    loss = F.mse_loss(pred, counts.float())
    return loss
# -------------------- Semantic
def seg_loss_A(model, batch):
    model.train()
    N =  batch["images"].size(0)
    # put variables in cuda
    images = batch["images"].cuda()
    labels = batch["labels"].cuda()

    loss_func = lambda s: F.cross_entropy(s, labels, size_average=False, 
                                  ignore_index=model.ignore_index) / N
    

    scores = model(images)
    loss = loss_func(scores)

    
    return loss

def gap_loss(model, batch):
    model.train()

    N = batch["images"].size(0)
    # put variables in cuda
    images = batch["images"].cuda()
    points = batch["points"].cuda()
    counts = batch["counts"].cuda()

    O = model(images)
    S = F.softmax(O, 1)
    n,k,h,w = S.size()

    S_log = F.log_softmax(O, 1)

    # IMAGE AND POINT LOSS
    # GET TARGET
    ones = torch.ones(counts.size(0), 1).long().cuda()
    BgFgCounts = torch.cat([ones, counts], 1)
    Target = (BgFgCounts.view(n*k).view(-1) > 0).view(-1).float()

    # GET INPUT
    Smax = S.view(n, k, h*w).mean(2)[0].view(-1)
    
    loss = F.binary_cross_entropy(Smax, Target, size_average=False)

    # POINT LOSS
    loss += F.nll_loss(S_log, points, 
                       ignore_index=0,
                       size_average=False)
    
    return loss / N

def wtp_loss(model, batch):
    model.train()

    N = batch["images"].size(0)
    # put variables in cuda
    images = batch["images"].cuda()
    points = batch["points"].cuda()
    counts = batch["counts"].cuda()

    O = model(images)
    S = F.softmax(O, 1)
    n,k,h,w = S.size()

    S_log = F.log_softmax(O, 1)

    # IMAGE AND POINT LOSS
    # GET TARGET
    ones = torch.ones(counts.size(0), 1).long().cuda()
    BgFgCounts = torch.cat([ones, counts], 1)
    Target = (BgFgCounts.view(n*k).view(-1) > 0).view(-1).float()

    # GET INPUT
    Smax = S.view(n, k, h*w).max(2)[0].view(-1)
    
    loss = F.binary_cross_entropy(Smax, Target, size_average=False)

    # POINT LOSS
    loss += F.nll_loss(S_log, points, 
                       ignore_index=0,
                       size_average=False)
    
    return loss / N

def wtpb_loss(model, batch):
    model.train()
    N =  batch["images"].size(0)

    blob_dict = methods.get_blob_dict(model, batch)
    
    # put variables in cuda
    images = batch["images"].cuda()
    points = batch["points"].cuda()
    counts = batch["counts"].cuda()

    O = model(images)
    S = F.softmax(O, 1)
    S_log = F.log_softmax(O, 1)

    # IMAGE LOSS
    loss = methods.image_loss(S, counts)

    # POINT LOSS
    loss += F.nll_loss(S_log, points, 
                       ignore_index=0,
                       size_average=False)
    # FP loss
    if blob_dict["n_fp"] > 0:
        loss += fp_loss(S_log, blob_dict)

    # Split loss
    if blob_dict["n_multi"] > 0:
        #loss += water_split_loss(model, batch, S_log)
        loss += split_loss(S_log, S, points, blob_dict, split_mode="line")

    return loss

# -------------------- OTHERS

def crf_constrain_loss(model, batch, S_log):
    Q = methods.dense_crf(model, batch, binary=True)
    Q = 1. - find_boundaries(Q)
    n_colors = int(np.unique(Q).size)

    return F.nll_loss(S_log, ms.t2l(Q), 
                             ignore_index=1,
                             size_average=True)


def crf_boundary_loss(model, batch, S_log):
    Q = methods.dense_crf(model, batch)
    Q[Q > 0] = 1

    return F.nll_loss(S_log, ms.t2l(Q), 
                             ignore_index=0,
                             size_average=True)


def instance_loss_A(model, batch):
    return blob_loss(model, batch, split_mode="water", 
                     loss_list=["water_boundary",
                                "crf_boundary"])


def lc_loss(model, batch, visualize=False):
    model.train()

    images = batch["images"].cuda()
    points = batch["points"].cuda()
    counts = batch["counts"].cuda()
    if 1:
        O = model(images)
        probs = F.softmax(O, 1)
        probs_log = F.log_softmax(O, 1)


    loss = torch.tensor(0.).cuda()
    
    loss += helpers.compute_lc_loss(counts, points, probs, probs_log)

    return loss
def water_loss_B(model, batch, visualize=False):
    return blob_loss(model, batch, split_mode="water", 
                     loss_list=["water_boundary"])


def expand_loss_B(model, batch):
    return blob_loss(model, batch, split_mode="water", 
                     loss_list=["water_boundary","expand_global", "crf_constrain"])







# def fp_split_loss(model, batch, S, S_log, points, 
#                   fp_only):
#     blobs = model.predict(batch, "blobs")
#     split_list = methods.get_split_list(S, blobs, points, 
#                     rule="line")

#     T = np.ones(S_log[0,0].size())
#     loss = 0.

#     for sl in split_list:
#         if sl["n_points"] >= 1 and fp_only: 
#             continue

#         Target = np.ones(S_log[0,0].size()) * 255
#         Target[sl["ind"]] = 0
#         T[sl["ind"]] = 0
#         Target = torch.LongTensor(Target[None]).cuda()

#         scale = float(sl["n_points"] + 1.)
#         loss += scale * F.nll_loss(S_log, Target,
#                             ignore_index=255, 
#                             size_average=True)

#     return loss 








def ez_dense_crf(model, batch):
    image = ms.f2l((vis.denormalize(batch["images"])*255).astype("uint8"))[0].copy()
    probs = ms.t2n(model.predict(batch))[0].copy()
    blobs = ms.t2n(model.predict(batch,"blobs"))[0].copy()
    points = ms.t2n(batch["points"])
    ind = points[0] == 1

    probs[0,ind] = 0
    probs[1,ind] = 1


    if 1:
        h = image.shape[0]
        w = image.shape[1]
        c = points.sum()

        P = np.zeros((c, h, w))

        import ipdb; ipdb.set_trace()  # breakpoint cd158def //

        d = dcrf.DenseCRF2D(w, h, c)
        U = -np.log(P)
        U = U.reshape((c, -1))
        U = np.ascontiguousarray(U)
        image = np.ascontiguousarray(image)
  
        d.setUnaryEnergy(U)

        d.addPairwiseGaussian(sxy=20, compat=3)
        d.addPairwiseBilateral(sxy=30, srgb=20, rgbim=image, compat=10)

        Q = d.inference(5)
        Q = np.argmax(np.array(Q), axis=0).reshape((h, w))


    return Q


# def wtp_loss(model, batch):
#     model.train()
#     # put variables in cuda
#     images = batch["images"].cuda()
#     points = batch["points"].cuda()
#     counts = batch["counts"].cuda()

#     O = model(images)
#     S = F.softmax(O, 1)
#     S_log = F.log_softmax(O, 1)

#     # IMAGE LOSS
#     loss = methods.image_loss(S, counts)

#     # POINT LOSS
#     loss += F.nll_loss(S_log, points, 
#                        ignore_index=0,
#                        size_average=False)

#     return loss



def crossentropy_loss(model, batch):
    model.train()
    N =  batch["images"].size(0)
    # put variables in cuda
    images = batch["images"].cuda()
    labels = batch["labels"].cuda()

    loss_func = lambda s: F.cross_entropy(s, labels, size_average=False, 
                                  ignore_index=model.ignore_index) / N
    
    if hasattr(model, "with_aux") and model.with_aux:
        scores = model.forward_aux(images, with_aux=True)
        loss = loss_func(scores["output"]) + 0.4 * loss_func(scores["aux"])
    else:
        scores = model(images)
        loss = loss_func(scores)

    
    return loss




def blob_loss_multi(model, batch, split_mode="line", 
              loss_list=(None,)):
    model.train()
    N =  batch["images"].size(0)
    assert N == 1

    
    # blobs = blob_dict["blobs"]
    # blob_list = blob_dict["blobList"]
    # put variables in cuda
    images = batch["images"].cuda()
    pointsAll = batch["points"].cuda()
    countsAll = batch["counts"].cuda()

    ind_pos = ms.t2n(countsAll > 0).ravel()
    ind_neg = ms.t2n(countsAll == 0).ravel()

    w_pos = ind_pos.sum()
    w_neg = ind_neg.sum()
    p = np.zeros(countsAll.shape[1]) 

    p[ind_neg.astype(bool)] = 1./w_neg
    p[ind_pos.astype(bool)] = 1./w_pos

    loss = 0.
    #for j, decoder in enumerate(model.decoderList):       
    j = int(np.random.choice(np.arange(model.n_classes-1),
                     size=None,
                     p=p/p.sum()))

    decoder = model.decoderList[j]
    decoder.train()
    blob_dict = methods.get_blob_dict(decoder, batch)

    points = (pointsAll == (j+1)).long().detach()
    counts = countsAll[:, [j]].detach()

    O = decoder(images)
    S = F.softmax(O, 1)
    S_log = F.log_softmax(O, 1)

    # if counts == 0:
    #     scale = 1. / w_neg
    # else:
    #     scale = 1. / w_pos

    # IMAGE LOSS
    loss_class = methods.image_loss(S, counts)

    # POINT LOSS
    loss_class += F.nll_loss(S_log, points, 
                       ignore_index=0,
                       size_average=False)
    # FP loss
    if blob_dict["n_fp"] > 0:
        loss_class += fp_loss(S_log, blob_dict)

    # Split loss
    if blob_dict["n_multi"] > 0:
        loss_class += split_loss(S_log, S, points, blob_dict, 
                           split_mode=split_mode)

    loss += loss_class



    return loss / N


def water_loss_multi(model, batch):
    return blob_loss_multi(model, batch, split_mode="water")

def water_loss_B_multi(model, batch):
    return blob_loss_multi(model, batch, split_mode="water", 
                     loss_list=["water_boundary"])







def salient_loss(model, batch):
    import ipdb; ipdb.set_trace()  # breakpoint 8f10f747 //
    
    return blob_loss(model, batch, split_mode="water")

def seg_loss(model, batch):
    model.train()
    N =  batch["images"].size(0)
    #assert N == 1

    # put variables in cuda

    density_images = batch["density_images"].cuda()
    maskClasses = batch["maskClasses"].cuda()

    O = model(density_images)
    S = F.log_softmax(O, 1)

    
    
    
    loss = F.nll_loss(S, maskClasses, size_average=False)


    return loss / N

def bbox2mask(bbox, img_h, img_w):
    mask = np.zeros((img_h, img_w),int)
    #bbox = bbox.squeeze()

    x, y, w, h, s = bbox
    x, y, w, h, s = int(x*img_w), int(y*img_h), int(w*img_w), int(h*img_h), s 
    mask[y:y+h,x] = 1
    mask[y:y+h,x+w] = 1
    mask[y,x:x+w] = 1
    mask[y+h,x:x+w] = 1
    return mask

def yolo_loss(model, batch):
    
    N =  batch["images"].size(0)
    images = batch["images"].cuda()
    maskClasses = batch["maskClasses"].cuda()

    outputs = model(images)

    labels = bboxes = torch.FloatTensor(np.array([[[0.1,0.2,0.5,0.1,1]]]))
    #model.yolo_losses[0](outs[0], bboxes)

    if 1:
        losses_name = ["total_loss", "x", "y", "w", "h", "conf", "cls"]
        losses = [[]] * len(losses_name)
        for i in range(3):
            _loss_item = model.yolo_losses[i](outputs[i], labels)
            for j, l in enumerate(_loss_item):
                losses[j].append(l)
        losses = [sum(l) for l in losses]
        loss = losses[0]

    # for i, name in enumerate(losses_name):
    #     print(name, losses[i].item())

    return loss 

from skimage.segmentation import find_boundaries

def unet_loss(model, batch, size_average=False, void_pixels=255, batch_average=True):
    
    images_points = batch["images"].cuda()
    true_masks = batch["maskClasses"].cuda().float()
    
    true_masks = (true_masks != 0).float()

    criterion = nn.BCELoss()

    masks_pred = model(images_points)
    masks_probs = F.sigmoid(masks_pred)
    masks_probs_flat = masks_probs.view(-1)

    true_masks_flat = true_masks.view(-1)
    ind_0 = true_masks_flat == 0
    ind_1 = true_masks_flat != 0

    loss_0 = criterion(masks_probs_flat[ind_0], true_masks_flat[ind_0])
    loss_1 = criterion(masks_probs_flat[ind_1], true_masks_flat[ind_1])

    boundaries = torch.from_numpy(1 - find_boundaries(batch["maskObjects"].numpy())).cuda()

    ind_b = boundaries == 0
    #print(masks_pred[:,0][ind_b])
    if not hasattr(model, "iterations"):
        model.iterations = 0

    model.iterations += 1
    if model.iterations > 18:
        pass
        # import ipdb; ipdb.set_trace()  # breakpoint 47ce6322 //


    loss_3 = batch["maskObjects"].max().item() * criterion(masks_probs[:,0][ind_b], boundaries[ind_b].float())
    # print("passed")

    return loss_1 + loss_0 + loss_3


def unetpoint_loss(model, batch, size_average=False, void_pixels=255, batch_average=True):
    
    images_points = batch["images_points"].cuda()
    true_masks = batch["maskClasses"].cuda().float()
    
    true_masks = (true_masks != 0).float()

    criterion = nn.BCELoss()

    masks_pred = model(images_points)
    masks_probs = F.sigmoid(masks_pred)
    masks_probs_flat = masks_probs.view(-1)

    true_masks_flat = true_masks.view(-1)
    ind_0 = true_masks_flat == 0
    ind_1 = true_masks_flat != 0

    loss_0 = criterion(masks_probs_flat[ind_0], true_masks_flat[ind_0])
    loss_1 = criterion(masks_probs_flat[ind_1], true_masks_flat[ind_1])

    boundaries = torch.from_numpy(1 - find_boundaries(batch["maskObjects"].numpy())).cuda()

    ind_b = boundaries == 0
    #print(masks_pred[:,0][ind_b])
    if not hasattr(model, "iterations"):
        model.iterations = 0

    model.iterations += 1
    if model.iterations > 18:
        pass
        # import ipdb; ipdb.set_trace()  # breakpoint 47ce6322 //


    loss_3 = batch["maskObjects"].max().item() * criterion(masks_probs[:,0][ind_b], boundaries[ind_b].float())
    # print("passed")

    return loss_1 + loss_0 + loss_3
def seg_balanced_loss(model, batch, size_average=False, void_pixels=255, batch_average=True):

    density_images = batch["density_images"].cuda()
    labels = batch["maskClasses"].cuda().float()

    output = model(density_images)
    #S = F.log_softmax(O, 1)
    # assert(output.size() == label.size())

    num_labels_pos = torch.sum(labels)
    num_labels_neg = torch.sum(1.0 - labels)
    num_total = num_labels_pos + num_labels_neg

    output_gt_zero = torch.ge(output, 0).float()
    loss_val = torch.mul(output, (labels - output_gt_zero)) - torch.log(
        1 + torch.exp(output - 2 * torch.mul(output, output_gt_zero)))

    loss_pos_pix = -torch.mul(labels, loss_val)
    loss_neg_pix = -torch.mul(1.0 - labels, loss_val)

    # if void_pixels is not None:
    #     w_void = torch.le(void_pixels, 0.5).float()
    #     loss_pos_pix = torch.mul(w_void, loss_pos_pix)
    #     loss_neg_pix = torch.mul(w_void, loss_neg_pix)
    #     num_total = num_total - torch.ge(void_pixels, 0.5).float().sum()

    loss_pos = torch.sum(loss_pos_pix)
    loss_neg = torch.sum(loss_neg_pix)

    final_loss = num_labels_neg / num_total * loss_pos + num_labels_pos / num_total * loss_neg

    if size_average:
        final_loss /= np.prod(labels.size())
    elif batch_average:
        final_loss /= labels.size()[0]

    return final_loss




# def old_blob_loss(model, batch, split_mode="line", 
#               loss_list=(None,)):
    
#     model.train()
#     N =  batch["images"].size(0)
#     assert N == 1

#     blob_dict = helpers.get_blob_dict(model, batch, training=True)
#     blobs = blob_dict["blobs"]
#     blob_list = blob_dict["blobList"]
#     # put variables in cuda
#     images = batch["images"].cuda()
#     points = batch["points"].cuda()
#     counts = batch["counts"].cuda()
#     #print(images.shape)

#     O = model(images)
#     S = F.softmax(O, 1)
#     S_log = F.log_softmax(O, 1)

#     import ipdb; ipdb.set_trace()  # breakpoint c426dc73 //

#     # IMAGE LOSS
#     loss = helpers.compute_image_loss(S, counts)

#     # POINT LOSS
#     loss += F.nll_loss(S_log, points, 
#                        ignore_index=0,
#                        size_average=False)
#     # FP loss
#     if blob_dict["n_fp"] > 0:
#         loss += fp_loss(S_log, blob_dict)

#     # Split loss
#     if blob_dict["n_multi"] > 0:
#         loss += helpers.compute_split_loss(S_log, S, points, blob_dict, split_mode=split_mode)

#     # Global loss
#     if "water_boundary" in loss_list:  
#         loss += helpers.compute_boundary_loss(S_log, S, points, counts)

#     # Boundary loss
#     # if "crf_boundary" in loss_list:        
#     #     loss += crf_boundary_loss(model, batch, S_log)
#     if "crf_constrain" in loss_list:
#         loss += crf_constrain_loss(model, batch, S_log)

#     if "expand_salient" in loss_list:
        
#         n_objects = counts.sum().item()
#         diff = abs(len(blob_list) - n_objects)

#         if int(counts.sum()) > 0 and diff < 3:
#             T = crfFunc.crf_fit(batch["images"], S, blobs=None)
#             if T is not None:
#                 #B = find_boundaries(ms.t2n(T))
#                 #B = torch.from_numpy(B).float().cuda()
#                 loss += F.binary_cross_entropy(S[:,1], T)
                
#                 loss += F.nll_loss(S_log, (T>0.5).long(),
#                          ignore_index=0) * len(blob_list)
#                 #loss += F.nll_loss(S, T,ignore_index=0)
#                 #loss += F.nll_loss(S, find_boundaries(T),
#                 #                  ignore_index=0)
#                 #loss += 5e-4 * S[:,1].sum() 

#     if "expand_fg" in loss_list and int(counts.sum()) > 0:

#         # Foreground        
#         n_size = blob_dict["total_size"]
#         S_probs = S[0,1].view(-1)
#         d_probs = 0.996

#         # IMAGE LOSS
#         n,k,h,w = S.size()
#         assert n == 1

#         Savg = torch.sort(S_probs, dim=0, descending=True)[0]
#         if n_size != Savg.size(0):
#             #print(n_size)
#             Savg = Savg[n_size:]
#             n_pixels = Savg.size(0)

#             if n_pixels != 0:
#                 Target = ms.t2f(np.ones(n_pixels))
#                 weight = ms.t2f(d_probs**np.arange(n_pixels))

#                 loss += F.binary_cross_entropy(Savg, Target, weight=weight,
#                           size_average=False) / weight.sum()

#     if "expand_nei_fg" in loss_list and int(counts.sum()) > 0:
#         n_pixels = 5000

#         # Foreground      
#         S_fg = S[0,1]  

#         ind_bg, W = methods.get_closest_nei(S_fg, images, thresh=0.7, 
#                                       n_pixels=5000, topk=1000)
#         if ind_bg is not None:
#             Probs = S_fg.view(-1)[ind_bg]
#             Target = torch.ones(Probs.size(0))

#             scale = (W/W.max()).sum()
#             loss += F.binary_cross_entropy(Probs, 
#                                            Target.cuda(), 
#                                            weight=W.cuda(),
#                                            size_average=False) / W.sum()

#     if "expand_nei_bg" in loss_list and int(counts.sum()) > 0:
#         n_pixels = 5000

#         # Background        
#         S_bg = S[0,0]

#         ind_fg, W = methods.get_closest_nei(S_bg, images, thresh=0.8, 
#                                       n_pixels=5000, topk=1000)
#         if ind_fg is not None:
#             Probs = S_bg.view(-1)[ind_fg]
#             Target = torch.zeros(Probs.size(0))

#             scale = (W/W.max()).sum()
#             loss += F.binary_cross_entropy(Probs, 
#                                            Target.cuda(), 
#                                            weight=W.cuda(),
#                                            size_average=False) / W.sum()

#     if "expand_bg" in loss_list and int(counts.sum()) > 0:
#         # Background      
       
#         S_probs = S[0,0].view(-1)
#         n_size = int((S_probs < 0.1).sum())
#         d_probs = 0.999
#         #print(d_probs)
#         # IMAGE LOSS
#         n,k,h,w = S.size()
#         assert n == 1


#         Savg = torch.sort(S_probs, dim=0, descending=True)[0]
#         Savg = Savg[n_size:]

#         n_pixels = Savg.size(0)
#         if n_pixels != 0:
#             Target = ms.t2f(np.zeros(n_pixels))
#             weight = ms.t2f(d_probs**np.arange(n_pixels))
#             loss += F.binary_cross_entropy(Savg, Target, weight=weight,
#                       size_average=False) / weight.sum()

#     if "expand_sp" in loss_list and int(counts.sum()) > 0:
#         img = ms.f2l(vis.denormalize(ms.t2n(batch["images"])))[0]
        


#         seg = slic_superpixels.slic(img, n_segments=1000)
        


#         full_mask = np.zeros(img.shape[:2])
#         #vis.images(batch["images"], seg, denorm=1)
#         b_count = 0
#         for blob_i  in blob_list:
#             if blob_i["n_points"] != 1:
#                 continue
#             else:
#                 b_count += 1
#                 yx = blob_i["pointsList"][0]
#                 mask = blobs == blob_i["label"]
#                 mask = np.isin(seg, np.delete(np.unique(mask*seg), 0))
#                 full_mask += mask

#         Target = torch.LongTensor(full_mask.clip(0,1))
#         loss += F.nll_loss(S_log, Target[None].cuda(), 
#                                        ignore_index=0,
#                                        size_average=True) * float(b_count)

#     if "constrain_sp" in loss_list and int(counts.sum()) > 0:
#         img = ms.f2l(vis.denormalize(ms.t2n(batch["images"])))[0]
        

#         seg = slic_superpixels.slic(img, n_segments=1000)
        

#         mask = ms.t2n((S[0,0] > 0.5))
#         uniques = np.delete(np.unique(mask*seg), 0)
#         mask = 1 - np.isin(seg, uniques)


#         Target = torch.LongTensor(mask.astype(int))
#         loss += F.nll_loss(S_log, Target[None].cuda(), 
#                                        ignore_index=1,
#                                        size_average=True)*float(batch["counts"].squeeze().item())

#     if "expand_sp_M" in loss_list and int(counts.sum()) > 0:
#         img = ms.f2l(vis.denormalize(ms.t2n(batch["images"])))[0]
        
        
#         seg = slic_superpixels.slic(img, n_segments=1000)
        


#         #vis.images(batch["images"], seg, denorm=1)
#         Target = torch.FloatTensor([1])[0].cuda()
#         for blob_i  in blob_list:
#             if blob_i["n_points"] != 1:
#                 continue
#             else:
#                 yx = blob_i["pointsList"][0]
#                 mask = blobs == blob_i["label"]
#                 uniques = np.delete(np.unique(mask*seg), 0)
#                 mask = np.isin(seg, uniques)

#                 Smax = (S[0,1]*torch.FloatTensor(mask.astype(int)).cuda()).view(-1).topk(1)[0]
#                 loss += F.binary_cross_entropy(Smax, Target.expand(Smax.shape[0]), 
#                                               size_average=True)
#                 #print("blob {}".format(3))
              

#     if "expand_each_region" in loss_list and int(counts.sum()) > 0:
#         #seg = sp.water_regions(ms.t2n(S)[0,1], ms.t2n(points).squeeze())
#         seg_dict = sp.water_regions(ms.t2n(S)[0,1], 
#                                     ms.t2n(points).squeeze(),
#                                     return_dict=True)
#         seg = seg_dict["seg"]
#         for seg_i in seg_dict["yx_list"]:
#             mask = seg == seg_i["label"]
            

#             blob_size = None
#             for blob_i  in blob_list:
#                 if blob_i["n_points"] != 1:
#                     continue
#                 else:
#                     yx = blob_i["pointsList"][0]
#                     if yx["y"] == seg_i["y"] and yx["x"] == seg_i["x"]:  
#                         blob_size = int(blob_i["size"])   
#                         break


#             if blob_size is None:
#                 continue

#             #image_max_loss(S, counts)
#             #loss += image_avg_loss(S, mask, blob_size=blob_size+10) / blob_size

#             #seg = sp.water_regions(probs, points)

#     if "affinity" in loss_list and int(counts.sum()) > 0:
#         ##########
#         A = get_affinity(images)
#         G = sparse_c2t(A).cuda()
#         _, _, h, w = images.shape
#         S_flat = S.view(1, 2, h*w)
#         for i in range(2):
#             Sk = S_flat[:, i]
#             U = torch.mm(Sk, torch.mm(G, 1. - Sk.t())).sum()

#             A_sum = torch.FloatTensor(np.asarray(A.sum(1))).cuda()
#             D = torch.mm(Sk, A_sum).sum()

#             loss += U / D
#         ##########

#     return loss / N


# # -------------------- MAIN LOSS
# def image_loss(model, batch):
#     model.train()
#     N =  batch["images"].size(0)
#     #assert N == 1

#     # put variables in cuda
#     images = batch["images"].cuda()
#     counts = batch["counts"].cuda()

#     O = model(images)
#     S = F.softmax(O, 1)

#     # IMAGE LOSS
#     n,k,h,w = S.size()

#     # GET TARGET
#     ones = torch.ones(counts.size(0), 1).long().cuda()
#     BgFgCounts = torch.cat([ones, counts], 1)
#     Target = (BgFgCounts.view(n*k).view(-1) > 0).view(-1).float()

#     # GET INPUT
#     Smax = S.view(n, k, h*w).max(2)[0].view(-1)
    
    
#     loss = F.binary_cross_entropy(Smax, Target, size_average=False)


#     return loss / N


# def acol_loss(model, batch):
#     model.train()
#     N =  batch["images"].size(0)
#     #assert N == 1

#     # put variables in cuda
#     images = batch["images"].cuda()
#     counts = batch["counts"].cuda()


#     O = model(images)
#     S = F.softmax(O, 1)

#     # IMAGE LOSS
#     n,k,h,w = S.size()

#     # GET TARGET
#     Target = torch.zeros(n, k).cuda()
#     Target[:,0] = 1

#     for i in range(n):
#         n_counts = counts[i].sum().item()
#         if n_counts == 1:
#             Target[i, 1] = 1
#         elif n_counts > 1:
#             Target[i] = 1

#     # GET INPUT
#     S_flat = S.view(n, k, h*w)
#     Smax = S_flat.max(2)[0]

    


#     import ipdb; ipdb.set_trace()  # breakpoint 83844ed7 //
#     loss = F.binary_cross_entropy(Smax.view(-1), Target.view(-1), size_average=False)

#     # Separate the blobs

#     return loss / N



# def nc_loss(model, batch):

#     model.train()
#     N =  batch["images"].size(0)
#     #assert N == 1

#     # put variables in cuda
#     images = X = batch["images"].cuda()
#     counts = batch["counts"].cuda()

#     O = model(images)
#     S = F.softmax(O, 1)

#     # IMAGE LOSS
#     n,k,h,w = S.size()

#     # GET TARGET
#     ones = torch.ones(counts.size(0), 1).long().cuda()
#     BgFgCounts = torch.cat([ones, counts], 1)
#     Target = (BgFgCounts.view(n*k).view(-1) > 0).view(-1).float()

#     # GET INPUT
#     S_flat = S.view(n, k, h*w)[:,1]
#     Smax = S.view(n, k, h*w).max(2)[0].view(-1)
    
    
#     loss = F.binary_cross_entropy(Smax, Target, size_average=False)

#     ##########
#     A = get_affinity(images)
#     G = sparse_c2t(A).cuda()

#     U = torch.mm(S_flat, torch.mm(G, 1. - S_flat.t())).sum()

#     A_sum = torch.FloatTensor(np.asarray(A.sum(1))).cuda()
#     D = torch.mm(S_flat, A_sum).sum()

#     loss += U / D
#     ##########

#     return loss / N



# def wildcat_reg_loss(model, batch):
#     images = batch["images"].cuda()
#     counts = batch["counts"].cuda()

#     target = counts.clamp(0, 1).float()
    
#     H = model.get_heatmaps(images)


#     diff = (H[:,1] - H[:,0])**2
#     loss = diff.sum()

#     out = model.spatial_pooling(H)
#     loss += F.multilabel_soft_margin_loss(out, target)
#     return loss


# def wildcat_loss(model, batch):
#     images = batch["images"].cuda()
#     counts = batch["counts"].cuda()

#     target = counts.clamp(0, 1).float()
    
#     out = model(images)
#     loss = F.multilabel_soft_margin_loss(out, target)
#     return loss

# def spn_loss(model, batch):
#     images = batch["images"].cuda()
#     counts = batch["counts"].cuda()

#     target = counts.clamp(0, 1).float()

#     out = model(images)
#     loss = F.multilabel_soft_margin_loss(out, target)
#     return loss

# # def spn_loss(model, batch):
# #     images = batch["images"].cuda()
# #     counts = batch["counts"].cuda()

# #     target = counts.clamp(0, 1).float()

# #     out = model(images)
# #     loss = F.multilabel_soft_margin_loss(out, target)
# #     return loss

# def weakly_loss(model, batch):
#     model.train()
#     N =  batch["images"].size(0)
#     #assert N == 1

#     # put variables in cuda
#     images = batch["images"].cuda()
#     counts = batch["counts"].cuda()


#     O = model(images)
#     S = F.softmax(O, 1)

#     # IMAGE LOSS
#     n,k,h,w = S.size()

#     # GET TARGET
#     Target = torch.zeros(n, k).cuda()
#     Target[:,0] = 1

#     for i in range(n):
#         n_counts = counts[i].sum().item()
#         if n_counts == 1:
#             Target[i, 1] = 1
#         elif n_counts > 1:
#             Target[i] = 1

#     # GET INPUT
#     S_flat = S.view(n, k, h*w)
#     Smax = S_flat.max(2)[0]

#     loss = F.binary_cross_entropy(Smax.view(-1), Target.view(-1), size_average=False)


