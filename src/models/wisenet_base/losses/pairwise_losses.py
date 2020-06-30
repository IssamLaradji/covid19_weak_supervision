import numpy as np 
import torch 
from ..addons.pycocotools import mask as maskUtils
from .. import ann_utils as au
from .. import misc as ms

def compute_triplet_loss(model, batch, random_proposal=False, similarity_function=""):
    n,c,h,w = batch["images"].shape

    model.train()
    O_dict = model(batch["images"].cuda())
    O = O_dict["embedding_mask"]

    n,c,h,w = O.shape
    
    points = batch["points"]
    batch["maskObjects"] = None 
    batch['maskClasses'] = None
    batch["maskVoid"] = None
    
    pointList = au.mask2pointList(points)["pointList"]

    loss = torch.tensor(0.).cuda()
    if len(pointList) == 0:
        return loss

    propDict = au.pointList2propDict(pointList, batch, 
                                     single_point=True,
                                     thresh=0.5)


    Embeddings, Labels = propDict2EmbedLabels(O, propDict, random_proposal=True)
    
    loss = triplet(Embeddings, torch.LongTensor(Labels))
    return loss


def pdist(vectors):
    D = -2 * vectors.mm(torch.t(vectors)) 
    D += vectors.pow(2).sum(dim=1).view(1, -1) 
    D += vectors.pow(2).sum(dim=1).view(-1, 1)

    return D


def propDict2EmbedLabels(O, propDict, n_neighbors=100, random_proposal=False):
    
    nodeList = []
    for prop in propDict["propDict"]:
        node = np.array([prop["point"]["y"], prop["point"]["x"], 
                        prop["point"]["category_id"]])
        nodeList += [node]

        if len(prop["annList"]) == 0:
            continue

        if random_proposal:
            scores = np.array([p["score"] for p in prop["annList"]])
            p = np.random.choice(prop["annList"], p=scores/scores.sum())
            mask = p["mask"]
        else:
            mask = prop["annList"][0]["mask"]

        n_indices = 100
        yx_list = get_random_indices(mask, n_indices=n_indices)
        yList, xList = yx_list["yList"], yx_list["xList"]
     
        node = np.zeros((n_indices, 3), int)
        node[:,0] = yList
        node[:,1] = xList
        node[:,2] = prop["point"]["category_id"]

        nodeList += [node]

    # Background
    background = propDict["background"]

    if background.sum() == 0:
        y_axis = np.random.randint(0, background.shape[1], 100)
        x_axis = np.random.randint(0, background.shape[2], 100)
        background[0,y_axis, x_axis] = 1
        asas
    else:
        n_indices = min(300, 100*(len(prop["annList"])+1))
        bg_seeds = get_random_indices(background, n_indices=n_indices)
    
    node = np.zeros((n_indices, 3), int)
    node[:, 0] = bg_seeds["yList"] 
    node[:, 1] = bg_seeds["xList"] 
    node[:, 2] = 0

    nodeList += [node]

    nodeArray = np.vstack(nodeList)
    Embeddings = O[:,:,nodeArray[:,0], nodeArray[:,1]].squeeze()
    return Embeddings.t(), nodeArray[:,2]


def get_random_indices(mask, n_indices=10):
    mask_ind = np.where(mask.squeeze())
    n_pixels = mask_ind[0].shape[0]
    P_ind = np.random.randint(0, n_pixels, n_indices)
    yList = mask_ind[0][P_ind]
    xList = mask_ind[1][P_ind]

    return {"yList":yList, "xList":xList}

import torch.nn.functional as F
def triplet(embeddings, labels):
    triplets = get_triplets(embeddings, labels)

    f_A = embeddings[triplets[:, 0]]
    f_P = embeddings[triplets[:, 1]]
    f_N = embeddings[triplets[:, 2]]

    ap_D = (f_A - f_P).pow(2).sum(1)  # .pow(.5)
    an_D = (f_A - f_N).pow(2).sum(1)  # .pow(.5)
    losses = F.relu(ap_D - an_D + 1.)

    return losses.mean()

from itertools import combinations
def get_triplets(embeddings, y):

  margin = 1
  D = pdist(embeddings)
  D = D.cpu()

  y = y.cpu().data.numpy().ravel()
  trip = []

  for label in set(y):
      label_mask = (y == label)
      label_indices = np.where(label_mask)[0]
      if len(label_indices) < 2:
          continue
      neg_ind = np.where(np.logical_not(label_mask))[0]
      
      ap = list(combinations(label_indices, 2))  # All anchor-positive pairs
      ap = np.array(ap)

      ap_D = D[ap[:, 0], ap[:, 1]]
      
      # # GET HARD NEGATIVE
      # if np.random.rand() < 0.5:
      #   trip += get_neg_hard(neg_ind, hardest_negative,
      #                D, ap, ap_D, margin)
      # else:
      trip += get_neg_hard(neg_ind, random_neg,
                 D, ap, ap_D, margin)

  if len(trip) == 0:
      ap = ap[0]
      trip.append([ap[0], ap[1], neg_ind[0]])

  trip = np.array(trip)

  return torch.LongTensor(trip)

def semihard_negative(loss_values, margin=1):
    semihard_negatives = np.where(np.logical_and(loss_values < margin, loss_values > 0))[0]
    return np.random.choice(semihard_negatives) if len(semihard_negatives) > 0 else None

def get_neg_hard(neg_ind, 
                      select_func,
                      D, ap, ap_D, margin):
    trip = []

    for ap_i, ap_di in zip(ap, ap_D):
        loss_values = (ap_di - 
               D[torch.LongTensor(np.array([ap_i[0]])), 
                torch.LongTensor(neg_ind)] + margin)

        loss_values = loss_values.data.cpu().numpy()
        neg_hard = select_func(loss_values)

        if neg_hard is not None:
            neg_hard = neg_ind[neg_hard]
            trip.append([ap_i[0], ap_i[1], neg_hard])

    return trip

def random_neg(loss_values):
    neg_hards = np.where(loss_values > 0)[0]
    return np.random.choice(neg_hards) if len(neg_hards) > 0 else None


def compute_pairwise_loss(O, batch, random_proposal=False, similarity_function="", 
    fp_loss=True, sim_loss=False, multi_proposals=False):

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
    aggDict = propDict["aggDict"]
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

        if multi_proposals and len(propDict[i]["annList"]):
            
            probs_i = aggDict[i][mask_ind]
            P_ind = np.random.choice(np.arange(n_pixels), size=min(100,n_pixels),
                                replace=False, p=probs_i/probs_i.sum())
        else:
            P_ind = np.random.randint(0, n_pixels, min(100,n_pixels))

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

    if fp_loss:
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
    if sim_loss:
        loss += compute_similarity_loss(O, batch)

    return loss / max(n_seeds, 1)



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



def compute_pairwise_random_loss(O, batch, similarity_function=""):

    n,c,h,w = O.shape
    
    points = batch["points"]
    batch["maskObjects"] = None 
    batch['maskClasses'] = None
    batch["maskVoid"] = None
    
    pointList = au.mask2pointList(points)["pointList"]

    loss = torch.tensor(0.).cuda()
    if len(pointList) == 0:
        return loss



    propDict = au.pointList2propDict(pointList, batch, 
                                     single_point=True,
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
        prop = propDict[i]
        annList = propDict[i]["annList"]

        if len(annList) == 0:
            mask = np.zeros(points.squeeze().shape)
            mask[propDict[i]["point"]["y"], propDict[i]["point"]["x"]] = 1
        else:            
            scores = np.array([p["score"] for p in prop["annList"]])
            p = np.random.choice(prop["annList"], p=scores/scores.sum())
            mask = p["mask"]
        
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






























def OneHeadLoss_HardNegatives(model, batch, visualize=False):
    n,c,h,w = batch["images"].shape

    model.train()
    O_dict = model(batch["images"].cuda())
    O = O_dict["embedding_mask"]

    base_dict = metric_base(O, batch)
    points = batch["points"]
    yList = base_dict["yList"]
    xList = base_dict["xList"]
    propDict = base_dict["propDict"]
    yList = base_dict["yList"]
    background = base_dict["background"]

    ###################################
    n,c,h,w = O.shape
    fg_seeds = O[:, :, yList, xList]
    n_seeds = fg_seeds.shape[-1]
    prop_mask = np.zeros((h, w))

    loss = torch.tensor(0.).cuda()

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
        
        ap = - torch.log(log_pairwise(f_A, fg_P)) 
        loss += ap.mean()

        # Get Negatives
        if n_seeds > 1:
            N_ind = [j for j in range(n_seeds) if j != i]
            f_N = fg_seeds[:,:,N_ind]
            an = - torch.log(1. - log_pairwise(f_A, f_N)) 
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
    ap = - torch.log(log_pairwise(f_A[:,:,None], f_P[:,:,:,None])) 
    an = - torch.log(1. - log_pairwise(f_A[:,:,None], fg_seeds[:,:,:,None])) 

    loss += ap.mean()
    loss += an.mean()

    if batch["dataset"][0] == "cityscapes" or batch["dataset"][0] == "coco2014":        
        n_max = 6
    else:
        n_max = 12

    if f_A.shape[2] < n_max:
        with torch.no_grad():

            diff = log_pairwise(O.view(1,c,-1)[:,:,:,None], 
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

            ap = - torch.log(log_pairwise(f_A[:,:,None], f_P[:,:,:,None])) 
            an = - torch.log(1. - log_pairwise(f_P[:,:,None], fg_seeds[:,:,:,None])) 

            # if i < 3:
            loss += ap.mean()
            loss += an.mean()


    return loss / max(n_seeds, 1)


def OneHeadLoss_Triplet(model, batch, visualize=False):
    n,c,h,w = batch["images"].shape

    model.train()
    O_dict = model(batch["images"].cuda())
    O = O_dict["embedding_mask"]

    base_dict = metric_base(O, batch)
    points = batch["points"]
    yList = base_dict["yList"]
    xList = base_dict["xList"]
    propDict = base_dict["propDict"]
    yList = base_dict["yList"]
    background = base_dict["background"]

    ###################################
    n,c,h,w = O.shape
    fg_seeds = O[:, :, yList, xList]
    n_seeds = fg_seeds.shape[-1]
    prop_mask = np.zeros((h, w))

    loss = torch.tensor(0.).cuda()

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
        
        ap = - torch.log(log_pairwise(f_A, fg_P)) 
        loss += ap.mean()

        # Get Negatives
        if n_seeds > 1:
            N_ind = [j for j in range(n_seeds) if j != i]
            f_N = fg_seeds[:,:,N_ind]
            an = - torch.log(1. - log_pairwise(f_A, f_N)) 
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
    ap = - torch.log(log_pairwise(f_A[:,:,None], f_P[:,:,:,None])) 
    an = - torch.log(1. - log_pairwise(f_A[:,:,None], fg_seeds[:,:,:,None])) 

    loss += ap.mean()
    loss += an.mean()

    if batch["dataset"][0] == "cityscapes" or batch["dataset"][0] == "coco2014":        
        n_max = 6
    else:
        n_max = 12

    if f_A.shape[2] < n_max:
        with torch.no_grad():

            diff = log_pairwise(O.view(1,c,-1)[:,:,:,None], 
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

            ap = - torch.log(log_pairwise(f_A[:,:,None], f_P[:,:,:,None])) 
            an = - torch.log(1. - log_pairwise(f_P[:,:,None], fg_seeds[:,:,:,None])) 

            # if i < 3:
            loss += ap.mean()
            loss += an.mean()


    return loss / max(n_seeds, 1)


def OneHeadLoss_new(model, batch, visualize=False):
    n,c,h,w = batch["images"].shape

    model.train()
    O_dict = model(batch["images"].cuda())
    O = O_dict["embedding_mask"]

    base_dict = metric_base(O, batch)
    points = batch["points"]
    yList = base_dict["yList"]
    xList = base_dict["xList"]
    propDict = base_dict["propDict"]
    yList = base_dict["yList"]
    background = base_dict["background"]

    ###################################
    n,c,h,w = O.shape
    fg_seeds = O[:, :, yList, xList]
    n_seeds = fg_seeds.shape[-1]
    prop_mask = np.zeros((h, w))

    loss = torch.tensor(0.).cuda()

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
        
        ap = - torch.log(log_pairwise(f_A, fg_P)) 
        loss += ap.mean()

        # Get Negatives
        if n_seeds > 1:
            N_ind = [j for j in range(n_seeds) if j != i]
            f_N = fg_seeds[:,:,N_ind]
            an = - torch.log(1. - log_pairwise(f_A, f_N)) 
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
    ap = - torch.log(log_pairwise(f_A[:,:,None], f_P[:,:,:,None])) 
    an = - torch.log(1. - log_pairwise(f_A[:,:,None], fg_seeds[:,:,:,None])) 

    loss += ap.mean()
    loss += an.mean()

    if batch["dataset"][0] == "cityscapes" or batch["dataset"][0] == "coco2014":        
        n_max = 6
    else:
        n_max = 12

    if f_A.shape[2] < n_max:
        with torch.no_grad():

            diff = log_pairwise(O.view(1,c,-1)[:,:,:,None], 
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

            ap = - torch.log(log_pairwise(f_A[:,:,None], f_P[:,:,:,None])) 
            an = - torch.log(1. - log_pairwise(f_P[:,:,None], fg_seeds[:,:,:,None])) 

            # if i < 3:
            loss += ap.mean()
            loss += an.mean()


    return loss / max(n_seeds, 1)


def OneHeadLoss_probs(O, batch, visualize=False):
    n,c,h,w = O.shape
    similarity_function = au.log_pairwise_sum
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



def OneHeadLoss_new(model, batch, visualize=False):
    n,c,h,w = batch["images"].shape

    model.train()
    O_dict = model(batch["images"].cuda())
    O = O_dict["embedding_mask"]
    
    base_dict = helpers.metric_base(O, batch)
    points = batch["points"]
    yList = base_dict["yList"]
    xList = base_dict["xList"]
    propDict = base_dict["propDict"]
    yList = base_dict["yList"]
    background = base_dict["background"]

    ###################################
    n,c,h,w = O.shape
    
    fg_seeds = O[:, :, yList, xList]
    n_seeds = fg_seeds.shape[-1]
    prop_mask = np.zeros((h, w))

    loss = torch.tensor(0.).cuda()

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
        
        ap = - torch.log(au.log_pairwise(f_A, fg_P)) 
        loss += ap.mean()

        # Get Negatives
        if n_seeds > 1:
            N_ind = [j for j in range(n_seeds) if j != i]
            f_N = fg_seeds[:,:,N_ind]
            an = - torch.log(1. - au.log_pairwise(f_A, f_N)) 
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
    ap = - torch.log(au.log_pairwise(f_A[:,:,None], f_P[:,:,:,None])) 
    an = - torch.log(1. - au.log_pairwise(f_A[:,:,None], fg_seeds[:,:,:,None])) 

    loss += ap.mean()
    loss += an.mean()

    if batch["dataset"][0] == "cityscapes" or batch["dataset"][0] == "coco2014":        
        n_max = 6
    else:
        n_max = 6

    if f_A.shape[2] < n_max:
        with torch.no_grad():

            diff = au.log_pairwise(O.view(1,c,-1)[:,:,:,None], 
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

            ap = - torch.log(au.log_pairwise(f_A[:,:,None], f_P[:,:,:,None])) 
            an = - torch.log(1. - au.log_pairwise(f_P[:,:,None], fg_seeds[:,:,:,None])) 

            # if i < 3:
            loss += ap.mean()
            loss += an.mean()


    return loss / max(n_seeds, 1)

def OneHeadLoss_new2(model, batch, visualize=False):
    n,c,h,w = batch["images"].shape

    model.train()
    O_dict = model(batch["images"].cuda())
    O = O_dict["embedding_mask"]

    base_dict = helpers.metric_base(O, batch)
    points = batch["points"]
    yList = base_dict["yList"]
    xList = base_dict["xList"]
    propDict = base_dict["propDict"]
    yList = base_dict["yList"]
    background = base_dict["background"]


    bg_dict = helpers.metric_bg(O, batch)
    ###################################
    n,c,h,w = O.shape
    fg_seeds = O[:, :, yList, xList]
    n_seeds = fg_seeds.shape[-1]
    prop_mask = np.zeros((h, w))

    loss = torch.tensor(0.).cuda()

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
        
        ap = - torch.log(au.log_pairwise(f_A, fg_P)) 
        loss += ap.mean()

        # Get Negatives
        if n_seeds > 1:
            N_ind = [j for j in range(n_seeds) if j != i]
            f_N = fg_seeds[:,:,N_ind]
            an = - torch.log(1. - au.log_pairwise(f_A, f_N)) 
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
    ap = - torch.log(au.log_pairwise(f_A[:,:,None], f_P[:,:,:,None])) 
    an = - torch.log(1. - au.log_pairwise(f_A[:,:,None], fg_seeds[:,:,:,None])) 

    loss += ap.mean()
    loss += an.mean()

    if batch["dataset"][0] == "cityscapes" or batch["dataset"][0] == "coco2014":        
        n_max = 6
    else:
        n_max = 6

    if f_A.shape[2] < n_max:
        with torch.no_grad():

            diff = au.log_pairwise(O.view(1,c,-1)[:,:,:,None], 
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

            ap = - torch.log(au.log_pairwise(f_A[:,:,None], f_P[:,:,:,None])) 
            an = - torch.log(1. - au.log_pairwise(f_P[:,:,None], fg_seeds[:,:,:,None])) 

            # if i < 3:
            loss += ap.mean()
            loss += an.mean()

    import ipdb; ipdb.set_trace()  # breakpoint 42b2a648 //

    return loss / max(n_seeds, 1)

def OneHeadLoss_prototypes(model, batch, visualize=False):
    n,c,h,w = batch["images"].shape

    model.eval()
    O_dict = model(batch["images"].cuda())
    O = O_dict["embedding_mask"]
    
    loss = torch.tensor(0.).cuda()

    base_dict = helpers.metric_base(O, batch)
    if base_dict is None:
        return loss

    points = batch["points"]
    yList = base_dict["yList"]
    xList = base_dict["xList"]
    propDict = base_dict["propDict"]
    background = base_dict["background"]
    yList = base_dict["yList"]

    # foreground = distance_transform_cdt(1 - background)
    ###################################
    n,c,h,w = O.shape
    
    fg_seeds = O[:, :, yList, xList]
    n_seeds = fg_seeds.shape[-1]

    

    for i in range(n_seeds):
        annList = propDict[i]["annList"]

        if len(annList) == 0:
            mask = np.zeros(points.squeeze().shape)
            mask[propDict[i]["point"]["y"], propDict[i]["point"]["x"]] = 1
        else:
            mask = annList[0]["mask"]
       
        mask_ind = np.where(mask)
        
        f_A = fg_seeds[:,:,[i]]
        
        # Positive Embeddings
        fg_P = O[:,:,mask_ind[0], mask_ind[1]]
        ap = - torch.log(au.log_pairwise_sum(f_A, fg_P)) 
        loss += ap.mean()

        # Get Negatives
        mask_ind = np.where(1 - mask)
        f_N = O[:,:,mask_ind[0], mask_ind[1]]
        an = - torch.log(1. - au.log_pairwise_sum(f_A, f_N)) 
        loss += an.mean()

    n_sp = 0
    background = base_dict["background"]
    blobs, categoryDict, propDict = models.pairwise.prototype_predict(model, batch, 
                                   pointList=au.mask2pointList(points)["pointList"],
                                   visualize=False)
    if background.mean() != 1:
        bg_dict = helpers.get_bg_dict(base_dict["background"])
        test_mask = np.zeros(base_dict["background"].shape).squeeze()

        f_P = O[:,:,bg_dict["mask_pos"][0], bg_dict["mask_pos"][1]]
        f_N = O[:,:,bg_dict["mask_neg"][0], bg_dict["mask_neg"][1]]

        n_sp = 0
        for y, x in zip(bg_dict["yList"], bg_dict["xList"]):
            # print(y, x)
            n_sp += 1

            test_mask[y, x] = 1
            f_A = O[:, :, [y], [x]]
            
            # Positive Embeddings
            
            ap = - torch.log(au.log_pairwise_sum(f_A, f_P)) 
            loss += ap.mean()

            # Get Negatives
            an = - torch.log(1. - au.log_pairwise_sum(f_A, f_N)) 
            loss += an.mean()

        # try:
        #     assert test_mask.sum() == (test_mask*base_dict["background"]).sum()
        # except Exception as exc:
        # Refinement


        ind = np.where((blobs!=0).squeeze()&(background==1).squeeze())
        f_P = O[:,:,ind[0], ind[1]]
        if ind[0].size != 0:
            for y, x in zip(bg_dict["yList"], bg_dict["xList"]):
                test_mask[y, x] = 1
                f_A = O[:, :, [y], [x]]
                
                # Positive Embeddings
                ap = - torch.log(au.log_pairwise_sum(f_A, f_P)) 
                loss += ap.mean()

        # Connected components
        # ind = np.where((blobs!=0).squeeze()&(background==1).squeeze())
        # if ind[0].size != 0:
            # f_P = O[:,:,ind[0], ind[1]]
            # for y, x in zip(bg_dict["yList"], bg_dict["xList"]):
            #     # print(y, x)
            #     n_sp += 1

            #     test_mask[y, x] = 1
            #     f_A = O[:, :, [y], [x]]
                
            #     # Positive Embeddings
            #     ap = - torch.log(au.log_pairwise_sum(f_A, f_P)) 
            #     loss += ap.mean()

    for l in np.unique(blobs):
        if l == 0:
            continue
        cc = label(blobs == l)
        
        point = propDict["propDict"][l-1]["point"] 
        y, x = point["y"], point["x"]
        true = cc[y, x]
        f_A = O[:, :, [y], [x]]
        print(np.unique(cc).size)
        for lc in np.unique(cc):
            if lc == 0 or lc == true:
                continue
            else:
                # Get Negatives
                n_sp+=1 
                ind = np.where((cc==lc).squeeze())
                f_N = O[:,:,ind[0], ind[1]]
                an = - torch.log(1. - au.log_pairwise_sum(f_A, f_N)) 
                loss += an.max()



    return loss / max(n_seeds + n_sp, 1)

def se_pairwise(fi, fj):
    return (fi - fj).pow(2).sum(1)

def log_pairwise(fi, fj):
    diff = se_pairwise(fi, fj)
    return  (2./ (1+torch.exp(diff))).clamp(min=1e-6, max=(1.-1e-6))

def mask2pointList(mask):
    pointList = []
    mask = t2n(mask)
    pointInd = np.where(mask.squeeze())
    n_points = pointInd[0].size

    for p in range(n_points):

        p_y, p_x = pointInd[0][p], pointInd[1][p]
        point_category = mask[0, p_y,p_x]

        pointList += [{"y":p_y,"x":p_x, "category_id":int(point_category), 
                       "shape":mask.shape}]

    return {"pointList":pointList}


def metric_base(O, batch):
    n,c,h,w = O.shape

    points = batch["points"]
    batch["maskObjects"] = None 
    batch['maskClasses'] = None
    batch["maskVoid"] = None

    pointList = mask2pointList(points)["pointList"]

    
    if len(pointList) == 0:
        return None

    if "single_point" in batch:
        single_point = True
    else:
        single_point = False


    propDict = pointList2propDict(pointList, batch, 
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



def pointList2propDict(pointList, batch, single_point=False, thresh=0.5, 
                        proposal_type="sharp"):

    proposals = batch2proposals(batch, proposal_type=proposal_type)

    propDict = []
    shape = pointList[0]["shape"]
    foreground = np.zeros(shape, int)


    idDict= {}
    annDict = {}
    for i, p in enumerate(pointList):
        annDict[i] = []
        idDict[i] = []

    n_points = len(annDict)
    for k in range(len(proposals)):
        proposal_ann = proposals[k]

        if not (proposal_ann["score"] > thresh):
            continue

        proposal_mask =  proposal_ann["mask"]

        for i, p in enumerate(pointList):
            if proposal_mask[p["y"], p["x"]]==0:
                continue
            
            # score = proposal_ann["score"]
           
            annDict[i] += [proposal_ann]
            idDict[i] += [k]

    for i in range(n_points):
        point_annList = annDict[i]
        point_idList = idDict[i]
        p = pointList[i]

        mask = annList2mask(point_annList)["mask"]

        if mask is not None:
            foreground = foreground + mask

        #foreground[foreground<2]=0
        propDict += [{"annList":point_annList,"point":p, "idList":point_idList, 
                      "category_id":int(p["category_id"])}]
        #########  

    return {"propDict":propDict,"foreground":foreground, "background":(foreground==0).astype(int)}

def annList2mask(annList):
    n_anns = len(annList)
    if n_anns == 0:
        return {"mask":None}

    ann = annList[0]
    try:
        h, w = ann["mask"].shape
    except:
        h, w = ann["height"], ann["width"]
    mask = np.zeros((h, w), int)

    for i in range(n_anns):
        ann = annList[i]

        if "mask" in ann:
            ann_mask = ann["mask"]
        else:
            ann_mask = maskUtils.decode(ann["segmentation"])

        assert ann_mask.max() <= 1
        mask += ann_mask

    # mask[mask==1] = ann["category_id"]
    return {"mask":mask}


#### MISC 
def batch2proposals(batch, proposal_type):
    if proposal_type == "sharp":
        print("Sharp used")
        proposals = SharpProposals(batch)
    else:
        import ipdb; ipdb.set_trace()  # breakpoint 8e909a15 //


    return proposals

import os
import json

def load_json(fname, decode=None):
    with open(fname, "r") as json_file:
        d = json.load(json_file)
        
    return d

class SharpProposals:
    def __init__(self, batch):
        # if dataset_name == "pascal":
        self.proposals_path = batch["proposals_path"][0]

        if "SharpProposals_name" in batch:
            batch_name = batch["SharpProposals_name"][0]
        else:
            batch_name = batch["name"][0]
        name_jpg = self.proposals_path + "{}.jpg.json".format(batch_name)
        name_png = self.proposals_path + "{}.json".format(batch_name)
        
        if os.path.exists(name_jpg):
            name = name_jpg
        else:
            name = name_png

            
        _, _, self.h, self.w = batch["images"].shape

        if "resized" in batch and batch["resized"].item() == 1:
            name_resized = self.proposals_path + "{}_{}_{}.json".format(batch["name"][0], 
                                                                        self.h, self.w)
  
        else:
            name_resized = name
        # name_resized = name         
        proposals = load_json(name_resized)
        self.proposals = sorted(proposals, key=lambda x:x["score"], 
                                reverse=True)         

    def __getitem__(self, i):
        encoded = self.proposals[i]["segmentation"]
        proposal_mask = maskUtils.decode(encoded)
        
        return {"mask":proposal_mask, 
                "score":self.proposals[i]["score"]}


    def __len__(self):
        return len(self.proposals)