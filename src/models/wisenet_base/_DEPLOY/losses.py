
def OneHeadLoss(model, batch, visualize=False):
    n,c,h,w = batch["images"].shape

    model.train()
    O_dict = model(batch["images"].cuda())
    embedding_mask = O_dict["embedding_mask"]

    loss = helpers.compute_metric_loss(embedding_mask, batch)

    return loss


def compute_metric_loss(O, batch, random_proposal=False):
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
        n_max = 12

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

    # if visualize:
    #     diff = log_func(O.view(1,64,-1)[:,:,:,None], torch.cat([se, f_A], 2)[:,:,None])
    #     labels = diff.max(2)[1] + 1
    #     labels[labels > n_se] = 0
    #     labels = labels.squeeze().reshape(h,w)

    #     ms.images(batch["images"], ms.t2n(labels),denorm=1, win="labels")
    #     ms.images(batch["images"], prop_mask.astype(int), denorm=1, win="true")
    #     ms.images(batch["images"], background.astype(int), denorm=1, win="bg")


    return loss / max(n_seeds, 1)