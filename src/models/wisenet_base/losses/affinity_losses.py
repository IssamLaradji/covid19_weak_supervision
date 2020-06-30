import torch
import numpy as np
import ann_utils as au
from skimage import morphology as morph
import torch.nn.functional as F
import misc as ms
def wiseaffinity_loss(model, batch):
    # model.lcfcn
    images = batch["images"].cuda()
    n,c,h,w = images.shape
    pointList = au.mask2pointList(batch["points"])["pointList"]

    loss = torch.tensor(0.).cuda()
    loss_bg = torch.tensor(0.).cuda()
    loss_fg = torch.tensor(0.).cuda()

    if len(pointList) == 0:
        return loss

    propDict = au.pointList2propDict(pointList, batch, 
                                     single_point=True,
                                     thresh=0.5)
    propList = propDict["propDict"]

    pred_dict = model.lcfcn.predict(batch, predict_method="pointList")
    blobs = pred_dict["blobs"]
    probs = pred_dict["probs"]
    blobs_components = morph.label(blobs!=0)
    image_pad = ms.pad_image(images)
    _, _, dheight, dwidth = image_pad.shape
    trans_mat = model.aff.forward_trans(image_pad)

    # bg_probs = probs[:,[0]]
    for i in range(len(propList)):
        prop = propList[i]
        if not len(prop["annList"]):
            continue
        proposal_mask = torch.LongTensor(prop["annList"][0]["mask"]).cuda()
        # proposal_mask = F.interpolate(proposal_mask, size=(dheight//8, dwidth//8))
        y, x = prop["point"]["y"], prop["point"]["x"]
        
        category = blobs[:,y,x][0]
        instance = blobs_components[:,y,x][0]
        blob_mask_ind = blobs_components == instance

        O = torch.FloatTensor(probs[:, [0, category]]).detach()
        O[:,1] = O[:,1] * torch.FloatTensor(blob_mask_ind.astype(float))
        O[:,1] = O[:,1].clamp(1e-10)

        O_scale = F.interpolate(O, size=(dheight//8, dwidth//8))
        O_scale = O_scale.view(1, 2, -1).cuda()

        O_rw = torch.matmul(O_scale, trans_mat)
        O_rw = O_rw.view(1, 2, dheight//8, dwidth//8)
        O_final = F.interpolate(O_rw, size=(h,w))

        S_log = F.log_softmax(O_final, 1)

        loss_bg += F.nll_loss(S_log, proposal_mask[None], 
                   ignore_index=1, size_average=True)
        loss_fg += F.nll_loss(S_log, proposal_mask[None], 
                   ignore_index=0, size_average=True)

    return (loss_bg + loss_fg)/len(propList)

def affinity_loss(model, batch):

    import ipdb; ipdb.set_trace()  # breakpoint 9874d40f //
    aff = model.predict(batch)
    aff = model.forward(batch[0])

    bg_label = batch[1][0].cuda(non_blocking=True)
    fg_label = batch[1][1].cuda(non_blocking=True)
    neg_label = batch[1][2].cuda(non_blocking=True)

    bg_count = torch.sum(bg_label) + 1e-5
    fg_count = torch.sum(fg_label) + 1e-5
    neg_count = torch.sum(neg_label) + 1e-5

    bg_loss = torch.sum(- bg_label * torch.log(aff + 1e-5)) / bg_count
    fg_loss = torch.sum(- fg_label * torch.log(aff + 1e-5)) / fg_count
    neg_loss = torch.sum(- neg_label * torch.log(1. + 1e-5 - aff)) / neg_count

    return bg_loss + fg_loss + neg_loss




def obtain_affinity(model, batch):

    import ipdb; ipdb.set_trace()  # breakpoint 9874d40f //
    
    aff = model.forward(batch[0])

    bg_label = batch[1][0].cuda(non_blocking=True)
    fg_label = batch[1][1].cuda(non_blocking=True)
    neg_label = batch[1][2].cuda(non_blocking=True)

    bg_count = torch.sum(bg_label) + 1e-5
    fg_count = torch.sum(fg_label) + 1e-5
    neg_count = torch.sum(neg_label) + 1e-5

    bg_loss = torch.sum(- bg_label * torch.log(aff + 1e-5)) / bg_count
    fg_loss = torch.sum(- fg_label * torch.log(aff + 1e-5)) / fg_count
    neg_loss = torch.sum(- neg_label * torch.log(1. + 1e-5 - aff)) / neg_count

    return bg_loss + fg_loss + neg_loss
