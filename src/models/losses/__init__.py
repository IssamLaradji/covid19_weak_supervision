import torch
import torch.nn.functional as F
from . import dice_loss, boundary_loss, surface_loss
# from kornia.losses import focal
from . import  density_loss
from src.modules.lcfcn import lcfcn_loss

def compute_loss(loss_name, images, logits, labels, batch=None):
    if loss_name == 'density':
        loss = density_loss.compute_density_loss(logits, torch.LongTensor(batch['points'][0]).cuda())
        return loss

    if loss_name == 'lcfcn':
        loss = lcfcn_loss.compute_lcfcn_loss(logits, torch.LongTensor(batch['points'][0]).cuda())
        # lcfcn_loss.save_tmp('tmp.png', images, logits, radius=2, points=torch.LongTensor(batch['points'][0]))
        return loss

    if loss_name == 'cross_entropy':
        probs = F.log_softmax(logits, dim=1)
        loss = F.nll_loss(
            probs, labels, reduction='mean', ignore_index=255)
        return loss

    if loss_name == 'cross_entropy':
        probs = F.log_softmax(logits, dim=1)
        loss = F.nll_loss(
            probs, labels, reduction='mean', ignore_index=255)
        return loss

    if loss_name == 'balanced_cross_entropy':
        probs = F.log_softmax(logits, dim=1)
        loss = 0.
        for l in labels.unique():
            ind = labels[0] == l
            loss += F.nll_loss(
                probs[:,:,ind], labels[:,ind], reduction='mean', ignore_index=255)
        return loss

    if loss_name == 'dice':
        probs = F.softmax(logits, dim=1)
        loss = 0.
        for l in labels.unique():
            if l == 255:
                continue
            ind = labels == l
            loss += dice_loss.dice_loss(probs[:, l][ind],
                                        (labels[ind] == l).long()) 
        return loss

    if loss_name == 'focal':
        assert(255 not in labels)
        
        loss = focal.focal_loss(logits, labels, alpha=1, gamma=2, reduction='mean')
        return loss
    
    if loss_name == 'boundary':
        assert(255 not in labels)
        
        loss = boundary_loss.boundary_loss(logits, labels)
        return loss

    if loss_name == 'surface_2':
        assert(255 not in labels)
        loss_fn = surface_loss.SurfaceLoss(idc=[2])
        probs = F.softmax(logits, dim=1)
        one_hot = surface_loss.class2one_hot(labels, C=3)
        distmap = torch.FloatTensor(surface_loss.one_hot2dist(one_hot.cpu().numpy()[0]))
        loss = loss_fn(probs, distmap[None].cuda(), _=None)
        return loss
    
    if loss_name == 'surface_12':
        assert(255 not in labels)
        loss_fn = surface_loss.SurfaceLoss(idc=[1,2])
        probs = F.softmax(logits, dim=1)
        one_hot = surface_loss.class2one_hot(labels, C=3)
        distmap = torch.FloatTensor(surface_loss.one_hot2dist(one_hot.cpu().numpy()[0]))
        loss = loss_fn(probs, distmap[None].cuda(), _=None)
        return loss

    if loss_name == 'surface_all':
        assert(255 not in labels)
        loss_fn = surface_loss.SurfaceLoss(idc=[0, 1,2])
        probs = F.softmax(logits, dim=1)
        one_hot = surface_loss.class2one_hot(labels, C=3)
        distmap = torch.FloatTensor(surface_loss.one_hot2dist(one_hot.cpu().numpy()[0]))
        loss = loss_fn(probs, distmap[None].cuda(), _=None)
        return loss