import torch
import tqdm
from torch import nn
from torch.nn import functional as F
import math
from PIL import Image
import os
import numpy as np
import torchvision
from src import proposals
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision import transforms
from haven import haven_utils as hu
from haven import haven_img as hi

from src import utils as ut
from src.metrics import ap



class InstSeg(nn.Module):
    def __init__(self, n_classes=10, exp_dict=None):
        super().__init__()
        self.n_classes = n_classes
        self.exp_dict = exp_dict
        self.model = model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained_backbone=True)

        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, n_classes)

        # now get the number of input features for the mask classifier
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # and replace the mask predictor with a new one
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                        hidden_layer,
                                                        n_classes)
        params = [p for p in model.parameters() if p.requires_grad]

        self.opt = torch.optim.SGD(params, lr=0.005,
                                    momentum=0.9, weight_decay=0.0005)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=3, gamma=0.1)

        warmup_factor = 1. / 1000
        warmup_iters = 1000

        self.lr_scheduler = warmup_lr_scheduler(self.opt, warmup_iters, warmup_factor)


    def get_state_dict(self):
        return {'model': self.state_dict(),
                'opt': self.opt.state_dict()} 

    def set_state_dict(self, state_dict):
        self.load_state_dict(state_dict['model'])
        self.opt.load_state_dict(state_dict['opt'])

    def train_on_batch(self, batch):
        """Train for one batch."""
        self.train()
        images, targets = batch['images'], batch['targets']

        if targets[0] is None:
            return 0.

        images = list(image.to('cuda:0') for image in images)
        targets = [{k: v.to('cuda:0') for k, v in t.items()} for t in targets]

        self.opt.zero_grad()
        loss_dict = self.model(images, targets)
        losses_sum = sum(loss for loss in loss_dict.values())

        if torch.isnan(losses_sum):
            return 0.

        losses_sum.backward()

        self.opt.step()
        self.lr_scheduler.step()

        return losses_sum.item()
        
    def train_on_loader(self, train_loader):
        """Train for one epoch."""
        self.train()
        loss_sum = 0.

        n_batches = len(train_loader)
        pbar = tqdm.tqdm(total=n_batches)
        for i, batch in enumerate(train_loader):
            loss_sum += float(self.train_on_batch(batch))

            pbar.set_description("Training loss: %.4f" % (loss_sum / (i + 1)))
            pbar.update(1)

        pbar.close()
        loss = loss_sum / n_batches

        return {"train_loss": loss}
    
    @torch.no_grad()
    def val_on_loader(self, val_loader, savedir='savedir'):
        """Validate the model."""
        self.eval()

        val_monitor = ap.APMonitor()

        n_batches = len(val_loader)
        pbar = tqdm.tqdm(total=n_batches)
        for i, batch in enumerate(val_loader):
            _, val_monitor = self.val_on_batch(batch, val_monitor=val_monitor)
            
            pbar.set_description("Val score: %.3f" % val_monitor.get_avg_score()['val_score'])
            pbar.update(1)


        pbar.close()
        val_dict_segm = val_monitor.get_avg_score(iouType='segm')
        val_dict_bbox = val_monitor.get_avg_score(iouType='bbox')
        metrics = ['mAP25','mAP50','mAP75']
        return {'val_score':val_dict_bbox['mAP50'], 
                'mAP_segm':tuple(["%.3f" % val_dict_segm[m] for m in metrics]), 
                'mAP_bbox':tuple(["%.3f" % val_dict_bbox[m] for m in metrics])}

    @torch.no_grad()
    def val_on_batch(self, batch, val_monitor=None, iouType='segm'):
        self.eval()
        if val_monitor is None:
            val_monitor = ap.APMonitor()

        gt_ann_list = ap.targets_to_ann_list(batch['images'], batch['targets'])

        preds = self.predict_on_batch(batch)
        preds_dict = {target["image_id"].item(): pred for target, pred in zip(batch['targets'], preds)}

        pred_ann_list = ap.preds_to_ann_list(preds_dict, mask_void=batch['mask_void'])

        val_monitor.add(gt_ann_list=gt_ann_list, pred_ann_list=pred_ann_list)
        
        val_dict = val_monitor.get_avg_score(iouType=iouType)
        return val_dict, val_monitor

    @torch.no_grad()
    def vis_on_batch(self, batch, savedir):
        self.eval()
        os.makedirs(savedir, exist_ok=True)
        overlayed = hi.mask_on_image(batch['image_pil'][0], np.array(batch['inst_pil'][0]), add_bbox=True)
        overlayed = Image.fromarray((overlayed*255).astype('uint8'))

        images = batch['images']
        img = images[0]
        
        prediction = self.model([img.to('cuda:0')])
        org_img = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
        pred_list = []
        score_list = prediction[0]['scores']
        for i in range(len(score_list)):
            if score_list[i] < 0.5:
                break
            pred = ((prediction[0]['masks'][i, 0]> 0.5).mul(255) ).byte().cpu().numpy()
            pred_list += [Image.fromarray(pred)]
        
        img_name = batch['meta'][0]['name']
        for pred in pred_list:
            org_img = Image.fromarray((hi.mask_on_image(org_img, pred)*255).astype('uint8'))

        fname = os.path.join(savedir, '%s.jpg' % img_name)
        
        overlayed = hi.text_on_image('gt', np.array(overlayed.resize((350,200), 2).copy()))   
        org_img = hi.text_on_image('preds', np.array(org_img.resize((350,200))).copy())   
        
        img = np.concatenate([org_img, overlayed.astype(float)], axis=1).astype('float32') / 255.
        hu.save_image(fname=fname, img=img)
        print('image saved: %s' % fname)

    @torch.no_grad()
    def vis_on_loader(self, vis_loader, savedir, n_images=3):
        self.eval()

        n_batches = len(vis_loader)
        # split = vis_loader.dataset.split
        for i, batch in enumerate(vis_loader):
            # print("%d - visualizing %s image - savedir:%s" % (i, batch["meta"]["split"][0], savedir.split("/")[-2]))
            self.vis_on_batch(batch, savedir=savedir)
            if (i+1) >= n_images:
                break

    def predict_on_batch(self, batch, **options):
        """Predict for one batch."""
        self.eval() 
        
        images = batch['images']
        images = list(image.to('cuda:0') for image in images)

        preds = self.model(images)
        cpu_device = torch.device("cpu")
        preds = [{k: v.to(cpu_device) for k, v in t.items()} for t in preds]

        return preds

def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):

    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)

def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


