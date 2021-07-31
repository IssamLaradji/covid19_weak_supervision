from collections import defaultdict

from scipy import spatial
import numpy as np
import torch
from . import struct_metric


class SegMeter:
    def __init__(self, split):
        self.cf = None
        self.struct_list = []
        self.split = split

    def val_on_batch(self, model, batch):
        masks_org = batch["masks"]

        pred_mask_org = model.predict_on_batch(batch)
        ind = masks_org != 255
        masks = masks_org[ind]
        pred_mask = pred_mask_org[ind]
        self.n_classes = model.n_classes
        if model.n_classes == 1:
            cf = confusion_binary_class(torch.as_tensor(pred_mask).float().cuda(), masks.cuda().float())
        else:
            labels = np.arange(model.n_classes)
            cf = confusion_multi_class(torch.as_tensor(pred_mask).float().cuda(), masks.cuda().float(),
                                    labels=labels)

        if self.cf is None:
            self.cf = cf
        else:
            self.cf += cf

        # structure
        struct_score = float(struct_metric.compute_struct_metric(pred_mask_org, masks_org))
        self.struct_list += [struct_score]

    def get_avg_score(self):
        TP = np.diag(self.cf)
        TP_FP = self.cf.sum(axis=1)
        TP_FN = self.cf.sum(axis=0)
        TN = TP[::-1]
        

        FP = TP_FP - TP
        FN = TP_FN - TP

        iou = TP / (TP + FP + FN)
        dice = 2*TP / (FP + FN + 2*TP)

        iou[np.isnan(iou)] = -1
        dice[np.isnan(dice)] = -1

        mDice = np.mean(dice)
        mIoU = np.mean(iou)

        prec = TP / np.maximum((TP + FP), 1e-8)
        recall = TP / np.maximum((TP + FN), 1e-8)
        spec = TN/ np.maximum((TN+FP), 1e-8)
        fscore = (( 2.0 * prec * recall ) / np.maximum((prec + recall), 1e-8))

        val_dict = {}
        if self.n_classes == 1:
            val_dict['%s_dice' % self.split] = dice[0]
            val_dict['%s_iou' % self.split] = iou[0]

            val_dict['%s_prec' % self.split] = prec[0]
            val_dict['%s_recall' % self.split] = recall[0]
            val_dict['%s_spec' % self.split] = spec[0]
            val_dict['%s_fscore' % self.split] = fscore[0]

            val_dict['%s_score' % self.split] = dice[0]
            val_dict['%s_struct' % self.split] = np.mean(self.struct_list)
        return val_dict

def confusion_multi_class(prediction, truth, labels):
    """
    cf = confusion_matrix(y_true=prediction.cpu().numpy().ravel(),
            y_pred=truth.cpu().numpy().ravel(),
                    labels=labels)
    """
    nclasses = labels.max() + 1
    cf2 = torch.zeros(nclasses, nclasses, dtype=torch.float,
                      device=prediction.device)
    prediction = prediction.view(-1).long()
    truth = truth.view(-1)
    to_one_hot = torch.eye(int(nclasses), dtype=cf2.dtype,
                           device=prediction.device)
    for c in range(nclasses):
        true_mask = (truth == c)
        pred_one_hot = to_one_hot[prediction[true_mask]].sum(0)
        cf2[:, c] = pred_one_hot

    return cf2.cpu().numpy()



def confusion_binary_class(pred_mask, gt_mask):
    intersect = pred_mask.bool() & gt_mask.bool()

    fp_tp = (pred_mask ==1).sum().item()
    fn_tp = gt_mask.sum().item()
    tn_fn = (pred_mask ==0).sum().item()

    tp = (intersect == 1).sum().item()
    fp = fp_tp - tp
    fn = fn_tp - tp
    tn = tn_fn - fn 

    cm = np.array([[tp, fp],
                   [fn, tn]])
    return cm