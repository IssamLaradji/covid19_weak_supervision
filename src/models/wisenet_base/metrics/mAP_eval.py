
from __future__ import division

from collections import defaultdict
import numpy as np
import six

from chainercv.evaluations import calc_detection_voc_ap
from chainercv.utils.mask.mask_iou import mask_iou


def eval_instance_segmentation_voc(
        pred_masks, pred_labels, pred_scores,
        gt_masks, gt_labels,
        iou_thresh=0.5, use_07_metric=False,
        n_pos=None, score=None, match=None):

    n_pos, score, match = cal_running_instance(pred_masks, pred_labels, pred_scores,
                         gt_masks, gt_labels, iou_thresh,
                         n_pos, score, match)

    prec, rec = calc_instance_segmentation_voc_prec_rec(n_pos, score, match)

    ap = calc_detection_voc_ap(prec, rec, use_07_metric=use_07_metric)

    return {'ap': ap, 'map': np.nanmean(ap), 
            "n_pos":n_pos, "score":score, "match":match}


def cal_running_instance(pred_masks, pred_labels, pred_scores,
                         gt_masks, gt_labels, iou_thresh,
                         n_pos=None, score=None, match=None):
    pred_masks = iter(pred_masks)
    pred_labels = iter(pred_labels)
    pred_scores = iter(pred_scores)
    gt_masks = iter(gt_masks)
    gt_labels = iter(gt_labels)

    if n_pos is None:
        n_pos = defaultdict(int)
        score = defaultdict(list)
        match = defaultdict(list)

    for pred_mask, pred_label, pred_score, gt_mask, gt_label in \
            six.moves.zip(
                pred_masks, pred_labels, pred_scores,
                gt_masks, gt_labels):

        for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):
            pred_keep_l = pred_label == l
            pred_mask_l = pred_mask[pred_keep_l]
            pred_score_l = pred_score[pred_keep_l]
            # sort by score
            order = pred_score_l.argsort()[::-1]
            pred_mask_l = pred_mask_l[order]
            pred_score_l = pred_score_l[order]

            gt_keep_l = gt_label == l
            gt_mask_l = gt_mask[gt_keep_l]

            n_pos[l] += gt_keep_l.sum()
            score[l].extend(pred_score_l)

            if len(pred_mask_l) == 0:
                continue
            if len(gt_mask_l) == 0:
                match[l].extend((0,) * pred_mask_l.shape[0])
                continue

            iou = mask_iou(pred_mask_l, gt_mask_l)
            gt_index = iou.argmax(axis=1)
            # set -1 if there is no matching ground truth
            gt_index[iou.max(axis=1) < iou_thresh] = -1
            del iou

            selec = np.zeros(gt_mask_l.shape[0], dtype=bool)
            for gt_idx in gt_index:
                if gt_idx >= 0:
                    if not selec[gt_idx]:
                        match[l].append(1)
                    else:
                        match[l].append(0)
                    selec[gt_idx] = True
                else:
                    match[l].append(0)

    for iter_ in (pred_masks, pred_labels, pred_scores, gt_masks, gt_labels):
        if next(iter_, None) is not None:
            raise ValueError('Length of input iterables need to be same.')

    return n_pos, score, match

def calc_instance_segmentation_voc_prec_rec(n_pos, score, match):
    n_fg_class = max(n_pos.keys()) + 1
    prec = [None] * n_fg_class
    rec = [None] * n_fg_class

    for l in n_pos.keys():
        score_l = np.array(score[l])
        match_l = np.array(
            match[l], dtype=np.int8)

        order = score_l.argsort()[::-1]
        match_l = match_l[order]

        tp = np.cumsum(match_l == 1)
        fp = np.cumsum(match_l == 0)

        # If an element of fp + tp is 0,
        # the corresponding element of prec[l] is nan.
        prec[l] = tp / (fp + tp)
        # If n_pos[l] is 0, rec[l] is None.
        if n_pos[l] > 0:
            rec[l] = tp / n_pos[l]

    return prec, rec
