import torch
from datasets import base_dataset
import misc as ms
import ann_utils as au
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

def MaskRCNNLoss(model, batch, prm_points=False, true_annList=False, visualize=False):
    """
    - images: [batch, H, W, C]
    - image_metas: [batch, size of image meta]
    - rpn_match: [batch, N] Integer (1=positive anchor, -1=negative, 0=neutral)
    - rpn_bbox: [batch, N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
    - gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs
    - gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]
    - gt_masks: [batch, height, width, MAX_GT_INSTANCES]. The height and width
                are those of the image unless use_mini_mask is True, in which
                case they are defined in MINI_MASK_SHAPE.
    """
    # 1. Data Generator
    ###################
    images = batch["images"].cuda()
    if true_annList:

        gt_annList = base_dataset.batch2annList(batch)
    else:
        if prm_points:
            fname = "{}/prm{}.pkl".format(batch["path"][0], batch["name"][0]) 
            points = torch.LongTensor(ms.load_pkl(fname))[None].cuda()

        else:
            points = batch["points"].cuda()
        
        pointList = au.mask2pointList(points)["pointList"]

        if len(pointList) == 0:
            return 0
            
        gt_annList = au.pointList2BestObjectness(pointList, batch, proposal_type="sharp")["annList"]

    
    if len(gt_annList) == 0:
        return 0

    # ms.images(batch["images"], au.annList2mask(gt_annList)["mask"], denorm=1)
    # image_id = batch["name"][0].replace("_","")
    image_id = 1

    image, image_metas, gt_class_ids, gt_boxes, gt_masks = model.load_image_gt(images, image_id, gt_annList, augment=False)
    h, w = images.shape[-2:]
    rpn_match, rpn_bbox = build_rpn_targets((h,w,3), model.anchors,
                                                    gt_class_ids, gt_boxes,
                                                     model.config)
    
    # n_anns = len(gt_annList)
    # h, w = images.shape[-2:]
    # gt_class_ids = np.zeros((n_anns))
    # gt_boxes = np.zeros((n_anns, 4))
    # gt_masks = np.zeros((h, w, n_anns))

    # for i, ann in enumerate(gt_annList):
    #     gt_class_ids[i] = ann["category_id"]
    #     x_y_xe_ye = au.ann2bbox(ann)["shape"]
    #     # (x, y, xe, ye)
    #     y1,x1,y2,x2 = x_y_xe_ye[1], x_y_xe_ye[0], x_y_xe_ye[-1], x_y_xe_ye[-2]
    #     gt_boxes[i] = np.array((y1,x1,y2,x2))
    #     gt_masks[:,:,i] = au.ann2mask(ann)["mask"]


    # rpn_match, rpn_bbox = helpers.build_rpn_targets((h,w,3), model.anchors,
    #                                                 gt_class_ids, gt_boxes, model.config)
    
    # 2. RPN Stuff
    ##############

    # If more instances than fits in the array, sub-sample from them.
    if gt_boxes.shape[0] > model.config.MAX_GT_INSTANCES:
        ids = np.random.choice(
            np.arange(gt_boxes.shape[0]), model.config.MAX_GT_INSTANCES, replace=False)
        gt_class_ids = gt_class_ids[ids]
        gt_boxes = gt_boxes[ids]
        gt_masks = gt_masks[:, :, ids]

    # Add to batch
    rpn_match = rpn_match[:, np.newaxis]
    images = model.mold_image(image.astype(np.float32), model.config)

    # Convert
    images = torch.from_numpy(images.transpose(2, 0, 1)).float()[None].cuda()
    image_metas = torch.from_numpy(image_metas)[None].cuda()
    rpn_match = torch.from_numpy(rpn_match)[None].cuda()
    rpn_bbox = torch.from_numpy(rpn_bbox).float()[None].cuda()
    gt_class_ids = torch.from_numpy(gt_class_ids)[None].cuda().int()
    gt_boxes = torch.from_numpy(gt_boxes).float()[None].cuda()
    gt_masks = torch.from_numpy(gt_masks.astype(int).transpose(2, 0, 1)).float()[None].cuda()

    
    rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask = \
                model.forward_train({"images":images, 
                                     "gt_class_ids": gt_class_ids, 
                                     "gt_boxes":gt_boxes, "gt_masks":gt_masks})

    # 3. Compute Losses
    ###################

    rpn_class_loss = compute_rpn_class_loss(rpn_match, rpn_class_logits)
    rpn_bbox_loss = compute_rpn_bbox_loss(rpn_bbox, rpn_match, rpn_pred_bbox)
    
    mrcnn_class_loss = compute_mrcnn_class_loss(target_class_ids, mrcnn_class_logits)
    mrcnn_bbox_loss = compute_mrcnn_bbox_loss(target_deltas, target_class_ids, mrcnn_bbox)
    mrcnn_mask_loss = compute_mrcnn_mask_loss(target_mask, target_class_ids, mrcnn_mask)

    
    loss = rpn_class_loss + rpn_bbox_loss + mrcnn_class_loss + mrcnn_bbox_loss + mrcnn_mask_loss

    return loss 





def build_rpn_targets(image_shape, anchors, gt_class_ids, gt_boxes, config):
    """Given the anchors and GT boxes, compute overlaps and identify positive
    anchors and deltas to refine them to match their corresponding GT boxes.
    anchors: [num_anchors, (y1, x1, y2, x2)]
    gt_class_ids: [num_gt_boxes] Integer class IDs.
    gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]
    Returns:
    rpn_match: [N] (int32) matches between anchors and GT boxes.
               1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_bbox: [N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
    """
    # RPN Match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)
    # RPN bounding boxes: [max anchors per image, (dy, dx, log(dh), log(dw))]
    rpn_bbox = np.zeros((config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4))

    # Handle COCO crowds
    # A crowd box in COCO is a bounding box around several instances. Exclude
    # them from training. A crowd box is given a negative class ID.
    crowd_ix = np.where(gt_class_ids < 0)[0]
    if crowd_ix.shape[0] > 0:
        # Filter out crowds from ground truth class IDs and boxes
        non_crowd_ix = np.where(gt_class_ids > 0)[0]
        crowd_boxes = gt_boxes[crowd_ix]
        gt_class_ids = gt_class_ids[non_crowd_ix]
        gt_boxes = gt_boxes[non_crowd_ix]
        # Compute overlaps with crowd boxes [anchors, crowds]
        crowd_overlaps = compute_overlaps(anchors, crowd_boxes)
        crowd_iou_max = np.amax(crowd_overlaps, axis=1)
        no_crowd_bool = (crowd_iou_max < 0.001)
    else:
        # All anchors don't intersect a crowd
        no_crowd_bool = np.ones([anchors.shape[0]], dtype=bool)

    # Compute overlaps [num_anchors, num_gt_boxes]
    overlaps = compute_overlaps(anchors, gt_boxes)

    # Match anchors to GT Boxes
    # If an anchor overlaps a GT box with IoU >= 0.7 then it's positive.
    # If an anchor overlaps a GT box with IoU < 0.3 then it's negative.
    # Neutral anchors are those that don't match the conditions above,
    # and they don't influence the loss function.
    # However, don't keep any GT box unmatched (rare, but happens). Instead,
    # match it to the closest anchor (even if its max IoU is < 0.3).
    #
    # 1. Set negative anchors first. They get overwritten below if a GT box is
    # matched to them. Skip boxes in crowd areas.
    anchor_iou_argmax = np.argmax(overlaps, axis=1)
    anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
    rpn_match[(anchor_iou_max < 0.3) & (no_crowd_bool)] = -1
    # 2. Set an anchor for each GT box (regardless of IoU value).
    # TODO: If multiple anchors have the same IoU match all of them
    gt_iou_argmax = np.argmax(overlaps, axis=0)
    rpn_match[gt_iou_argmax] = 1
    # 3. Set anchors with high overlap as positive.
    rpn_match[anchor_iou_max >= 0.7] = 1

    # Subsample to balance positive and negative anchors
    # Don't let positives be more than half the anchors
    ids = np.where(rpn_match == 1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2)
    if extra > 0:
        # Reset the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0
    # Same for negative proposals
    ids = np.where(rpn_match == -1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE -
                        np.sum(rpn_match == 1))
    if extra > 0:
        # Rest the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0

    # For positive anchors, compute shift and scale needed to transform them
    # to match the corresponding GT boxes.
    ids = np.where(rpn_match == 1)[0]
    ix = 0  # index into rpn_bbox
    # TODO: use box_refinment() rather than duplicating the code here
    for i, a in zip(ids, anchors[ids]):
        # Closest gt box (it might have IoU < 0.7)
        gt = gt_boxes[anchor_iou_argmax[i]]

        # Convert coordinates to center plus width/height.
        # GT Box
        gt_h = gt[2] - gt[0]
        gt_w = gt[3] - gt[1]
        gt_center_y = gt[0] + 0.5 * gt_h
        gt_center_x = gt[1] + 0.5 * gt_w
        # Anchor
        a_h = (a[2] - a[0]).item()
        a_w = (a[3] - a[1]).item()
        a_center_y = (a[0] + 0.5 * a_h).item()
        a_center_x = (a[1] + 0.5 * a_w).item()

        # Compute the bbox refinement that the RPN should predict.
        rpn_bbox[ix] = [
            (gt_center_y - a_center_y) / a_h,
            (gt_center_x - a_center_x) / a_w,
            np.log(gt_h / a_h),
            np.log(gt_w / a_w),
        ]
        # Normalize
        rpn_bbox[ix] /= config.RPN_BBOX_STD_DEV
        ix += 1

    return rpn_match, rpn_bbox

def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.

    Note: the areas are passed in rather than calculated here for
          efficency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = (np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)).float()
    union = float(box_area) + (boxes_area[:].cpu() - intersection[:])
    iou = intersection / union
    return iou


def compute_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].

    For better performance, pass the largest set first and the smaller second.
    """
    # Areas of anchors and GT boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
    return overlaps

def compute_rpn_class_loss(rpn_match, rpn_class_logits):
    """RPN anchor classifier loss.
    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_class_logits: [batch, anchors, 2]. RPN classifier logits for FG/BG.
    """

    # Squeeze last dim to simplify
    rpn_match = rpn_match.squeeze(2)

    # Get anchor classes. Convert the -1/+1 match to 0/1 values.
    anchor_class = (rpn_match == 1).long()

    # Positive and Negative anchors contribute to the loss,
    # but neutral anchors (match value = 0) don't.
    indices = torch.nonzero(rpn_match != 0)

    # Pick rows that contribute to the loss and filter out the rest.
    rpn_class_logits = rpn_class_logits[indices.data[:,0],indices.data[:,1],:]
    anchor_class = anchor_class[indices.data[:,0],indices.data[:,1]]

    # Crossentropy loss
    loss = F.cross_entropy(rpn_class_logits, anchor_class)

    return loss

def compute_rpn_bbox_loss(target_bbox, rpn_match, rpn_bbox):
    """Return the RPN bounding box loss graph.
    target_bbox: [batch, max positive anchors, (dy, dx, log(dh), log(dw))].
        Uses 0 padding to fill in unsed bbox deltas.
    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
    """

    # Squeeze last dim to simplify
    rpn_match = rpn_match.squeeze(2)

    # Positive anchors contribute to the loss, but negative and
    # neutral anchors (match value of 0 or -1) don't.
    indices = torch.nonzero(rpn_match==1)

    # Pick bbox deltas that contribute to the loss
    rpn_bbox = rpn_bbox[indices.data[:,0],indices.data[:,1]]

    # Trim target bounding box deltas to the same length as rpn_bbox.
    target_bbox = target_bbox[0,:rpn_bbox.size()[0],:]

    # Smooth L1 loss
    loss = F.smooth_l1_loss(rpn_bbox, target_bbox)

    return loss


def compute_mrcnn_class_loss(target_class_ids, pred_class_logits):
    """Loss for the classifier head of Mask RCNN.
    target_class_ids: [batch, num_rois]. Integer class IDs. Uses zero
        padding to fill in the array.
    pred_class_logits: [batch, num_rois, num_classes]
    """

    # Loss
    if np.sum(target_class_ids.size()):
        loss = F.cross_entropy(pred_class_logits,target_class_ids.long())
    else:
        loss = Variable(torch.FloatTensor([0]), requires_grad=False)
        loss = loss.cuda()

    return loss


def compute_mrcnn_bbox_loss(target_bbox, target_class_ids, pred_bbox):
    """Loss for Mask R-CNN bounding box refinement.
    target_bbox: [batch, num_rois, (dy, dx, log(dh), log(dw))]
    target_class_ids: [batch, num_rois]. Integer class IDs.
    pred_bbox: [batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))]
    """

    if np.sum(target_class_ids.size()):
        # Only positive ROIs contribute to the loss. And only
        # the right class_id of each ROI. Get their indicies.
        positive_roi_ix = torch.nonzero(target_class_ids > 0)[:, 0]
        positive_roi_class_ids = target_class_ids[positive_roi_ix.data].long()
        indices = torch.stack((positive_roi_ix,positive_roi_class_ids), dim=1)

        # Gather the deltas (predicted and true) that contribute to loss
        target_bbox = target_bbox[indices[:,0].data,:]
        pred_bbox = pred_bbox[indices[:,0].data,indices[:,1].data,:]

        # Smooth L1 loss
        loss = F.smooth_l1_loss(pred_bbox, target_bbox)
    else:
        loss = Variable(torch.FloatTensor([0]), requires_grad=False)
        if target_class_ids.is_cuda:
            loss = loss.cuda()

    return loss


def compute_mrcnn_mask_loss(target_masks, target_class_ids, pred_masks):
    """Mask binary cross-entropy loss for the masks head.
    target_masks: [batch, num_rois, height, width].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_masks: [batch, proposals, height, width, num_classes] float32 tensor
                with values from 0 to 1.
    """
    if np.sum(target_class_ids.size()):
        # print(True)
        # Only positive ROIs contribute to the loss. And only
        # the class specific mask of each ROI.
        positive_ix = torch.nonzero(target_class_ids > 0)[:, 0]
        positive_class_ids = target_class_ids[positive_ix.data].long()
        indices = torch.stack((positive_ix, positive_class_ids), dim=1)

        # Gather the masks (predicted and true) that contribute to loss
        y_true = target_masks[indices[:,0].data,:,:]
        y_pred = pred_masks[indices[:,0].data,indices[:,1].data,:,:]

        # Binary cross entropy
        loss = F.binary_cross_entropy(y_pred, y_true)
    else:
        loss = Variable(torch.FloatTensor([0]), requires_grad=False)
        if target_class_ids.is_cuda:
            loss = loss.cuda()

    return loss

def compute_losses(rpn_match, rpn_bbox, rpn_class_logits, rpn_pred_bbox, target_class_ids, 
                   mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask):

    rpn_class_loss = compute_rpn_class_loss(rpn_match, rpn_class_logits)
    rpn_bbox_loss = compute_rpn_bbox_loss(rpn_bbox, rpn_match, rpn_pred_bbox)
    mrcnn_class_loss = compute_mrcnn_class_loss(target_class_ids, mrcnn_class_logits)
    mrcnn_bbox_loss = compute_mrcnn_bbox_loss(target_deltas, target_class_ids, mrcnn_bbox)
    mrcnn_mask_loss = compute_mrcnn_mask_loss(target_mask, target_class_ids, mrcnn_mask)

    return [rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss]
