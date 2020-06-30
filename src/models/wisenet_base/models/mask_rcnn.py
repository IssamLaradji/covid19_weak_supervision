
# import datetime
# import math
# import os
# import random
# import re
# import misc as ms
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import torch.utils.data
# from torch.autograd import Variable
# from . import base_model as bm
# # from nms.nms_wrapper import nms
# import ann_utils as au
# from compiled.roialign.roi_align.crop_and_resize import CropAndResizeFunction
# from compiled.nms.nms_wrapper import nms
# import torchvision
# class MaskRCNN(bm.BaseModel):

#     class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
#                    'bus', 'train', 'truck', 'boat', 'traffic light',
#                    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
#                    'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
#                    'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
#                    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
#                    'kite', 'baseball bat', 'baseball glove', 'skateboard',
#                    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
#                    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
#                    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
#                    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
#                    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
#                    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
#                    'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
#                    'teddy bear', 'hair drier', 'toothbrush']


#     def __init__(self, train_set, **model_options):

#         super().__init__(train_set, **model_options)
#         # Image size must be dividable by 2 multiple times
#         config = self.config = Config()
#         config.NUM_CLASSES = train_set.n_classes
#         self.config.NUM_CLASSES = train_set.n_classes
#         h, w = config.IMAGE_SHAPE[:2]

#         if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
#             raise Exception("Image size must be dividable by 2 at least 6 times "
#                             "to avoid fractions when downscaling and upscaling."
#                             "For example, use 256, 320, 384, 448, 512, ... etc. ")

#         # Build the shared convolutional layers.
#         # Bottom-up Layers

        

#         resnet = torchvision.models.resnet50(pretrained=True)
#         C1 = nn.Sequential(resnet.conv1,
#                            resnet.bn1, 
#                            resnet.relu, 
#                            resnet.maxpool)

#         C2 = resnet.layer1
#         C3 = resnet.layer2
#         C4 = resnet.layer3
#         C5 = resnet.layer4

#         # Top-down Layers
#         # TODO: add assert to varify feature map sizes match what's in config
#         self.fpn = FPN(C1, C2, C3, C4, C5, out_channels=256)

#         self.anchors = Variable(torch.from_numpy(generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
#                                                                                 config.RPN_ANCHOR_RATIOS,
#                                                                                 config.BACKBONE_SHAPES,
#                                                                                 config.BACKBONE_STRIDES,
#                                                                                 config.RPN_ANCHOR_STRIDE)).float(), requires_grad=False)
       
#         self.anchors = self.anchors.cuda()

#         # RPN
#         self.rpn = RPN(len(config.RPN_ANCHOR_RATIOS), config.RPN_ANCHOR_STRIDE, 256)

#         # FPN Classifier
#         self.classifier = Classifier(256, config.POOL_SIZE, config.IMAGE_SHAPE, config.NUM_CLASSES)

#         # FPN Mask
#         self.mask = Mask(256, config.MASK_POOL_SIZE, config.IMAGE_SHAPE, config.NUM_CLASSES)

#         # Fix batch norm layers
#         def set_bn_fix(m):
#             classname = m.__class__.__name__
#             if classname.find('BatchNorm') != -1:
#                 for p in m.parameters(): p.requires_grad = False

#         self.apply(set_bn_fix)
#         for m in self.modules():
#             if isinstance(m, nn.BatchNorm2d):
#                 m.weight.requires_grad = False
#                 m.bias.requires_grad = False
#         COCO_MODEL_PATH = "/mnt/projects/counting/Summaries/mask_rcnn_coco.pth"
#         # self.load_state_dict(torch.load(COCO_MODEL_PATH))

#     def extract_rpn(self, molded_images, proposal_count):
#       # Feature extraction
#       [p2_out, p3_out, p4_out, p5_out, p6_out] = self.fpn(molded_images)

#       # Note that P6 is used in RPN, but not in the classifier heads.
#       rpn_feature_maps = [p2_out, p3_out, p4_out, p5_out, p6_out]
#       mrcnn_feature_maps = [p2_out, p3_out, p4_out, p5_out]

#       # Loop through pyramid layers
#       layer_outputs = []  # list of lists
#       for p in rpn_feature_maps:
#           layer_outputs.append(self.rpn(p))

#       # Concatenate layer outputs
#       # Convert from list of lists of level outputs to list of lists
#       # of outputs across levels.
#       # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
#       outputs = list(zip(*layer_outputs))
#       outputs = [torch.cat(list(o), dim=1) for o in outputs]
#       rpn_class_logits, rpn_class, rpn_bbox = outputs

#       # Generate proposals
#       # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
#       # and zero padded.
#       # proposal_count = self.config.POST_NMS_ROIS_TRAINING 
#       rpn_rois = proposal_layer([rpn_class, rpn_bbox],
#                                proposal_count=proposal_count,
#                                nms_threshold=self.config.RPN_NMS_THRESHOLD,
#                                anchors=self.anchors,
#                                config=self.config)

#       return {"rpn_rois":rpn_rois,  "rpn_class_logits":rpn_class_logits,  
#               "rpn_class":rpn_class, "rpn_bbox":rpn_bbox, "mrcnn_feature_maps":mrcnn_feature_maps}

#     def image2window(self, image):
#       x1, y1 = 0, 0
#       y2, x2 = image.shape[-2:]
#       return (x1, y1, x2, y2)



#     def mold_image(self, images, config):
#         """Takes RGB images with 0-255 values and subtraces
#         the mean pixel and converts it to float. Expects image
#         colors in RGB order.
#         """
#         images = images.astype(np.float32) 
#         if images.max() > 1:
#           return (images/ 255. - config.MEAN_PIXEL) / config.STD_PIXEL
#         else:
#           werwer


#     def mold_inputs(self, images):
#         """Takes a list of images and modifies them to the format expected
#         as an input to the neural network.
#         images: List of image matricies [height,width,depth]. Images can have
#             different sizes.
#         Returns 3 Numpy matricies:
#         molded_images: [N, h, w, 3]. Images resized and normalized.
#         image_metas: [N, length of meta data]. Details about each image.
#         windows: [N, (y1, x1, y2, x2)]. The portion of the image that has the
#             original image (padding excluded).
#         """
#         molded_images = []
#         image_metas = []
#         windows = []
#         for image in images:
#             # Resize image to fit the model expected size
#             # TODO: move resizing to mold_image()
#             molded_image, window, scale, padding = resize_image(
#                 image,
#                 min_dim=self.config.IMAGE_MIN_DIM,
#                 max_dim=self.config.IMAGE_MAX_DIM,
#                 padding=self.config.IMAGE_PADDING)
#             molded_image = self.mold_image(molded_image, self.config)
#             # Build image_meta
#             image_meta = compose_image_meta(
#                 0, image.shape, window,
#                 np.zeros([self.config.NUM_CLASSES], dtype=np.int32))
#             # Append
#             molded_images.append(molded_image)
#             windows.append(window)
#             image_metas.append(image_meta)
#         # Pack into arrays
#         molded_images = np.stack(molded_images)
#         image_metas = np.stack(image_metas)
#         windows = np.stack(windows)

#         return molded_images, image_metas, windows

#     def load_image_gt(self, image, image_id, annList,
#                       augment=False):
#         """Load and return ground truth data for an image (image, mask, bounding boxes).
#         augment: If true, apply random image augmentation. Currently, only
#             horizontal flipping is offered.
#         use_mini_mask: If False, returns full-size masks that are the same height
#             and width as the original image. These can be big, for example
#             1024x1024x100 (for 100 instances). Mini masks are smaller, typically,
#             224x224 and are generated by extracting the bounding box of the
#             object and resizing it to MINI_MASK_SHAPE.
#         Returns:
#         image: [height, width, 3]
#         shape: the original shape of the image before resizing and cropping.
#         class_ids: [instance_count] Integer class IDs
#         bbox: [instance_count, (y1, x1, y2, x2)]
#         mask: [height, width, instance_count]. The height and width are those
#             of the image unless use_mini_mask is True, in which case they are
#             defined in MINI_MASK_SHAPE.
#         """
#         image = ms.f2l(ms.t2n((ms.denormalize(image)))).squeeze()

#         num_classes = self.n_classes
#         config = self.config

#         # Load image and mask
#         mask_dict = au.annList2maskList(annList)


#         mask, class_ids = mask_dict["maskList"], mask_dict["categoryList"]


#         shape = image.shape
#         image, window, scale, padding = resize_image(
#             image,
#             min_dim=config.IMAGE_MIN_DIM,
#             max_dim=config.IMAGE_MAX_DIM,
#             padding=config.IMAGE_PADDING)

#         mask = resize_mask(mask, scale, padding)

#         # Random horizontal flips.
#         if augment:
#             if random.randint(0, 1):
#                 image = np.fliplr(image)
#                 mask = np.fliplr(mask)

#         # Bounding boxes. Note that some boxes might be all zeros
#         # if the corresponding mask got cropped out.
#         # bbox: [num_instances, (y1, x1, y2, x2)]
#         bbox = extract_bboxes(mask)

#         # Active classes
#         # Different datasets have different classes, so track the
#         # classes supported in the dataset of this image.
#         active_class_ids = np.zeros([num_classes], dtype=np.int32)
#         active_class_ids[np.unique(class_ids)] = 1

#         # Resize masks to smaller size to reduce memory usage
#         if config.USE_MINI_MASK:
#             mask = minimize_mask(bbox, mask, config.MINI_MASK_SHAPE)

#         # Image meta data
#         image_meta = compose_image_meta(image_id, shape, window, active_class_ids)

#         return image, image_meta, class_ids, bbox, mask

#     @torch.no_grad()
#     def predict(self, batch, **options):
#         """Runs the detection pipeline.
#         images: List of images, potentially of different sizes.
#         Returns a list of dicts, one dict per image. The dict contains:
#         rois: [N, (y1, x1, y2, x2)] detection bounding boxes
#         class_ids: [N] int class IDs
#         scores: [N] float probability scores for the class IDs
#         masks: [H, W, N] instance binary masks
#         """

#         # Mold inputs to format expected by the neural network
#         # molded_images, image_metas, windows = self.mold_inputs(images)

#         # Convert images to torch tensor
#         # molded_images = torch.from_numpy(molded_images.transpose(0, 3, 1, 2)).float()

#         # To GPU
#         self.eval()

#         images = ms.f2l(ms.t2n((ms.denormalize(batch["images"])))).squeeze()

#         pred_dict = self.forward_test(images, image_id=batch["name"][0], 
#           maskVoid=None)
#         print("no best dice")
#         # annList = au.annList2BestDice(pred_dict["annList"], batch)["annList"]
#         # pred_dict["annList"] = annList

#         if options["predict_method"] == "blobs":

#           return {"blobs":au.annList2mask(pred_dict["annList"], color=1)["mask"]}
#         return pred_dict



#     def forward_train(self, batch):
#       molded_images = batch["images"]
#       gt_class_ids = batch["gt_class_ids"]
#       gt_boxes = batch["gt_boxes"]
#       gt_masks = batch["gt_masks"]


#       # Set batchnorm always in eval mode during training
#       def set_bn_eval(m):
#           classname = m.__class__.__name__
#           if classname.find('BatchNorm') != -1:
#               m.eval()

#       self.apply(set_bn_eval)

#       rpn_dict = self.extract_rpn(molded_images, proposal_count=self.config.POST_NMS_ROIS_TRAINING)


#       # Normalize coordinates
#       h, w = self.config.IMAGE_SHAPE[:2]
#       scale = Variable(torch.from_numpy(np.array([h, w, h, w])).float(), requires_grad=False)
#       if self.config.GPU_COUNT:
#           scale = scale.cuda()
#       gt_boxes = gt_boxes / scale

#       # Generate detection targets
#       # Subsamples proposals and generates target outputs for training
#       # Note that proposal class IDs, gt_boxes, and gt_masks are zero
#       # padded. Equally, returned rois and targets are zero padded.

#       rois, target_class_ids, target_deltas, target_mask = \
#           detection_target_layer(rpn_dict["rpn_rois"].squeeze(), gt_class_ids, gt_boxes, gt_masks, self.config)

#       if not np.prod(rois.size()):
#           mrcnn_class_logits = Variable(torch.FloatTensor())
#           mrcnn_class = Variable(torch.IntTensor())
#           mrcnn_bbox = Variable(torch.FloatTensor())
#           mrcnn_mask = Variable(torch.FloatTensor())

#           mrcnn_class_logits = mrcnn_class_logits.cuda()
#           mrcnn_class = mrcnn_class.cuda()
#           mrcnn_bbox = mrcnn_bbox.cuda()
#           mrcnn_mask = mrcnn_mask.cuda()
#       else:
#           # Network Heads
#           # Proposal classifier and BBox regressor heads
#           mrcnn_class_logits, mrcnn_class, mrcnn_bbox = self.classifier(rpn_dict["mrcnn_feature_maps"], rois)

#           # Create masks for detections
#           mrcnn_mask = self.mask(rpn_dict["mrcnn_feature_maps"], rois)

#       return [rpn_dict["rpn_class_logits"], rpn_dict["rpn_bbox"], 
#               target_class_ids, mrcnn_class_logits, 
#               target_deltas, mrcnn_bbox, target_mask, mrcnn_mask]


#     def forward_test(self, images, image_id, maskVoid=None):
#       molded_images, image_metas, windows = self.mold_inputs([images])
#       molded_images = torch.from_numpy(molded_images.transpose(0, 3, 1, 2)).float().cuda()


      
#       # Set batchnorm always in eval mode during training
#       def set_bn_eval(m):
#           classname = m.__class__.__name__
#           if classname.find('BatchNorm') != -1:
#               m.eval()

#       self.apply(set_bn_eval)

#       rpn_dict = self.extract_rpn(molded_images, proposal_count=self.config.POST_NMS_ROIS_INFERENCE)

#       mrcnn_feature_maps = rpn_dict["mrcnn_feature_maps"]
#       rpn_rois = rpn_dict["rpn_rois"].squeeze()

#       mrcnn_class_logits, mrcnn_class, mrcnn_bbox = self.classifier(mrcnn_feature_maps, 
#                                                                     rpn_rois.squeeze())

#       # Detections
#       # output is [batch, num_detections, (y1, x1, y2, x2, class_id, score)] in image coordinates
      
#       detections = detection_layer(self.config, rpn_rois, mrcnn_class, mrcnn_bbox, image_metas)
#       if detections is None:
#         return {"annList": []}
#       # Convert boxes to normalized coordinates
#       # TODO: let DetectionLayer return normalized coordinates to avoid
#       #       unnecessary conversions

#       h, w = self.config.IMAGE_SHAPE[:2]
#       scale = torch.from_numpy(np.array([h, w, h, w])).float()
#       scale = scale.cuda()

#       detection_boxes = detections[:, :4] / scale

#       # Add back batch dimension
#       detection_boxes = detection_boxes.unsqueeze(0)

#       # Create masks for detections
#       mrcnn_mask = self.mask(mrcnn_feature_maps, detection_boxes)

#       # Add back batch dimension
#       detections = detections.unsqueeze(0)
#       mrcnn_mask = mrcnn_mask.unsqueeze(0)

#       # Convert to numpy
#       detections = detections.data.cpu().numpy()
#       mrcnn_mask = mrcnn_mask.permute(0, 1, 3, 4, 2).data.cpu().numpy()

#       # Process detections

#       final_rois, final_class_ids, final_scores, final_masks =\
#           self.unmold_detections(detections[0], mrcnn_mask[0],
#                                  images.shape, windows[0])
#       results_dict = {
#           "rois": final_rois,
#           "class_ids": final_class_ids,
#           "scores": final_scores,
#           "masks": final_masks,
#       }

#       maskList = results_dict["masks"]
#       maskList = maskList.transpose(2, 0, 1)
#       if maskVoid is not None:
#         maskList = (ms.t2n(maskVoid) * maskList)

#       annList = au.maskList2annList(maskList, final_class_ids, image_id=image_id, scoreList=final_scores)

#       return {"annList": annList}


#     def unmold_detections(self, detections, mrcnn_mask, image_shape, window):
#         """Reformats the detections of one image from the format of the neural
#         network output to a format suitable for use in the rest of the
#         application.
#         detections: [N, (y1, x1, y2, x2, class_id, score)]
#         mrcnn_mask: [N, height, width, num_classes]
#         image_shape: [height, width, depth] Original size of the image before resizing
#         window: [y1, x1, y2, x2] Box in the image where the real image is
#                 excluding the padding.
#         Returns:
#         boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
#         class_ids: [N] Integer class IDs for each bounding box
#         scores: [N] Float probability scores of the class_id
#         masks: [height, width, num_instances] Instance masks
#         """
#         # How many detections do we have?
#         # Detections array is padded with zeros. Find the first class_id == 0.
#         zero_ix = np.where(detections[:, 4] == 0)[0]
#         N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

#         # Extract boxes, class_ids, scores, and class-specific masks
#         boxes = detections[:N, :4]
#         class_ids = detections[:N, 4].astype(np.int32)
#         scores = detections[:N, 5]
#         masks = mrcnn_mask[np.arange(N), :, :, class_ids]

#         # Compute scale and shift to translate coordinates to image domain.
#         h_scale = image_shape[0] / (window[2] - window[0])
#         w_scale = image_shape[1] / (window[3] - window[1])
#         scale = min(h_scale, w_scale)
#         shift = window[:2]  # y, x
#         scales = np.array([scale, scale, scale, scale])
#         shifts = np.array([shift[0], shift[1], shift[0], shift[1]])

#         # Translate bounding boxes to image domain
#         boxes = np.multiply(boxes - shifts, scales).astype(np.int32)

#         # Filter out detections with zero area. Often only happens in early
#         # stages of training when the network weights are still a bit random.
#         exclude_ix = np.where(
#             (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
#         if exclude_ix.shape[0] > 0:
#             boxes = np.delete(boxes, exclude_ix, axis=0)
#             class_ids = np.delete(class_ids, exclude_ix, axis=0)
#             scores = np.delete(scores, exclude_ix, axis=0)
#             masks = np.delete(masks, exclude_ix, axis=0)
#             N = class_ids.shape[0]

#         # Resize masks to original image size and set boundary threshold.
#         full_masks = []
#         for i in range(N):
#             # Convert neural network mask to full size mask
#             full_mask = unmold_mask(masks[i], boxes[i], image_shape)
#             full_masks.append(full_mask)
#         full_masks = np.stack(full_masks, axis=-1)\
#             if full_masks else np.empty((0,) + masks.shape[1:3])

#         return boxes, class_ids, scores, full_masks

# ############################################################
# #  Pytorch Utility Functions
# ############################################################

# def unique1d(tensor):
#     if tensor.size()[0] == 0 or tensor.size()[0] == 1:
#         return tensor
#     tensor = tensor.sort()[0]
#     unique_bool = tensor[1:] != tensor [:-1]
#     first_element = Variable(torch.ByteTensor([True]), requires_grad=False)
#     if tensor.is_cuda:
#         first_element = first_element.cuda()
#     unique_bool = torch.cat((first_element, unique_bool),dim=0)
#     return tensor[unique_bool.data]

# def intersect1d(tensor1, tensor2):
#     aux = torch.cat((tensor1, tensor2),dim=0)
#     aux = aux.sort()[0]
#     return aux[:-1][(aux[1:] == aux[:-1]).data]

# def log2(x):
#     """Implementatin of Log2. Pytorch doesn't have a native implemenation."""
#     ln2 = Variable(torch.log(torch.FloatTensor([2.0])), requires_grad=False)
#     if x.is_cuda:
#         ln2 = ln2.cuda()
#     return torch.log(x) / ln2

# class SamePad2d(nn.Module):
#     """Mimics tensorflow's 'SAME' padding.
#     """

#     def __init__(self, kernel_size, stride):
#         super(SamePad2d, self).__init__()
#         self.kernel_size = torch.nn.modules.utils._pair(kernel_size)
#         self.stride = torch.nn.modules.utils._pair(stride)

#     def forward(self, input):
#         in_width = input.size()[2]
#         in_height = input.size()[3]
#         out_width = math.ceil(float(in_width) / float(self.stride[0]))
#         out_height = math.ceil(float(in_height) / float(self.stride[1]))
#         pad_along_width = ((out_width - 1) * self.stride[0] +
#                            self.kernel_size[0] - in_width)
#         pad_along_height = ((out_height - 1) * self.stride[1] +
#                             self.kernel_size[1] - in_height)
#         pad_left = math.floor(pad_along_width / 2)
#         pad_top = math.floor(pad_along_height / 2)
#         pad_right = pad_along_width - pad_left
#         pad_bottom = pad_along_height - pad_top
#         return F.pad(input, (pad_left, pad_right, pad_top, pad_bottom), 'constant', 0)

#     def __repr__(self):
#         return self.__class__.__name__


# ############################################################
# #  FPN Graph
# ############################################################

# class FPN(nn.Module):
#     def __init__(self, C1, C2, C3, C4, C5, out_channels):
#         super(FPN, self).__init__()
#         self.out_channels = out_channels
#         self.C1 = C1
#         self.C2 = C2
#         self.C3 = C3
#         self.C4 = C4
#         self.C5 = C5
#         self.P6 = nn.MaxPool2d(kernel_size=1, stride=2)
#         self.P5_conv1 = nn.Conv2d(2048, self.out_channels, kernel_size=1, stride=1)
#         self.P5_conv2 = nn.Sequential(
#             SamePad2d(kernel_size=3, stride=1),
#             nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1),
#         )
#         self.P4_conv1 =  nn.Conv2d(1024, self.out_channels, kernel_size=1, stride=1)
#         self.P4_conv2 = nn.Sequential(
#             SamePad2d(kernel_size=3, stride=1),
#             nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1),
#         )
#         self.P3_conv1 = nn.Conv2d(512, self.out_channels, kernel_size=1, stride=1)
#         self.P3_conv2 = nn.Sequential(
#             SamePad2d(kernel_size=3, stride=1),
#             nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1),
#         )
#         self.P2_conv1 = nn.Conv2d(256, self.out_channels, kernel_size=1, stride=1)
#         self.P2_conv2 = nn.Sequential(
#             SamePad2d(kernel_size=3, stride=1),
#             nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1),
#         )

#     def forward(self, x):
#         x = self.C1(x)
#         x = self.C2(x)
#         c2_out = x
#         x = self.C3(x)
#         c3_out = x
#         x = self.C4(x)
#         c4_out = x
#         x = self.C5(x)
#         p5_out = self.P5_conv1(x)

#         c4_conv = self.P4_conv1(c4_out)
#         # p4_out = c4_conv + F.interpolate(p5_out, size=c4_conv.size()[-2:], mode="nearest")
#         p4_out = c4_conv + F.interpolate(p5_out, scale_factor=2, mode="nearest")

#         c3_conv = self.P3_conv1(c3_out)
#         # p3_out = c3_conv +  F.interpolate(p4_out, size=c3_conv.size()[-2:], mode="nearest")
#         p3_out = c3_conv +  F.interpolate(p4_out, scale_factor=2, mode="nearest")

#         c2_conv = self.P2_conv1(c2_out)
#         # p2_out = c2_conv + F.interpolate(p3_out, size=c2_conv.size()[-2:], mode="nearest")
#         p2_out = c2_conv + F.interpolate(p3_out, scale_factor=2, mode="nearest")

#         p5_out = self.P5_conv2(p5_out)
#         p4_out = self.P4_conv2(p4_out)
#         p3_out = self.P3_conv2(p3_out)
#         p2_out = self.P2_conv2(p2_out)

#         # P6 is used for the 5th anchor scale in RPN. Generated by
#         # subsampling from P5 with stride of 2.
#         p6_out = self.P6(p5_out)

#         return [p2_out, p3_out, p4_out, p5_out, p6_out]




# ############################################################
# #  Proposal Layer
# ############################################################

# def apply_box_deltas(boxes, deltas):
#     """Applies the given deltas to the given boxes.
#     boxes: [N, 4] where each row is y1, x1, y2, x2
#     deltas: [N, 4] where each row is [dy, dx, log(dh), log(dw)]
#     """
#     # Convert to y, x, h, w
#     height = boxes[:, 2] - boxes[:, 0]
#     width = boxes[:, 3] - boxes[:, 1]
#     center_y = boxes[:, 0] + 0.5 * height
#     center_x = boxes[:, 1] + 0.5 * width
#     # Apply deltas
#     center_y += deltas[:, 0] * height
#     center_x += deltas[:, 1] * width
#     height *= torch.exp(deltas[:, 2])
#     width *= torch.exp(deltas[:, 3])
#     # Convert back to y1, x1, y2, x2
#     y1 = center_y - 0.5 * height
#     x1 = center_x - 0.5 * width
#     y2 = y1 + height
#     x2 = x1 + width
#     result = torch.stack([y1, x1, y2, x2], dim=1)
#     return result

# def clip_boxes(boxes, window):
#     """
#     boxes: [N, 4] each col is y1, x1, y2, x2
#     window: [4] in the form y1, x1, y2, x2
#     """
#     boxes = torch.stack( \
#         [boxes[:, 0].clamp(float(window[0]), float(window[2])),
#          boxes[:, 1].clamp(float(window[1]), float(window[3])),
#          boxes[:, 2].clamp(float(window[0]), float(window[2])),
#          boxes[:, 3].clamp(float(window[1]), float(window[3]))], 1)
#     return boxes

# def proposal_layer(inputs, proposal_count, nms_threshold, anchors, config=None):
#     """Receives anchor scores and selects a subset to pass as proposals
#     to the second stage. Filtering is done based on anchor scores and
#     non-max suppression to remove overlaps. It also applies bounding
#     box refinment detals to anchors.
#     Inputs:
#         rpn_probs: [batch, anchors, (bg prob, fg prob)]
#         rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
#     Returns:
#         Proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)]
#     """

#     # Currently only supports batchsize 1
#     inputs[0] = inputs[0].squeeze(0)
#     inputs[1] = inputs[1].squeeze(0)

#     # Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]
#     scores = inputs[0][:, 1]

#     # Box deltas [batch, num_rois, 4]
#     deltas = inputs[1]
#     std_dev = Variable(torch.from_numpy(np.reshape(config.RPN_BBOX_STD_DEV, [1, 4])).float(), requires_grad=False)
#     if config.GPU_COUNT:
#         std_dev = std_dev.cuda()
#     deltas = deltas * std_dev

#     # Improve performance by trimming to top anchors by score
#     # and doing the rest on the smaller subset.
#     pre_nms_limit = min(6000, anchors.size()[0])
#     scores, order = scores.sort(descending=True)
#     order = order[:pre_nms_limit]
#     scores = scores[:pre_nms_limit]
#     deltas = deltas[order.data, :] # TODO: Support batch size > 1 ff.
#     anchors = anchors[order.data, :]

#     # Apply deltas to anchors to get refined anchors.
#     # [batch, N, (y1, x1, y2, x2)]
#     boxes = apply_box_deltas(anchors, deltas)

#     # Clip to image boundaries. [batch, N, (y1, x1, y2, x2)]
#     height, width = config.IMAGE_SHAPE[:2]
#     window = np.array([0, 0, height, width]).astype(np.float32)
#     boxes = clip_boxes(boxes, window)

#     # Filter out small boxes
#     # According to Xinlei Chen's paper, this reduces detection accuracy
#     # for small objects, so we're skipping it.

#     # Non-max suppression

#     keep = nms(torch.cat((boxes, scores.unsqueeze(1)), 1).data, nms_threshold)
#     keep = keep[:proposal_count].long()
#     boxes = boxes[keep, :]

#     # Normalize dimensions to range of 0 to 1.
#     norm = Variable(torch.from_numpy(np.array([height, width, height, width])).float(), requires_grad=False)
#     norm = norm.cuda()
#     normalized_boxes = boxes / norm

#     # Add back batch dimension
#     normalized_boxes = normalized_boxes.unsqueeze(0)

#     return normalized_boxes


# ############################################################
# #  ROIAlign Layer
# ############################################################

# def pyramid_roi_align(inputs, pool_size, image_shape):
#     """Implements ROI Pooling on multiple levels of the feature pyramid.
#     Params:
#     - pool_size: [height, width] of the output pooled regions. Usually [7, 7]
#     - image_shape: [height, width, channels]. Shape of input image in pixels
#     Inputs:
#     - boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
#              coordinates.
#     - Feature maps: List of feature maps from different levels of the pyramid.
#                     Each is [batch, channels, height, width]
#     Output:
#     Pooled regions in the shape: [num_boxes, height, width, channels].
#     The width and height are those specific in the pool_shape in the layer
#     constructor.
#     """

#     # Currently only supports batchsize 1
#     for i in range(len(inputs)):
#         inputs[i] = inputs[i].squeeze(0)

#     # Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
#     boxes = inputs[0]

#     # Feature Maps. List of feature maps from different level of the
#     # feature pyramid. Each is [batch, height, width, channels]
#     feature_maps = inputs[1:]

#     # Assign each ROI to a level in the pyramid based on the ROI area.
#     y1, x1, y2, x2 = boxes.chunk(4, dim=1)
#     h = y2 - y1
#     w = x2 - x1

#     # Equation 1 in the Feature Pyramid Networks paper. Account for
#     # the fact that our coordinates are normalized here.
#     # e.g. a 224x224 ROI (in pixels) maps to P4
#     image_area = Variable(torch.FloatTensor([float(image_shape[0]*image_shape[1])]), requires_grad=False)
#     if boxes.is_cuda:
#         image_area = image_area.cuda()
#     roi_level = 4 + log2(torch.sqrt(h*w)/(224.0/torch.sqrt(image_area)))
#     roi_level = roi_level.round().int()
#     roi_level = roi_level.clamp(2,5)


#     # Loop through levels and apply ROI pooling to each. P2 to P5.
#     pooled = []
#     box_to_level = []
#     for i, level in enumerate(range(2, 6)):
#         ix  = roi_level==level
#         if not ix.any():
#             continue
#         ix = torch.nonzero(ix)[:,0]
#         level_boxes = boxes[ix.data, :]

#         # Keep track of which box is mapped to which level
#         box_to_level.append(ix.data)

#         # Stop gradient propogation to ROI proposals
#         level_boxes = level_boxes.detach()

#         # Crop and Resize
#         # From Mask R-CNN paper: "We sample four regular locations, so
#         # that we can evaluate either max or average pooling. In fact,
#         # interpolating only a single value at each bin center (without
#         # pooling) is nearly as effective."
#         #
#         # Here we use the simplified approach of a single value per bin,
#         # which is how it's done in tf.crop_and_resize()
#         # Result: [batch * num_boxes, pool_height, pool_width, channels]
#         ind = Variable(torch.zeros(level_boxes.size()[0]),requires_grad=False).int()
#         if level_boxes.is_cuda:
#             ind = ind.cuda()
#         feature_maps[i] = feature_maps[i].unsqueeze(0)  #CropAndResizeFunction needs batch dimension
#         pooled_features = CropAndResizeFunction(pool_size, pool_size, 0)(feature_maps[i], level_boxes, ind)
#         pooled.append(pooled_features)

#     # Pack pooled features into one tensor
#     pooled = torch.cat(pooled, dim=0)

#     # Pack box_to_level mapping into one array and add another
#     # column representing the order of pooled boxes
#     box_to_level = torch.cat(box_to_level, dim=0)

#     # Rearrange pooled features to match the order of the original boxes
#     _, box_to_level = torch.sort(box_to_level)
#     pooled = pooled[box_to_level, :, :]

#     return pooled


# ############################################################
# #  Detection Target Layer
# ############################################################
# def bbox_overlaps(boxes1, boxes2):
#     """Computes IoU overlaps between two sets of boxes.
#     boxes1, boxes2: [N, (y1, x1, y2, x2)].
#     """
#     # 1. Tile boxes2 and repeate boxes1. This allows us to compare
#     # every boxes1 against every boxes2 without loops.
#     # TF doesn't have an equivalent to np.repeate() so simulate it
#     # using tf.tile() and tf.reshape.
#     boxes1_repeat = boxes2.size()[0]
#     boxes2_repeat = boxes1.size()[0]
#     boxes1 = boxes1.repeat(1, boxes1_repeat).view(-1,4)
#     boxes2 = boxes2.repeat(boxes2_repeat,1)

#     # 2. Compute intersections
#     b1_y1, b1_x1, b1_y2, b1_x2 = boxes1.chunk(4, dim=1)
#     b2_y1, b2_x1, b2_y2, b2_x2 = boxes2.chunk(4, dim=1)
#     y1 = torch.max(b1_y1, b2_y1)[:, 0]
#     x1 = torch.max(b1_x1, b2_x1)[:, 0]
#     y2 = torch.min(b1_y2, b2_y2)[:, 0]
#     x2 = torch.min(b1_x2, b2_x2)[:, 0]
#     zeros = Variable(torch.zeros(y1.size()[0]), requires_grad=False)
#     if y1.is_cuda:
#         zeros = zeros.cuda()
#     intersection = torch.max(x2 - x1, zeros) * torch.max(y2 - y1, zeros)

#     # 3. Compute unions
#     b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
#     b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
#     union = b1_area[:,0] + b2_area[:,0] - intersection

#     # 4. Compute IoU and reshape to [boxes1, boxes2]
#     iou = intersection / union
#     overlaps = iou.view(boxes2_repeat, boxes1_repeat)

#     return overlaps

# def detection_target_layer(proposals, gt_class_ids, gt_boxes, gt_masks, config):
#     """Subsamples proposals and generates target box refinment, class_ids,
#     and masks for each.
#     Inputs:
#     proposals: [batch, N, (y1, x1, y2, x2)] in normalized coordinates. Might
#                be zero padded if there are not enough proposals.
#     gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs.
#     gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized
#               coordinates.
#     gt_masks: [batch, height, width, MAX_GT_INSTANCES] of boolean type
#     Returns: Target ROIs and corresponding class IDs, bounding box shifts,
#     and masks.
#     rois: [batch, TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized
#           coordinates
#     target_class_ids: [batch, TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
#     target_deltas: [batch, TRAIN_ROIS_PER_IMAGE, NUM_CLASSES,
#                     (dy, dx, log(dh), log(dw), class_id)]
#                    Class-specific bbox refinments.
#     target_mask: [batch, TRAIN_ROIS_PER_IMAGE, height, width)
#                  Masks cropped to bbox boundaries and resized to neural
#                  network output size.
#     """

#     # Currently only supports batchsize 1
#     proposals = proposals.squeeze(0)
#     gt_class_ids = gt_class_ids.squeeze(0)
#     gt_boxes = gt_boxes.squeeze(0)
#     gt_masks = gt_masks.squeeze(0)

#     # Handle COCO crowds
#     # A crowd box in COCO is a bounding box around several instances. Exclude
#     # them from training. A crowd box is given a negative class ID.
#     if torch.nonzero(gt_class_ids < 0).sum().item():
#         crowd_ix = torch.nonzero(gt_class_ids < 0)[:, 0]
#         non_crowd_ix = torch.nonzero(gt_class_ids > 0)[:, 0]
#         crowd_boxes = gt_boxes[crowd_ix.data, :]
#         crowd_masks = gt_masks[crowd_ix.data, :, :]
#         gt_class_ids = gt_class_ids[non_crowd_ix.data]
#         gt_boxes = gt_boxes[non_crowd_ix.data, :]
#         gt_masks = gt_masks[non_crowd_ix.data, :]

#         # Compute overlaps with crowd boxes [anchors, crowds]
#         crowd_overlaps = bbox_overlaps(proposals, crowd_boxes)
#         crowd_iou_max = torch.max(crowd_overlaps, dim=1)[0]
#         no_crowd_bool = crowd_iou_max < 0.001
#     else:
#         no_crowd_bool =  Variable(torch.ByteTensor(proposals.size()[0]*[True]), requires_grad=False)
#         no_crowd_bool = no_crowd_bool.cuda()

#     # Compute overlaps matrix [proposals, gt_boxes]
#     overlaps = bbox_overlaps(proposals, gt_boxes)

#     # Determine postive and negative ROIs
#     roi_iou_max = torch.max(overlaps, dim=1)[0]

#     # 1. Positive ROIs are those with >= 0.5 IoU with a GT box
#     positive_roi_bool = roi_iou_max >= 0.5

#     # Subsample ROIs. Aim for 33% positive
#     # Positive ROIs
#     if torch.nonzero(positive_roi_bool).sum().item():
#         positive_indices = torch.nonzero(positive_roi_bool)[:, 0]

#         positive_count = int(config.TRAIN_ROIS_PER_IMAGE *
#                              config.ROI_POSITIVE_RATIO)
#         rand_idx = torch.randperm(positive_indices.size()[0])
#         rand_idx = rand_idx[:positive_count]
#         if config.GPU_COUNT:
#             rand_idx = rand_idx.cuda()
#         positive_indices = positive_indices[rand_idx]
#         positive_count = positive_indices.size()[0]
#         positive_rois = proposals[positive_indices.data,:]

#         # Assign positive ROIs to GT boxes.
#         positive_overlaps = overlaps[positive_indices.data,:]
#         roi_gt_box_assignment = torch.max(positive_overlaps, dim=1)[1]
#         roi_gt_boxes = gt_boxes[roi_gt_box_assignment.data,:]
#         roi_gt_class_ids = gt_class_ids[roi_gt_box_assignment.data]

#         # Compute bbox refinement for positive ROIs
#         deltas = Variable(box_refinement(positive_rois.data, roi_gt_boxes.data), requires_grad=False)
#         std_dev = Variable(torch.from_numpy(config.BBOX_STD_DEV).float(), requires_grad=False)
#         if config.GPU_COUNT:
#             std_dev = std_dev.cuda()
#         deltas /= std_dev

#         # Assign positive ROIs to GT masks
#         roi_masks = gt_masks[roi_gt_box_assignment.data,:,:]

#         # Compute mask targets
#         boxes = positive_rois
#         if config.USE_MINI_MASK:
#             # Transform ROI corrdinates from normalized image space
#             # to normalized mini-mask space.
#             y1, x1, y2, x2 = positive_rois.chunk(4, dim=1)
#             gt_y1, gt_x1, gt_y2, gt_x2 = roi_gt_boxes.chunk(4, dim=1)
#             gt_h = gt_y2 - gt_y1
#             gt_w = gt_x2 - gt_x1
#             y1 = (y1 - gt_y1) / gt_h
#             x1 = (x1 - gt_x1) / gt_w
#             y2 = (y2 - gt_y1) / gt_h
#             x2 = (x2 - gt_x1) / gt_w
#             boxes = torch.cat([y1, x1, y2, x2], dim=1)
#         box_ids = Variable(torch.arange(roi_masks.size()[0]), requires_grad=False).int()
#         if config.GPU_COUNT:
#             box_ids = box_ids.cuda()
#         masks = Variable(CropAndResizeFunction(config.MASK_SHAPE[0], config.MASK_SHAPE[1], 0)(roi_masks.unsqueeze(1), boxes, box_ids).data, requires_grad=False)
#         masks = masks.squeeze(1)

#         # Threshold mask pixels at 0.5 to have GT masks be 0 or 1 to use with
#         # binary cross entropy loss.
#         masks = torch.round(masks)
#     else:
#         positive_count = 0

#     # 2. Negative ROIs are those with < 0.5 with every GT box. Skip crowds.
#     negative_roi_bool = roi_iou_max < 0.5
#     negative_roi_bool = negative_roi_bool & no_crowd_bool
#     # Negative ROIs. Add enough to maintain positive:negative ratio.
#     if torch.nonzero(negative_roi_bool).size() and positive_count>0:
#         negative_indices = torch.nonzero(negative_roi_bool)[:, 0]
#         r = 1.0 / config.ROI_POSITIVE_RATIO
#         negative_count = int(r * positive_count - positive_count)
#         rand_idx = torch.randperm(negative_indices.size()[0])
#         rand_idx = rand_idx[:negative_count]
#         if config.GPU_COUNT:
#             rand_idx = rand_idx.cuda()
#         negative_indices = negative_indices[rand_idx]
#         negative_count = negative_indices.size()[0]
#         negative_rois = proposals[negative_indices.data, :]
#     else:
#         negative_count = 0

#     # Append negative ROIs and pad bbox deltas and masks that
#     # are not used for negative ROIs with zeros.
#     if positive_count > 0 and negative_count > 0:
#         rois = torch.cat((positive_rois, negative_rois), dim=0)
#         zeros = Variable(torch.zeros(negative_count), requires_grad=False).int()
#         if config.GPU_COUNT:
#             zeros = zeros.cuda()
#         roi_gt_class_ids = torch.cat([roi_gt_class_ids, zeros], dim=0)
#         zeros = Variable(torch.zeros(negative_count,4), requires_grad=False)
#         if config.GPU_COUNT:
#             zeros = zeros.cuda()
#         deltas = torch.cat([deltas, zeros], dim=0)
#         zeros = Variable(torch.zeros(negative_count,config.MASK_SHAPE[0],config.MASK_SHAPE[1]), requires_grad=False)
#         if config.GPU_COUNT:
#             zeros = zeros.cuda()
#         masks = torch.cat([masks, zeros], dim=0)
#     elif positive_count > 0:
#         rois = positive_rois
#     elif negative_count > 0:
#         rois = negative_rois
#         zeros = Variable(torch.zeros(negative_count), requires_grad=False)
#         if config.GPU_COUNT:
#             zeros = zeros.cuda()
#         roi_gt_class_ids = zeros
#         zeros = Variable(torch.zeros(negative_count,4), requires_grad=False).int()
#         if config.GPU_COUNT:
#             zeros = zeros.cuda()
#         deltas = zeros
#         zeros = Variable(torch.zeros(negative_count,config.MASK_SHAPE[0],config.MASK_SHAPE[1]), requires_grad=False)
#         if config.GPU_COUNT:
#             zeros = zeros.cuda()
#         masks = zeros
#     else:
#         rois = Variable(torch.FloatTensor(), requires_grad=False)
#         roi_gt_class_ids = Variable(torch.IntTensor(), requires_grad=False)
#         deltas = Variable(torch.FloatTensor(), requires_grad=False)
#         masks = Variable(torch.FloatTensor(), requires_grad=False)
#         if config.GPU_COUNT:
#             rois = rois.cuda()
#             roi_gt_class_ids = roi_gt_class_ids.cuda()
#             deltas = deltas.cuda()
#             masks = masks.cuda()

#     return rois, roi_gt_class_ids, deltas, masks


# ############################################################
# #  Detection Layer
# ############################################################

# def clip_to_window(window, boxes):
#     """
#         window: (y1, x1, y2, x2). The window in the image we want to clip to.
#         boxes: [N, (y1, x1, y2, x2)]
#     """
#     boxes[:, 0] = boxes[:, 0].clamp(float(window[0]), float(window[2]))
#     boxes[:, 1] = boxes[:, 1].clamp(float(window[1]), float(window[3]))
#     boxes[:, 2] = boxes[:, 2].clamp(float(window[0]), float(window[2]))
#     boxes[:, 3] = boxes[:, 3].clamp(float(window[1]), float(window[3]))

#     return boxes

# def refine_detections(rois, probs, deltas, window, config):
#     """Refine classified proposals and filter overlaps and return final
#     detections.
#     Inputs:
#         rois: [N, (y1, x1, y2, x2)] in normalized coordinates
#         probs: [N, num_classes]. Class probabilities.
#         deltas: [N, num_classes, (dy, dx, log(dh), log(dw))]. Class-specific
#                 bounding box deltas.
#         window: (y1, x1, y2, x2) in image coordinates. The part of the image
#             that contains the image excluding the padding.
#     Returns detections shaped: [N, (y1, x1, y2, x2, class_id, score)]
#     """

#     # Class IDs per ROI
#     _, class_ids = torch.max(probs, dim=1)

#     # Class probability of the top class of each ROI
#     # Class-specific bounding box deltas
#     idx = torch.arange(class_ids.size()[0]).long()
#     if config.GPU_COUNT:
#         idx = idx.cuda()
#     class_scores = probs[idx, class_ids.data]
#     deltas_specific = deltas[idx, class_ids.data]

#     # Apply bounding box deltas
#     # Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
#     std_dev = torch.from_numpy(np.reshape(config.RPN_BBOX_STD_DEV, [1, 4])).float()
#     std_dev = std_dev.cuda()

#     refined_rois = apply_box_deltas(rois, deltas_specific * std_dev)

#     # Convert coordiates to image domain
#     height, width = config.IMAGE_SHAPE[:2]
#     scale = torch.from_numpy(np.array([height, width, height, width])).float()
#     scale = scale.cuda()
#     refined_rois *= scale

#     # Clip boxes to image window
#     refined_rois = clip_to_window(window, refined_rois)

#     # Round and cast to int since we're deadling with pixels now
#     refined_rois = torch.round(refined_rois)

#     # TODO: Filter out boxes with zero area

#     # Filter out background boxes
#     keep_bool = class_ids>0

#     # Filter out low confidence boxes
#     if config.DETECTION_MIN_CONFIDENCE:
#         keep_bool = keep_bool & (class_scores >= config.DETECTION_MIN_CONFIDENCE)

#     if keep_bool.sum().item():
#       keep = torch.nonzero(keep_bool)[:,0]

#       # Apply per-class NMS
#       pre_nms_class_ids = class_ids[keep.data]
#       pre_nms_scores = class_scores[keep.data]
#       pre_nms_rois = refined_rois[keep.data]

#       for i, class_id in enumerate(unique1d(pre_nms_class_ids)):
#           # Pick detections of this class
#           ixs = torch.nonzero(pre_nms_class_ids == class_id)[:,0]

#           # Sort
#           ix_rois = pre_nms_rois[ixs.data]
#           ix_scores = pre_nms_scores[ixs]

#           ix_scores, order = ix_scores.sort(descending=True)
#           ix_rois = ix_rois[order.data,:]

#           class_keep = nms(torch.cat((ix_rois, ix_scores.unsqueeze(1)), dim=1).data, config.DETECTION_NMS_THRESHOLD).long()

#           # Map indicies
#           class_keep = keep[ixs[order[class_keep].data].data].squeeze()
#           if class_keep.dim() == 0:
#             class_keep = class_keep[None]

#           if i==0:
#             nms_keep = class_keep
#           else:

#             nms_keep = unique1d(torch.cat((nms_keep, class_keep)))

#       keep = intersect1d(keep, nms_keep)

#       # Keep top detections
#       roi_count = config.DETECTION_MAX_INSTANCES
#       top_ids = class_scores[keep.data].sort(descending=True)[1][:roi_count]
#       keep = keep[top_ids.data]

#       # Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
#       # Coordinates are in image domain.

#       result = torch.cat((refined_rois[keep.data],
#                           class_ids[keep.data].unsqueeze(1).float(),
#                           class_scores[keep.data].unsqueeze(1)), dim=1)

#       return result
#     else:
#       return None


# def detection_layer(config, rois, mrcnn_class, mrcnn_bbox, image_meta):
#     """Takes classified proposal boxes and their bounding box deltas and
#     returns the final detection boxes.
#     Returns:
#     [batch, num_detections, (y1, x1, y2, x2, class_score)] in pixels
#     """

#     # Currently only supports batchsize 1
#     rois = rois.squeeze(0)

#     _, _, window, _ = parse_image_meta(image_meta)
#     window = window[0]
#     detections = refine_detections(rois, mrcnn_class, mrcnn_bbox, window, config)

#     return detections

# # Two functions (for Numpy and TF) to parse image_meta tensors.
# def parse_image_meta(meta):
#     """Parses an image info Numpy array to its components.
#     See compose_image_meta() for more details.
#     """
#     image_id = meta[:, 0]
#     image_shape = meta[:, 1:4]
#     window = meta[:, 4:8]   # (y1, x1, y2, x2) window of image in in pixels
#     active_class_ids = meta[:, 8:]
#     return image_id, image_shape, window, active_class_ids
# ############################################################
# #  Region Proposal Network
# ############################################################

# class RPN(nn.Module):
#     """Builds the model of Region Proposal Network.
#     anchors_per_location: number of anchors per pixel in the feature map
#     anchor_stride: Controls the density of anchors. Typically 1 (anchors for
#                    every pixel in the feature map), or 2 (every other pixel).
#     Returns:
#         rpn_logits: [batch, H, W, 2] Anchor classifier logits (before softmax)
#         rpn_probs: [batch, W, W, 2] Anchor classifier probabilities.
#         rpn_bbox: [batch, H, W, (dy, dx, log(dh), log(dw))] Deltas to be
#                   applied to anchors.
#     """

#     def __init__(self, anchors_per_location, anchor_stride, depth):
#         super(RPN, self).__init__()
#         self.anchors_per_location = anchors_per_location
#         self.anchor_stride = anchor_stride
#         self.depth = depth

#         self.padding = SamePad2d(kernel_size=3, stride=self.anchor_stride)
#         self.conv_shared = nn.Conv2d(self.depth, 512, kernel_size=3, stride=self.anchor_stride)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv_class = nn.Conv2d(512, 2 * anchors_per_location, kernel_size=1, stride=1)
#         self.softmax = nn.Softmax(dim=2)
#         self.conv_bbox = nn.Conv2d(512, 4 * anchors_per_location, kernel_size=1, stride=1)

#     def forward(self, x):
#         # Shared convolutional base of the RPN
#         x = self.relu(self.conv_shared(self.padding(x)))

#         # Anchor Score. [batch, anchors per location * 2, height, width].
#         rpn_class_logits = self.conv_class(x)

#         # Reshape to [batch, 2, anchors]
#         rpn_class_logits = rpn_class_logits.permute(0,2,3,1)
#         rpn_class_logits = rpn_class_logits.contiguous()
#         rpn_class_logits = rpn_class_logits.view(x.size()[0], -1, 2)

#         # Softmax on last dimension of BG/FG.
#         rpn_probs = self.softmax(rpn_class_logits)

#         # Bounding box refinement. [batch, H, W, anchors per location, depth]
#         # where depth is [x, y, log(w), log(h)]
#         rpn_bbox = self.conv_bbox(x)

#         # Reshape to [batch, 4, anchors]
#         rpn_bbox = rpn_bbox.permute(0,2,3,1)
#         rpn_bbox = rpn_bbox.contiguous()
#         rpn_bbox = rpn_bbox.view(x.size()[0], -1, 4)

#         return [rpn_class_logits, rpn_probs, rpn_bbox]


# ############################################################
# #  Feature Pyramid Network Heads
# ############################################################

# class Classifier(nn.Module):
#     def __init__(self, depth, pool_size, image_shape, num_classes):
#         super(Classifier, self).__init__()
#         self.depth = depth
#         self.pool_size = pool_size
#         self.image_shape = image_shape
#         self.num_classes = num_classes
#         self.conv1 = nn.Conv2d(self.depth, 1024, kernel_size=self.pool_size, stride=1)
#         self.bn1 = nn.BatchNorm2d(1024, eps=0.001, momentum=0.01)
#         self.conv2 = nn.Conv2d(1024, 1024, kernel_size=1, stride=1)
#         self.bn2 = nn.BatchNorm2d(1024, eps=0.001, momentum=0.01)
#         self.relu = nn.ReLU(inplace=True)

#         self.linear_class = nn.Linear(1024, num_classes)
#         self.softmax = nn.Softmax(dim=1)

#         self.linear_bbox = nn.Linear(1024, num_classes * 4)

#     def forward(self, x, rois):
#         x = pyramid_roi_align([rois]+x, self.pool_size, self.image_shape)
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu(x)

#         x = x.view(-1,1024)
#         mrcnn_class_logits = self.linear_class(x)
#         mrcnn_probs = self.softmax(mrcnn_class_logits)

#         mrcnn_bbox = self.linear_bbox(x)
#         mrcnn_bbox = mrcnn_bbox.view(mrcnn_bbox.size()[0], -1, 4)

#         return [mrcnn_class_logits, mrcnn_probs, mrcnn_bbox]

# class Mask(nn.Module):
#     def __init__(self, depth, pool_size, image_shape, num_classes):
#         super(Mask, self).__init__()
#         self.depth = depth
#         self.pool_size = pool_size
#         self.image_shape = image_shape
#         self.num_classes = num_classes
#         self.padding = SamePad2d(kernel_size=3, stride=1)
#         self.conv1 = nn.Conv2d(self.depth, 256, kernel_size=3, stride=1)
#         self.bn1 = nn.BatchNorm2d(256, eps=0.001)
#         self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
#         self.bn2 = nn.BatchNorm2d(256, eps=0.001)
#         self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
#         self.bn3 = nn.BatchNorm2d(256, eps=0.001)
#         self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
#         self.bn4 = nn.BatchNorm2d(256, eps=0.001)
#         self.deconv = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
#         self.conv5 = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
#         self.sigmoid = nn.Sigmoid()
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x, rois):
#         x = pyramid_roi_align([rois] + x, self.pool_size, self.image_shape)
#         x = self.conv1(self.padding(x))
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.conv2(self.padding(x))
#         x = self.bn2(x)
#         x = self.relu(x)
#         x = self.conv3(self.padding(x))
#         x = self.bn3(x)
#         x = self.relu(x)
#         x = self.conv4(self.padding(x))
#         x = self.bn4(x)
#         x = self.relu(x)
#         x = self.deconv(x)
#         x = self.relu(x)
#         x = self.conv5(x)
#         x = self.sigmoid(x)

#         return x




# class Config(object):
#     """Base configuration class. For custom configurations, create a
#     sub-class that inherits from this one and override properties
#     that need to be changed.
#     """
#     # Name the configurations. For example, 'COCO', 'Experiment 3', ...etc.
#     # Useful if your code needs to do things differently depending on which
#     # experiment is running.
#     NAME = None  # Override in sub-classes

#     # Path to pretrained imagenet model
#     IMAGENET_MODEL_PATH = os.path.join(os.getcwd(), "resnet50_imagenet.pth")

#     # NUMBER OF GPUs to use. For CPU use 0
#     GPU_COUNT = 1

#     # Number of images to train with on each GPU. A 12GB GPU can typically
#     # handle 2 images of 1024x1024px.
#     # Adjust based on your GPU memory and image sizes. Use the highest
#     # number that your GPU can handle for best performance.
#     IMAGES_PER_GPU = 1

#     # Number of training steps per epoch
#     # This doesn't need to match the size of the training set. Tensorboard
#     # updates are saved at the end of each epoch, so setting this to a
#     # smaller number means getting more frequent TensorBoard updates.
#     # Validation stats are also calculated at each epoch end and they
#     # might take a while, so don't set this too small to avoid spending
#     # a lot of time on validation stats.
#     STEPS_PER_EPOCH = 1000

#     # Number of validation steps to run at the end of every training epoch.
#     # A bigger number improves accuracy of validation stats, but slows
#     # down the training.
#     VALIDATION_STEPS = 50

#     # The strides of each layer of the FPN Pyramid. These values
#     # are based on a Resnet101 backbone.
#     BACKBONE_STRIDES = [4, 8, 16, 32, 64]

#     # Number of classification classes (including background)
#     NUM_CLASSES = 1  # Override in sub-classes

#     # Length of square anchor side in pixels
#     RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)

#     # Ratios of anchors at each cell (width/height)
#     # A value of 1 represents a square anchor, and 0.5 is a wide anchor
#     RPN_ANCHOR_RATIOS = [0.5, 1, 2]

#     # Anchor stride
#     # If 1 then anchors are created for each cell in the backbone feature map.
#     # If 2, then anchors are created for every other cell, and so on.
#     RPN_ANCHOR_STRIDE = 1

#     # Non-max suppression threshold to filter RPN proposals.
#     # You can reduce this during training to generate more propsals.
#     RPN_NMS_THRESHOLD = 0.7

#     # How many anchors per image to use for RPN training
#     RPN_TRAIN_ANCHORS_PER_IMAGE = 256

#     # ROIs kept after non-maximum supression (training and inference)
#     POST_NMS_ROIS_TRAINING = 2000
#     POST_NMS_ROIS_INFERENCE = 1000

#     # If enabled, resizes instance masks to a smaller size to reduce
#     # memory load. Recommended when using high-resolution images.
#     USE_MINI_MASK = True
#     MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

#     # Input image resing
#     # Images are resized such that the smallest side is >= IMAGE_MIN_DIM and
#     # the longest side is <= IMAGE_MAX_DIM. In case both conditions can't
#     # be satisfied together the IMAGE_MAX_DIM is enforced.
#     IMAGE_MIN_DIM = 800
#     IMAGE_MAX_DIM = 1024
#     # If True, pad images with zeros such that they're (max_dim by max_dim)
#     IMAGE_PADDING = True  # currently, the False option is not supported

#     # Image mean (RGB)
#     # MEAN_PIXEL = np.array([123.7, 116.8, 103.9])
#     MEAN_PIXEL = np.array([0.485, 0.456, 0.406])
#     STD_PIXEL = np.array([0.229, 0.224, 0.225])

#     # Number of ROIs per image to feed to classifier/mask heads
#     # The Mask RCNN paper uses 512 but often the RPN doesn't generate
#     # enough positive proposals to fill this and keep a positive:negative
#     # ratio of 1:3. You can increase the number of proposals by adjusting
#     # the RPN NMS threshold.
#     TRAIN_ROIS_PER_IMAGE = 200

#     # Percent of positive ROIs used to train classifier/mask heads
#     ROI_POSITIVE_RATIO = 0.33

#     # Pooled ROIs
#     POOL_SIZE = 7
#     MASK_POOL_SIZE = 14
#     MASK_SHAPE = [28, 28]

#     # Maximum number of ground truth instances to use in one image
#     MAX_GT_INSTANCES = 100

#     # Bounding box refinement standard deviation for RPN and final detections.
#     RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
#     BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])

#     # Max number of final detections
#     DETECTION_MAX_INSTANCES = 100

#     # Minimum probability value to accept a detected instance
#     # ROIs below this threshold are skipped
#     DETECTION_MIN_CONFIDENCE = 0.7

#     # Non-maximum suppression threshold for detection
#     DETECTION_NMS_THRESHOLD = 0.3

#     # Learning rate and momentum
#     # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
#     # weights to explode. Likely due to differences in optimzer
#     # implementation.
#     LEARNING_RATE = 0.001
#     LEARNING_MOMENTUM = 0.9

#     # Weight decay regularization
#     WEIGHT_DECAY = 0.0001

#     # Use RPN ROIs or externally generated ROIs for training
#     # Keep this True for most situations. Set to False if you want to train
#     # the head branches on ROI generated by code rather than the ROIs from
#     # the RPN. For example, to debug the classifier head without having to
#     # train the RPN.
#     USE_RPN_ROIS = True

#     def __init__(self):
#         """Set values of computed attributes."""
#         # Effective batch size
#         if self.GPU_COUNT > 0:
#             self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT
#         else:
#             self.BATCH_SIZE = self.IMAGES_PER_GPU

#         # Adjust step size based on batch size
#         self.STEPS_PER_EPOCH = self.BATCH_SIZE * self.STEPS_PER_EPOCH

#         # Input image size
#         self.IMAGE_SHAPE = np.array(
#             [self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM, 3])

#         # Compute backbone size from input image size
#         self.BACKBONE_SHAPES = np.array(
#             [[int(math.ceil(self.IMAGE_SHAPE[0] / stride)),
#               int(math.ceil(self.IMAGE_SHAPE[1] / stride))]
#              for stride in self.BACKBONE_STRIDES])

#     def display(self):
#         """Display Configuration values."""
#         print("\nConfigurations:")
#         for a in dir(self):
#             if not a.startswith("__") and not callable(getattr(self, a)):
#                 print("{:30} {}".format(a, getattr(self, a)))
#         print("\n")




# """
# Mask R-CNN
# Common utility functions and classes.

# Copyright (c) 2017 Matterport, Inc.
# Licensed under the MIT License (see LICENSE for details)
# Written by Waleed Abdulla
# """

# import sys
# import os
# import math
# import random
# import numpy as np
# import scipy.misc
# import scipy.ndimage
# import skimage.color
# import skimage.io
# import torch

# ############################################################
# #  Bounding Boxes
# ############################################################

# def extract_bboxes(mask):
#     """Compute bounding boxes from masks.
#     mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

#     Returns: bbox array [num_instances, (y1, x1, y2, x2)].
#     """
#     boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
#     for i in range(mask.shape[-1]):
#         m = mask[:, :, i]
#         # Bounding box.
#         horizontal_indicies = np.where(np.any(m, axis=0))[0]
#         vertical_indicies = np.where(np.any(m, axis=1))[0]
#         if horizontal_indicies.shape[0]:
#             x1, x2 = horizontal_indicies[[0, -1]]
#             y1, y2 = vertical_indicies[[0, -1]]
#             # x2 and y2 should not be part of the box. Increment by 1.
#             x2 += 1
#             y2 += 1
#         else:
#             # No mask for this instance. Might happen due to
#             # resizing or cropping. Set bbox to zeros
#             x1, x2, y1, y2 = 0, 0, 0, 0
#         boxes[i] = np.array([y1, x1, y2, x2])
#     return boxes.astype(np.int32)


# def compute_iou(box, boxes, box_area, boxes_area):
#     """Calculates IoU of the given box with the array of the given boxes.
#     box: 1D vector [y1, x1, y2, x2]
#     boxes: [boxes_count, (y1, x1, y2, x2)]
#     box_area: float. the area of 'box'
#     boxes_area: array of length boxes_count.

#     Note: the areas are passed in rather than calculated here for
#           efficency. Calculate once in the caller to avoid duplicate work.
#     """
#     # Calculate intersection areas
#     y1 = np.maximum(box[0], boxes[:, 0])
#     y2 = np.minimum(box[2], boxes[:, 2])
#     x1 = np.maximum(box[1], boxes[:, 1])
#     x2 = np.minimum(box[3], boxes[:, 3])
#     intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
#     union = box_area + boxes_area[:] - intersection[:]
#     iou = intersection / union
#     return iou


# def compute_overlaps(boxes1, boxes2):
#     """Computes IoU overlaps between two sets of boxes.
#     boxes1, boxes2: [N, (y1, x1, y2, x2)].

#     For better performance, pass the largest set first and the smaller second.
#     """
#     # Areas of anchors and GT boxes
#     area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
#     area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

#     # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
#     # Each cell contains the IoU value.
#     overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
#     for i in range(overlaps.shape[1]):
#         box2 = boxes2[i]
#         overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
#     return overlaps

# def box_refinement(box, gt_box):
#     """Compute refinement needed to transform box to gt_box.
#     box and gt_box are [N, (y1, x1, y2, x2)]
#     """

#     height = box[:, 2] - box[:, 0]
#     width = box[:, 3] - box[:, 1]
#     center_y = box[:, 0] + 0.5 * height
#     center_x = box[:, 1] + 0.5 * width

#     gt_height = gt_box[:, 2] - gt_box[:, 0]
#     gt_width = gt_box[:, 3] - gt_box[:, 1]
#     gt_center_y = gt_box[:, 0] + 0.5 * gt_height
#     gt_center_x = gt_box[:, 1] + 0.5 * gt_width

#     dy = (gt_center_y - center_y) / height
#     dx = (gt_center_x - center_x) / width
#     dh = torch.log(gt_height / height)
#     dw = torch.log(gt_width / width)

#     result = torch.stack([dy, dx, dh, dw], dim=1)
#     return result


# ############################################################
# #  Dataset
# ############################################################

# class Dataset(object):
#     """The base class for dataset classes.
#     To use it, create a new class that adds functions specific to the dataset
#     you want to use. For example:

#     class CatsAndDogsDataset(Dataset):
#         def load_cats_and_dogs(self):
#             ...
#         def load_mask(self, image_id):
#             ...
#         def image_reference(self, image_id):
#             ...

#     See COCODataset and ShapesDataset as examples.
#     """

#     def __init__(self, class_map=None):
#         self._image_ids = []
#         self.image_info = []
#         # Background is always the first class
#         self.class_info = [{"source": "", "id": 0, "name": "BG"}]
#         self.source_class_ids = {}

#     def add_class(self, source, class_id, class_name):
#         assert "." not in source, "Source name cannot contain a dot"
#         # Does the class exist already?
#         for info in self.class_info:
#             if info['source'] == source and info["id"] == class_id:
#                 # source.class_id combination already available, skip
#                 return
#         # Add the class
#         self.class_info.append({
#             "source": source,
#             "id": class_id,
#             "name": class_name,
#         })

#     def add_image(self, source, image_id, path, **kwargs):
#         image_info = {
#             "id": image_id,
#             "source": source,
#             "path": path,
#         }
#         image_info.update(kwargs)
#         self.image_info.append(image_info)

#     def image_reference(self, image_id):
#         """Return a link to the image in its source Website or details about
#         the image that help looking it up or debugging it.

#         Override for your dataset, but pass to this function
#         if you encounter images not in your dataset.
#         """
#         return ""

#     def prepare(self, class_map=None):
#         """Prepares the Dataset class for use.

#         TODO: class map is not supported yet. When done, it should handle mapping
#               classes from different datasets to the same class ID.
#         """
#         def clean_name(name):
#             """Returns a shorter version of object names for cleaner display."""
#             return ",".join(name.split(",")[:1])

#         # Build (or rebuild) everything else from the info dicts.
#         self.num_classes = len(self.class_info)
#         self.class_ids = np.arange(self.num_classes)
#         self.class_names = [clean_name(c["name"]) for c in self.class_info]
#         self.num_images = len(self.image_info)
#         self._image_ids = np.arange(self.num_images)

#         self.class_from_source_map = {"{}.{}".format(info['source'], info['id']): id
#                                       for info, id in zip(self.class_info, self.class_ids)}

#         # Map sources to class_ids they support
#         self.sources = list(set([i['source'] for i in self.class_info]))
#         self.source_class_ids = {}
#         # Loop over datasets
#         for source in self.sources:
#             self.source_class_ids[source] = []
#             # Find classes that belong to this dataset
#             for i, info in enumerate(self.class_info):
#                 # Include BG class in all datasets
#                 if i == 0 or source == info['source']:
#                     self.source_class_ids[source].append(i)

#     def map_source_class_id(self, source_class_id):
#         """Takes a source class ID and returns the int class ID assigned to it.

#         For example:
#         dataset.map_source_class_id("coco.12") -> 23
#         """
#         return self.class_from_source_map[source_class_id]

#     def get_source_class_id(self, class_id, source):
#         """Map an internal class ID to the corresponding class ID in the source dataset."""
#         info = self.class_info[class_id]
#         assert info['source'] == source
#         return info['id']

#     def append_data(self, class_info, image_info):
#         self.external_to_class_id = {}
#         for i, c in enumerate(self.class_info):
#             for ds, id in c["map"]:
#                 self.external_to_class_id[ds + str(id)] = i

#         # Map external image IDs to internal ones.
#         self.external_to_image_id = {}
#         for i, info in enumerate(self.image_info):
#             self.external_to_image_id[info["ds"] + str(info["id"])] = i

#     @property
#     def image_ids(self):
#         return self._image_ids

#     def source_image_link(self, image_id):
#         """Returns the path or URL to the image.
#         Override this to return a URL to the image if it's availble online for easy
#         debugging.
#         """
#         return self.image_info[image_id]["path"]

#     def load_image(self, image_id):
#         """Load the specified image and return a [H,W,3] Numpy array.
#         """
#         # Load image
#         image = skimage.io.imread(self.image_info[image_id]['path'])
#         # If grayscale. Convert to RGB for consistency.
#         if image.ndim != 3:
#             image = skimage.color.gray2rgb(image)
#         return image

#     def load_mask(self, image_id):
#         """Load instance masks for the given image.

#         Different datasets use different ways to store masks. Override this
#         method to load instance masks and return them in the form of am
#         array of binary masks of shape [height, width, instances].

#         Returns:
#             masks: A bool array of shape [height, width, instance count] with
#                 a binary mask per instance.
#             class_ids: a 1D array of class IDs of the instance masks.
#         """
#         # Override this function to load a mask from your dataset.
#         # Otherwise, it returns an empty mask.
#         mask = np.empty([0, 0, 0])
#         class_ids = np.empty([0], np.int32)
#         return mask, class_ids


# def resize_image(image, min_dim=None, max_dim=None, padding=False):
#     """
#     Resizes an image keeping the aspect ratio.

#     min_dim: if provided, resizes the image such that it's smaller
#         dimension == min_dim
#     max_dim: if provided, ensures that the image longest side doesn't
#         exceed this value.
#     padding: If true, pads image with zeros so it's size is max_dim x max_dim

#     Returns:
#     image: the resized image
#     window: (y1, x1, y2, x2). If max_dim is provided, padding might
#         be inserted in the returned image. If so, this window is the
#         coordinates of the image part of the full image (excluding
#         the padding). The x2, y2 pixels are not included.
#     scale: The scale factor used to resize the image
#     padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
#     """
#     # Default window (y1, x1, y2, x2) and default scale == 1.
#     h, w = image.shape[:2]
#     window = (0, 0, h, w)
#     scale = 1

#     # Scale?
#     if min_dim:
#         # Scale up but not down
#         scale = max(1, min_dim / min(h, w))
#     # Does it exceed max dim?
#     if max_dim:
#         image_max = max(h, w)
#         if round(image_max * scale) > max_dim:
#             scale = max_dim / image_max
#     # Resize image and mask
#     if scale != 1:
#         image = scipy.misc.imresize(
#             image, (round(h * scale), round(w * scale)))
#     # Need padding?
#     if padding:
#         # Get new height and width
#         h, w = image.shape[:2]
#         top_pad = (max_dim - h) // 2
#         bottom_pad = max_dim - h - top_pad
#         left_pad = (max_dim - w) // 2
#         right_pad = max_dim - w - left_pad
#         padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
#         image = np.pad(image, padding, mode='constant', constant_values=0)
#         window = (top_pad, left_pad, h + top_pad, w + left_pad)
#     return image, window, scale, padding


# def resize_mask(mask, scale, padding):
#     """Resizes a mask using the given scale and padding.
#     Typically, you get the scale and padding from resize_image() to
#     ensure both, the image and the mask, are resized consistently.

#     scale: mask scaling factor
#     padding: Padding to add to the mask in the form
#             [(top, bottom), (left, right), (0, 0)]
#     """
#     h, w = mask.shape[:2]
#     mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1], order=0)
#     mask = np.pad(mask, padding, mode='constant', constant_values=0)
#     return mask


# def minimize_mask(bbox, mask, mini_shape):
#     """Resize masks to a smaller version to cut memory load.
#     Mini-masks can then resized back to image scale using expand_masks()

#     See inspect_data.ipynb notebook for more details.
#     """
#     mini_mask = np.zeros(mini_shape + (mask.shape[-1],), dtype=bool)
#     for i in range(mask.shape[-1]):
#         m = mask[:, :, i]
#         y1, x1, y2, x2 = bbox[i][:4]
#         m = m[y1:y2, x1:x2]
#         if m.size == 0:
#             raise Exception("Invalid bounding box with area of zero")
#         m = scipy.misc.imresize(m.astype(float), mini_shape, interp='bilinear')
#         mini_mask[:, :, i] = np.where(m >= 128, 1, 0)
#     return mini_mask


# def expand_mask(bbox, mini_mask, image_shape):
#     """Resizes mini masks back to image size. Reverses the change
#     of minimize_mask().

#     See inspect_data.ipynb notebook for more details.
#     """
#     mask = np.zeros(image_shape[:2] + (mini_mask.shape[-1],), dtype=bool)
#     for i in range(mask.shape[-1]):
#         m = mini_mask[:, :, i]
#         y1, x1, y2, x2 = bbox[i][:4]
#         h = y2 - y1
#         w = x2 - x1
#         m = scipy.misc.imresize(m.astype(float), (h, w), interp='bilinear')
#         mask[y1:y2, x1:x2, i] = np.where(m >= 128, 1, 0)
#     return mask


# # TODO: Build and use this function to reduce code duplication
# def mold_mask(mask, config):
#     pass


# def unmold_mask(mask, bbox, image_shape):
#     """Converts a mask generated by the neural network into a format similar
#     to it's original shape.
#     mask: [height, width] of type float. A small, typically 28x28 mask.
#     bbox: [y1, x1, y2, x2]. The box to fit the mask in.

#     Returns a binary mask with the same size as the original image.
#     """
#     threshold = 0.5
#     y1, x1, y2, x2 = bbox
#     mask = scipy.misc.imresize(
#         mask, (y2 - y1, x2 - x1), interp='bilinear').astype(np.float32) / 255.0
#     mask = np.where(mask >= threshold, 1, 0).astype(np.uint8)

#     # Put the mask in the right location.
#     full_mask = np.zeros(image_shape[:2], dtype=np.uint8)
#     full_mask[y1:y2, x1:x2] = mask
#     return full_mask


# ############################################################
# #  Anchors
# ############################################################

# def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
#     """
#     scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
#     ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
#     shape: [height, width] spatial shape of the feature map over which
#             to generate anchors.
#     feature_stride: Stride of the feature map relative to the image in pixels.
#     anchor_stride: Stride of anchors on the feature map. For example, if the
#         value is 2 then generate anchors for every other feature map pixel.
#     """
#     # Get all combinations of scales and ratios
#     scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
#     scales = scales.flatten()
#     ratios = ratios.flatten()

#     # Enumerate heights and widths from scales and ratios
#     heights = scales / np.sqrt(ratios)
#     widths = scales * np.sqrt(ratios)

#     # Enumerate shifts in feature space
#     shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
#     shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
#     shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

#     # Enumerate combinations of shifts, widths, and heights
#     box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
#     box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

#     # Reshape to get a list of (y, x) and a list of (h, w)
#     box_centers = np.stack(
#         [box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
#     box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

#     # Convert to corner coordinates (y1, x1, y2, x2)
#     boxes = np.concatenate([box_centers - 0.5 * box_sizes,
#                             box_centers + 0.5 * box_sizes], axis=1)
#     return boxes


# def generate_pyramid_anchors(scales, ratios, feature_shapes, feature_strides,
#                              anchor_stride):
#     """Generate anchors at different levels of a feature pyramid. Each scale
#     is associated with a level of the pyramid, but each ratio is used in
#     all levels of the pyramid.

#     Returns:
#     anchors: [N, (y1, x1, y2, x2)]. All generated anchors in one array. Sorted
#         with the same order of the given scales. So, anchors of scale[0] come
#         first, then anchors of scale[1], and so on.
#     """
#     # Anchors
#     # [anchor_count, (y1, x1, y2, x2)]
#     anchors = []
#     for i in range(len(scales)):
#         anchors.append(generate_anchors(scales[i], ratios, feature_shapes[i],
#                                         feature_strides[i], anchor_stride))
#     return np.concatenate(anchors, axis=0)



# def compose_image_meta(image_id, image_shape, window, active_class_ids):
#     """Takes attributes of an image and puts them in one 1D array. Use
#     parse_image_meta() to parse the values back.
#     image_id: An int ID of the image. Useful for debugging.
#     image_shape: [height, width, channels]
#     window: (y1, x1, y2, x2) in pixels. The area of the image where the real
#             image is (excluding the padding)
#     active_class_ids: List of class_ids available in the dataset from which
#         the image came. Useful if training on images from multiple datasets
#         where not all classes are present in all datasets.
#     """
#     meta = np.array(
#         [image_id] +            # size=1
#         list(image_shape) +     # size=3
#         list(window) +          # size=4 (y1, x1, y2, x2) in image cooredinates
#         list(active_class_ids)  # size=num_classes
#     )
#     return meta



# class MaskRCNN_MCG(MaskRCNN):
#     def __init__(self, train_set, **model_options):
#         super().__init__(train_set, **model_options)
#         self.proposal_type = "MCG"