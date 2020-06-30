# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision import models
# import torch
# import torch.nn as nn
# import torchvision
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from functools import partial
# from torch.autograd import Variable
# import numpy as np 
# from .. import misc as ms
# from skimage import morphology as morph
# from torch.autograd import Function

# from ..core import blobs_utils as bu
# # from core import proposals as prp
# from . import base_model as bm
# from ..addons.pycocotools import mask as maskUtils
# # from core import score_functions as sf
# from .. import ann_utils as au
# import cv2
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# from ..datasets import base_dataset

# class PRM(bm.BaseModel):
#     def __init__(self, train_set, **model_options):
#         super().__init__(train_set, **model_options)
#         num_classes = train_set.n_classes - 1
#         model = torchvision.models.resnet50(pretrained=True)
#         # feature encoding
#         self.features = nn.Sequential(
#             model.conv1,
#             model.bn1,
#             model.relu,
#             model.maxpool,
#             model.layer1,
#             model.layer2,
#             model.layer3,
#             model.layer4)

#         # classifier
#         num_features = model.layer4[1].conv1.in_channels
#         self.classifier = nn.Sequential(
#             nn.Conv2d(num_features, num_classes, kernel_size=1, bias=True))

#         # # FREEZE BATCH NORMS
#         for m in self.modules():
#             if isinstance(m, nn.BatchNorm2d):
#                 m.weight.requires_grad = False
#                 m.bias.requires_grad = False


#     def forward(self, x):
#         self.features.eval()
#         x = self.features(x)
#         class_response_maps = self.classifier(x)
#         self.sub_pixel_locating_factor = 8
#         self.win_size, self.peak_filter = 3, _median_filter

#         class_response_maps = F.upsample(class_response_maps, scale_factor=self.sub_pixel_locating_factor, mode='bilinear', align_corners=True)
#         # aggregate responses from informative receptive fields estimated via class peak responses
#         peak_list, aggregation = peak_stimulation(class_response_maps, win_size=self.win_size, peak_filter=self.peak_filter)

#         return aggregation

#     @torch.no_grad()
#     def forward_test(self, x_input, class_threshold=0, peak_threshold = 30):
#         self.features.eval()
#         with torch.enable_grad():
#             x_input.requires_grad=True
#             points = np.zeros(x_input.shape[-2:])
#             x = self.features(x_input)
#             class_response_maps = self.classifier(x)
#             # class_response_maps.requires_grad=True

#             if self.sub_pixel_locating_factor > 1:
#                 class_response_maps = F.upsample(class_response_maps, scale_factor=self.sub_pixel_locating_factor, mode='bilinear', align_corners=True)
#             # aggregate responses from informative receptive fields estimated via class peak responses
#             peak_list, aggregation = peak_stimulation(class_response_maps, win_size=self.win_size, peak_filter=self.peak_filter)


            
#             # extract instance-aware visual cues, i.e., peak response maps
#             if peak_list is None:
#                 peak_list = peak_stimulation(class_response_maps, return_aggregation=False, win_size=self.win_size, peak_filter=self.peak_filter)

#             peak_response_maps = []
#             valid_peak_list = []
#             # peak backpropagation
#             grad_output = class_response_maps.new_empty(class_response_maps.size())
#             for idx in range(peak_list.size(0)):
#                 if aggregation[peak_list[idx, 0], peak_list[idx, 1]] >= class_threshold:
#                     peak_val = class_response_maps[peak_list[idx, 0], peak_list[idx, 1], peak_list[idx, 2], peak_list[idx, 3]]
#                     if peak_val > peak_threshold:
#                         grad_output.zero_()
#                         # starting from the peak
#                         grad_output[peak_list[idx, 0], 
#                                     peak_list[idx, 1], 
#                                     peak_list[idx, 2], 
#                                     peak_list[idx, 3]] = 1

#                         # points[peak_list[idx, 2], peak_list[idx, 3]] = 1
#                         if x_input.grad is not None:
#                             x_input.grad.zero_()
#                         class_response_maps.backward(grad_output, retain_graph=True)
#                         prm = x_input.grad.detach().sum(1).clone().clamp(min=0)
                        

#                         points[ms.t2n(prm==prm.max()).astype(bool).squeeze()] = peak_list[idx, 1].item() + 1
                        
#                         peak_response_maps.append(prm / prm.sum())
#                         valid_peak_list.append(peak_list[idx, :])
#         #  self.classifier[0].weight.grad
#         # print(points.sum())
#         # return results
#         self.zero_grad()

#         class_response_maps = class_response_maps.detach()
#         aggregation = aggregation.detach()

#         if len(peak_response_maps) > 0:

#             valid_peak_list = torch.stack(valid_peak_list)
#             peak_response_maps = torch.cat(peak_response_maps, 0)

#             # classification confidence scores, class-aware and instance-aware visual cues
#             return {"a":aggregation, "c":class_response_maps, 
#                     "v":valid_peak_list, "p":peak_response_maps, 
#                     "points":points}
         
#         else:
#             return {"points":points}

#     def get_points(self, batch):

#         results = self.predict(batch, "ewr") 

#         return results["points"]


#     @torch.no_grad()
#     def extract_proposalMasks(self, batch):
#         self.eval()
#         import ipdb; ipdb.set_trace()  # breakpoint aa7e62ed //

#         proposals = base_dataset.SharpProposals(batch)
#         import ipdb; ipdb.set_trace()  # breakpoint cd8bb791 //

#         n,c,h,w = batch["images"].shape
#         x_input = batch["images"].cuda()
#         O = self(x_input)
#         if predict_method == "counts":
#             return {"counts":ms.t2n(torch.sigmoid(O)>0.5)}
#         else:
#             return self.forward_test(x_input, class_threshold=0, peak_threshold = 30)


#     @torch.no_grad()
#     def predict(self, batch, predict_method="counts", **options):
#         self.eval()

#         n,c,h,w = batch["images"].shape
#         x_input = batch["images"].cuda()
#         O = self(x_input)
#         if predict_method == "counts":
#             return {"counts":ms.t2n(torch.sigmoid(O)>0.5)}

#         if predict_method == "points":
#             pred_dict =  self.forward_test(x_input, class_threshold=0, peak_threshold = 30)
#             return pred_dict["points"]
            
#         elif predict_method == "BestDice":
            
#             points = self.get_points(batch)
#             pointList = au.mask2pointList(points[None])["pointList"]
#             if len(pointList) == 0:
#                 return {"annList":[]}
#             pred_dict = au.pointList2BestObjectness(pointList, batch)

#             return {"annList":pred_dict["annList"]}

#         else:
#             return self.forward_test(x_input, class_threshold=0, peak_threshold = 30)

#     @torch.no_grad()
#     def visualize(self, batch, predict_method="counts", i=0, **options):
#         results = self.predict(batch, "ewr") 

#         mask = results[3][i]
#         ms.images(ms.gray2cmap(mask), win="mask")
#         ms.images(batch["images"], mask>0.5, denorm=1)
#         ms.images(batch["images"], results[-1].astype(int),win="points",
#                      enlarge=1,denorm=1)

# def _median_filter(input):
#     batch_size, num_channels, h, w = input.size()
#     threshold, _ = torch.median(input.view(batch_size, num_channels, h * w), dim=2)
#     return threshold.contiguous().view(batch_size, num_channels, 1, 1)

# class PeakStimulation(Function):

#     @staticmethod
#     def forward(ctx, input, return_aggregation, win_size, peak_filter):
#         ctx.num_flags = 4

#         # peak finding
#         assert win_size % 2 == 1, 'Window size for peak finding must be odd.'
#         offset = (win_size - 1) // 2
#         padding = torch.nn.ConstantPad2d(offset, float('-inf'))
#         padded_maps = padding(input)
#         batch_size, num_channels, h, w = padded_maps.size()
#         element_map = torch.arange(0, h * w).long().view(1, 1, h, w)[:, :, offset: -offset, offset: -offset]
#         element_map = element_map.to(input.device)
#         _, indices  = F.max_pool2d(
#             padded_maps,
#             kernel_size = win_size, 
#             stride = 1, 
#             return_indices = True)
#         peak_map = (indices == element_map)

#         # peak filtering
#         if peak_filter:
#             mask = input >= peak_filter(input)
#             peak_map = (peak_map & mask)
#         peak_list = torch.nonzero(peak_map)
#         ctx.mark_non_differentiable(peak_list)
        
#         # peak aggregation
#         if return_aggregation:
#             peak_map = peak_map.float()
#             ctx.save_for_backward(input, peak_map)
#             return peak_list, (input * peak_map).view(batch_size, num_channels, -1).sum(2) / \
#                 peak_map.view(batch_size, num_channels, -1).sum(2)
#         else:
#             return peak_list

#     @staticmethod
#     def backward(ctx, grad_peak_list, grad_output):
#         input, peak_map, = ctx.saved_tensors
#         batch_size, num_channels, _, _ = input.size()
#         grad_input = peak_map * grad_output.view(batch_size, num_channels, 1, 1)
#         return (grad_input,) + (None,) * ctx.num_flags


#     def instance_nms(self, instance_list, threshold=0.3, merge_peak_response=True):
#         selected_instances = []
#         while len(instance_list) > 0:
#             instance = instance_list.pop(0)
#             selected_instances.append(instance)
#             src_mask = instance[2].astype(bool)
#             src_peak_response = instance[3]
#             def iou_filter(x):
#                 dst_mask = x[2].astype(bool)
#                 # IoU
#                 intersection = np.logical_and(src_mask, dst_mask).sum()
#                 union = np.logical_or(src_mask, dst_mask).sum()
#                 iou = intersection / (union + 1e-10)
#                 if iou < threshold:
#                     return x
#                 else:
#                     if merge_peak_response:
#                         nonlocal src_peak_response
#                         src_peak_response += x[3]
#                     return None
#             instance_list = list(filter(iou_filter, instance_list))
#         return selected_instances

    

# def peak_stimulation(input, return_aggregation=True, win_size=3, peak_filter=None):
#     return PeakStimulation.apply(input, return_aggregation, win_size, peak_filter)
