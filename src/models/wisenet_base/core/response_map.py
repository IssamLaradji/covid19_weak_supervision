import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch.autograd import Variable
import numpy as np 
import misc as ms
from skimage import morphology as morph
from torch.autograd import Function

from pycocotools import mask as maskUtils




def get_backprop_guided_pixels(model, images, points, 
                              which=None):
 
    points_single, point_class = helpers.points2single(points, which)

    points_single = ms.t2n(points_single)
    mask = torch.from_numpy(ms.mask2hot(points_single, model.n_classes)).float()

    excited_mask = guided_backprop(model, images, gradient=mask.cuda())
    
    return excited_mask, point_class


def relu_hook_function(module, grad_in, grad_out):
    # print("clamped")
    return (torch.clamp(grad_in[0], min=0.0),)

def guided_backprop(model, images,  gradient=None):
    # Loop through layers, hook up ReLUs with relu_hook_function

    with torch.enable_grad():
        handleList = []
        for module in model.modules():
            if module.__class__.__name__ == 'ReLU':
                handleList += [module.register_backward_hook(relu_hook_function)]

        x = images.cuda()
        x.requires_grad = True
        out = model(x)

        #Zero gradients
        model.zero_grad()
        out.backward(gradient=gradient)

        for handle in handleList:
            handle.remove()

        return ms.t2n(x.grad.data)

from types import MethodType

def peak_response(model, img, class_threshold=0, peak_threshold=1, retrieval_cfg=None):
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            module._original_forward = module.forward
            module.forward = MethodType(pr_conv2d, module)

    assert img.dim() == 4
    
    img.requires_grad_()

    # classification network forwarding
    class_response_maps = model.forward(img)
      
    # aggregate responses from informative receptive fields estimated via class peak responses
    peak_list, aggregation = peak_stimulation(class_response_maps, win_size=3,
                                               peak_filter=median_filter)


    # extract instance-aware visual cues, i.e., peak response maps
    assert class_response_maps.size(0) == 1, 'Currently inference mode (with peak backpropagation) only supports one image at a time.'
    if peak_list is None:
        peak_list = peak_stimulation(class_response_maps, return_aggregation=False,
                         win_size=3, peak_filter=median_filter)

    peak_response_maps = []
    valid_peak_list = []
    # peak backpropagation
    grad_output = class_response_maps.new_empty(class_response_maps.size())
    for idx in range(peak_list.size(0)):
        if aggregation[peak_list[idx, 0], peak_list[idx, 1]] >= class_threshold:
            peak_val = class_response_maps[peak_list[idx, 0], peak_list[idx, 1], peak_list[idx, 2], peak_list[idx, 3]]
            if peak_val > peak_threshold:
                grad_output.zero_()

                # starting from the peak
                grad_output[peak_list[idx, 0], peak_list[idx, 1], peak_list[idx, 2], peak_list[idx, 3]] = 1
                if img.grad is not None:
                    img.grad.zero_()
                import ipdb; ipdb.set_trace()  # breakpoint 6f5f5f20 //

                class_response_maps.backward(grad_output, retain_graph=True)
                prm = img.grad.detach().sum(1).clone().clamp(min=0)
                peak_response_maps.append(prm / prm.sum())
                valid_peak_list.append(peak_list[idx, :])
    
    # return results
    class_response_maps = class_response_maps.detach()
    aggregation = aggregation.detach()

    if len(peak_response_maps) > 0:
        valid_peak_list = torch.stack(valid_peak_list)
        peak_response_maps = torch.cat(peak_response_maps, 0)
        
        # classification confidence scores, class-aware and instance-aware visual cues
        return aggregation, class_response_maps, valid_peak_list, peak_response_maps
  
    else:
        return None

def median_filter(img):
    batch_size, num_channels, h, w = img.size()
    threshold, _ = torch.median(img.view(batch_size, num_channels, h * w), dim=2)
    return threshold.contiguous().view(batch_size, num_channels, 1, 1)


class PreHook(Function):
    
    @staticmethod
    def forward(ctx, input, offset):
        ctx.save_for_backward(input, offset)
        return input.clone()
    
    @staticmethod
    def backward(ctx, grad_output):
        input, offset = ctx.saved_variables
        return (input - offset) * grad_output, None
    
class PostHook(Function):
    
    @staticmethod
    def forward(ctx, input, norm_factor):
        ctx.save_for_backward(norm_factor)
        return input.clone()
    
    @staticmethod
    def backward(ctx, grad_output):
        norm_factor, = ctx.saved_variables
        eps = 1e-10
        zero_mask = norm_factor < eps
        grad_input = grad_output / (torch.abs(norm_factor) + eps)
        grad_input[zero_mask.detach()] = 0
        return None, grad_input


def pr_conv2d(self, input):
    offset = input.min().detach()
    input = PreHook.apply(input, offset)
    resp = F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups).detach()
    pos_weight = F.relu(self.weight).detach()
    norm_factor = F.conv2d(input - offset, pos_weight, None, self.stride, self.padding, self.dilation, self.groups)
    output = PostHook.apply(resp, norm_factor)
    return output

class PeakStimulation(Function):

    @staticmethod
    def forward(ctx, input, return_aggregation, win_size, peak_filter):
        ctx.num_flags = 4

        # peak finding
        assert win_size % 2 == 1, 'Window size for peak finding must be odd.'
        offset = (win_size - 1) // 2
        padding = torch.nn.ConstantPad2d(offset, float('-inf'))
        padded_maps = padding(input)
        batch_size, num_channels, h, w = padded_maps.size()
        element_map = torch.arange(0, h * w).long().view(1, 1, h, w)[:, :, offset: -offset, offset: -offset]
        element_map = element_map.to(input.device)
        _, indices  = F.max_pool2d(
            padded_maps,
            kernel_size = win_size, 
            stride = 1, 
            return_indices = True)
        peak_map = (indices == element_map)

        # peak filtering
        if peak_filter:
            mask = input >= peak_filter(input)
            peak_map = (peak_map & mask)
        peak_list = torch.nonzero(peak_map)
        ctx.mark_non_differentiable(peak_list)
        
        # peak aggregation
        if return_aggregation:
            peak_map = peak_map.float()
            ctx.save_for_backward(input, peak_map)
            return peak_list, (input * peak_map).view(batch_size, num_channels, -1).sum(2) / \
                peak_map.view(batch_size, num_channels, -1).sum(2)
        else:
            return peak_list

    @staticmethod
    def backward(ctx, grad_peak_list, grad_output):
        input, peak_map, = ctx.saved_tensors
        batch_size, num_channels, _, _ = input.size()
        grad_input = peak_map * grad_output.view(batch_size, num_channels, 1, 1)
        return (grad_input,) + (None,) * ctx.num_flags


def peak_stimulation(input, return_aggregation=True, win_size=3, peak_filter=None):
    return PeakStimulation.apply(input, return_aggregation, win_size, peak_filter)