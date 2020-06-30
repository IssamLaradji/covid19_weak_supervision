# import os
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from functools import partial
# from torch.autograd import Variable
# import numpy as np 
# import misc as ms
# from skimage import morphology as morph
# from torch.autograd import Function

# from pycocotools import mask as maskUtils

# def predict_sharpmask(model, batch, points,
#                     path="/mnt/datasets/public/issam/VOCdevkit/" \
#                          "proposals/sharpmask/pascal_proposals/",
#                          score_function=None):
    
#     name = batch["name"][0]
#     if "2007" not in name:
#         name = name.replace("_","")

#     else:
#         name = str(int(name.split("_")[1]))

    
#     n, c = batch["counts"].shape
#     n, _, h, w = batch["images"].shape
#     blobs = np.zeros((n, c, h, w), int)

#     for j in range((points!=0).sum()):
#         #########################################
#         sal, label_class = np.abs(predict_saliency_guided(model, 
#                                   batch["images"], points, 
#                                   label_class=None, which=j)).squeeze()

#         sal_mean = sal.mean(0)
#         #########################################

#         best_score = 0

#         proposals = ms.load_json(path + "{}.json".format(name))
#         proposals = sorted(proposals, key=lambda x:x["score"], 
#                        reverse=True)
        
#         for k in range(100):
#             mask = maskUtils.decode(proposals[k]["segmentation"])

#             if score_function == None:
#                 score = (sal_mean*mask).mean()
                
#             if score > best_score:
#                 best_mask = mask
#                 best_score = score 

#         object_label = blobs[0, label_class-1].max()+1        
#         blobs[0, label_class-1, best_mask!=0] = object_label


#     return blobs

# def predict_saliency_guided(model, images, points, 
#                             label_class=None, which=None):
#     label = points

#     if label_class is not None:
#         label[label!=label_class] = 0

#     if which is not None:
#         points_loc = np.hstack([np.where(label!=0)]).T
#         i,j,k = points_loc[which]

#         label_class = label[i,j,k]
#         label = np.zeros(label.shape, dtype=int)
#         label[i,j,k] = int(label_class)

#     label = ms.t2n(label)
#     mask = torch.from_numpy(ms.mask2hot(label, model.n_classes)).float()

#     sal = guided_saliency(model, images, gradient=mask.cuda())
    
#     return sal, label_class




# def relu_hook_function(module, grad_in, grad_out):
#     # print("clamped")
#     return (torch.clamp(grad_in[0], min=0.0),)

# def guided_saliency(model, images,  gradient=None):
#     # Loop through layers, hook up ReLUs with relu_hook_function
#     handleList = []
#     for module in model.modules():
#         if module.__class__.__name__ == 'ReLU':
#             handleList += [module.register_backward_hook(relu_hook_function)]

#     x = images.cuda()
#     x.requires_grad = True
#     out = model(x)

#     #Zero gradients
#     model.zero_grad()
#     out.backward(gradient=gradient)

#     for handle in handleList:
#         handle.remove()

#     return ms.t2n(x.grad.data)

















# class _GlobalConvModule(nn.Module):
#     def __init__(self, in_dim, out_dim, kernel_size):
#         super(_GlobalConvModule, self).__init__()
#         pad0 = (kernel_size[0] - 1) // 2
#         pad1 = (kernel_size[1] - 1) // 2
#         # kernel size had better be odd number so as to avoid alignment error
#         super(_GlobalConvModule, self).__init__()
#         self.conv_l1 = nn.Conv2d(in_dim, out_dim, kernel_size=(kernel_size[0], 1),
#                                  padding=(pad0, 0))
#         self.conv_l2 = nn.Conv2d(out_dim, out_dim, kernel_size=(1, kernel_size[1]),
#                                  padding=(0, pad1))
#         self.conv_r1 = nn.Conv2d(in_dim, out_dim, kernel_size=(1, kernel_size[1]),
#                                  padding=(0, pad1))
#         self.conv_r2 = nn.Conv2d(out_dim, out_dim, kernel_size=(kernel_size[0], 1),
#                                  padding=(pad0, 0))

#     def forward(self, x):
#         x_l = self.conv_l1(x)
#         x_l = self.conv_l2(x_l)
#         x_r = self.conv_r1(x)
#         x_r = self.conv_r2(x_r)
#         x = x_l + x_r
#         return x


# class _BoundaryRefineModule(nn.Module):
#     def __init__(self, dim):
#         super(_BoundaryRefineModule, self).__init__()
#         self.relu = nn.ReLU(inplace=True)
#         self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

#     def forward(self, x):
#         residual = self.conv1(x)
#         residual = self.relu(residual)
#         residual = self.conv2(residual)
#         out = x + residual
#         return out


# def initialize_weights(*models):
#     for model in models:
#         for module in model.modules():
#             if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
#                 nn.init.kaiming_normal_(module.weight)
#                 if module.bias is not None:
#                     module.bias.data.zero_()
#             elif isinstance(module, nn.BatchNorm2d):
#                 module.weight.data.fill_(1)
#                 module.bias.data.zero_()



# def get_upsampling_weight(in_channels, out_channels, kernel_size):
#     """Make a 2D bilinear kernel suitable for upsampling"""
#     factor = (kernel_size + 1) // 2
#     if kernel_size % 2 == 1:
#         center = factor - 1
#     else:
#         center = factor - 0.5
#     og = np.ogrid[:kernel_size, :kernel_size]
#     filt = (1 - abs(og[0] - center) / factor) * \
#            (1 - abs(og[1] - center) / factor)
#     weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
#                       dtype=np.float64)
#     weight[range(in_channels), range(out_channels), :, :] = filt
#     return torch.from_numpy(weight).float()


# def conv3x3(in_planes, out_planes, stride=1, padding=1):
#     "3x3 convolution with padding"
#     return nn.Conv2d(in_planes, out_planes, kernel_size=(3,3), stride=(stride,stride),
#                      padding=(padding,padding))

# def conv1x1(in_planes, out_planes, stride=1):
#     "1x1 convolution with padding"
#     return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
#                      padding=0)





# class GuidedBackpropReLUModel:
#     def __init__(self, model, use_cuda):
#         self.model = model
#         self.model.eval()
#         self.cuda = use_cuda
#         if self.cuda:
#             self.model = model.cuda()

#         for module in model.modules():
#           if module.__class__.__name__ == 'ReLU':
#               print("ReLU")


#         # replace ReLU with GuidedBackpropReLU
#         for module in self.modules():
#             if module.__class__.__name__ == 'ReLU':
#                  module = GuidedBackpropReLU

#     def forward(self, input):
#         return self.model(input)

#     def __call__(self, input, index = None):
#         if self.cuda:
#             output = self.forward(input.cuda())
#         else:
#             output = self.forward(input)

#         if index == None:
#             index = np.argmax(output.cpu().data.numpy())

#         one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
#         one_hot[0][index] = 1
#         one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True)
#         if self.cuda:
#             one_hot = torch.sum(one_hot.cuda() * output)
#         else:
#             one_hot = torch.sum(one_hot * output)

#         # self.model.features.zero_grad()
#         # self.model.classifier.zero_grad()
#         one_hot.backward(retain_variables=True)

#         output = input.grad.cpu().data.numpy()
#         output = output[0,:,:,:]

#         return output

# class GuidedBackpropReLU(Function):

#     def forward(self, input):
#         positive_mask = (input > 0).type_as(input)
#         output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
#         self.save_for_backward(input, output)
#         return output

#     def backward(self, grad_output):
#         input, output = self.saved_tensors
#         grad_input = None

#         positive_mask_1 = (input > 0).type_as(grad_output)
#         positive_mask_2 = (grad_output > 0).type_as(grad_output)
#         grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input), torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output, positive_mask_1), positive_mask_2)

#         return grad_input


# # def unregister_relus(model):
# #     def relu_hook_function(module, grad_in, grad_out):
# #         print("clamped")
# #         return (torch.clamp(grad_in[0], min=0.0),)

# #     # Loop through layers, hook up ReLUs with relu_hook_function
# #     for module in model.modules():
# #         if module.__class__.__name__ == 'ReLU':
# #             module.register_backward_hook(relu_hook_function)

