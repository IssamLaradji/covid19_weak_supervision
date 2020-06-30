import torch
import skimage
import torch.nn.functional as F
import numpy as np
from skimage.morphology import watershed
from skimage.segmentation import find_boundaries
from scipy import ndimage
from skimage import morphology as morph
from skimage.morphology import watershed
from skimage.segmentation import find_boundaries
import kornia

def compute_density_loss(logits, points, sigma=1):
    kernel_size = (3, 3)
    sigma_list = (sigma, sigma)
    gfilter = kornia.filters.get_gaussian_kernel2d(kernel_size, sigma_list)
    density = kornia.filters.filter2D(points[None].float(), kernel=gfilter[None], border_type='reflect') 
   
    diff = (logits[:, 1] - density)**2
    loss = torch.sqrt(diff.mean())

    return loss