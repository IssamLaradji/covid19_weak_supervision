import numpy as np
#import minimizers

import torch
from sklearn.utils.extmath import randomized_svd
from numpy.linalg import norm
import scipy.sparse
import PIL.Image
import numpy as np
from sklearn.metrics import pairwise_distances
import pydensecrf.densecrf as dcrf
from scipy import sparse
from skimage.segmentation import find_boundaries
#from sklearn.metrics.pairwise import rbf_kernel
from .. import misc as ms
import torch.nn.functional as F
from skimage.morphology import watershed
from scipy import ndimage
# def compute_split_loss(S_log, S, points, blob_dict, split_mode="line"):
#     blobs = blob_dict["blobs"]
#     S_numpy = ms.t2n(S[0])
#     points_numpy = ms.t2n(points).squeeze() 

#     loss = 0.

#     for b in blob_dict["blobList"]:
#         if b["n_points"] < 2:
#             continue

#         l = b["class"] + 1
#         probs = S_numpy[b["class"] + 1]

#         points_class = (points_numpy==l).astype("int")
#         blob_ind = blobs[b["class"] ] == b["label"]

#         if split_mode == "line":
#             T = line_splits(probs*blob_ind, points_class*blob_ind)
#         elif split_mode == "water":
#             T = watersplit(probs, points_class*blob_ind)*blob_ind
#             T = 1 - T
#         else:
#             raise ValueError("%s LOL" % split_mode)

#         scale = b["n_points"] + 1
#         loss += float(scale) * F.nll_loss(S_log, ms.n2l(T)[None],
#                         ignore_index=1, size_average=True)

#     return loss 


def line_splits(probs_, points_):
    img = probs_.copy()
    if img.ndim == 2:
        img = img[:,:,None]

    points = points_.copy().astype(float)

    res = np.zeros(points_.shape)
    pointsArr = np.vstack(np.where(points==1)).T

    pairs = set()
    D = np.argsort(pairwise_distances(pointsArr), axis=1)
   

    for i in range(pointsArr.shape[0]):
        p1 = pointsArr[i]
        p2 = pointsArr[D[i][1]]
   
        p1p2 = tuple(np.hstack([p1, p2]).tolist())
        p2p1 = tuple(np.hstack([p2, p1]).tolist())

        if p1p2 in pairs or p2p1 in pairs:
            continue
        else:
            pairs.add(p1p2)

        pointList = create_line(p1, p2, dydx_only=False)["line"]

        lineList = create_perpline(pointList, p1, p2, points.shape)
        #print("%s, pointList: %d, lines: %d" % (str(pointList), len(pointList), len(lines)))
        min_score = np.inf
        best_line = None

        for i, l in enumerate(lineList):
            mask = np.zeros(img.shape[:2])
      
            mask[l[:,0],l[:,1]] = 1


            #print(mask)
            vals = img[:,:,0] * mask
            nz = vals.nonzero()
      
            #score = vals[nz].max()
            score = vals[nz].mean()
            #print(mask)
            if score < min_score:
                best_line = lineList[i]
                min_score = score
                
        if best_line is not None:
            res[best_line[:,0], best_line[:,1]] = 1

    
    return 1 - (res.astype(int) * (probs_!=0))


def watersplit(_probs, _points):
   seg = water_regions(_probs, _points)

   return find_boundaries(seg)

def water_regions(_probs, _points, return_dict=False):
   points = _points.copy()
   if return_dict:
       y_ind, x_ind = np.where(points!=0)
       yx_list = []
       for i in range(y_ind.shape[0]):
         label = i+1
         points[y_ind[i], x_ind[i]] = label
         yx_list += [{"y":y_ind[i], "x":x_ind[i], "label":label}]

       points = points.astype(float)
   else:
       points[points!=0] = np.arange(1, points.sum()+1)
       points = points.astype(float)

   
   probs = ndimage.black_tophat(_probs.copy(), 7)   

   seg =  watershed(probs, points)

   if return_dict:
      return  {"seg":seg, "yx_list": yx_list}

   return seg






def create_line(p1, p2, dydx_only=False):
    ys = p1[0]
    ye = p2[0]

    xs = p1[1]
    xe = p2[1]

    ydiff = ye - ys
    xdiff = xe - xs

    if abs(ydiff) > abs(xdiff):

        # across y
        dy = np.sign(ydiff)
        if ydiff == 0:
            dy = 0
        else:
            dx = abs(xdiff / ydiff) * np.sign(xdiff)
        end = abs(ydiff)+1

    if abs(ydiff) <= abs(xdiff):
        # across x
        dx = np.sign(xdiff)
        if xdiff == 0:
            dy = 0
        else:
            dy = abs(ydiff / xdiff) * np.sign(ydiff)

        end = abs(xdiff)+1

    if dydx_only:
        return dy, dx

    iList = np.arange(end)
    line_ind = np.zeros((end, 2))

    line_ind[:, 0] = ys + iList*dy
    line_ind[:, 1] = xs + iList*dx 

    #print(line_ind)
    return {"line": line_ind.astype(int), "dy":dy, "dx":dx}


def create_perpline(pointList, p1, p2, shape):
    # if not isinstance(pointList, list):
    #     pointList = [pointList]
    h, w = shape

    dy, dx = create_line(p1, p2, dydx_only=1)

    if dy == 0:
        # Horizontal line - Get Vertical Perpendicular
        dy = 1
        dx = 0
        iList = np.arange(-h,h)

    elif dx == 0:
        # Vertical line - Get Horizontal Perpendicular
        dx = 1
        dy = 0
        iList = np.arange(-w,w)

    elif abs(dx) > abs(dy):
        # Otherwise
        dy = -1. / dy
        dx = dx
        iList = np.arange(-h,h)

    else:
        dx = -1. / dx
        dy = dy
        iList = np.arange(-w,w)


    norm = max(abs(dx), abs(dy))
    dx = dx/norm
    dy = dy/norm


    perpList = []
    for p in pointList[1:-1]:
        ys = p[0]
        xs = p[1]

        r = ys + iList*dy
        #print(r)
        iListTmp = iList[(r>=0) * (r<h)]
        r = xs + iListTmp*dx
        iListTmp = iListTmp[(r>=0) * (r<w)]

        new = np.zeros((iListTmp.shape[0], 2)).astype(int)


        new[:, 0] = ys + iListTmp*dy
        new[:, 1] = xs + iListTmp*dx 

        # Set indices within range

        # # new[:,0] = np.maximum(new[:,0], 0)
        # # new[:,0] = np.minimum(new[:,0], h-1)
        # new[:,1] = np.maximum(new[:,1], 0)
        # new[:,1] = np.minimum(new[:,1], w-1)     
        perpList += [new]

    return perpList

