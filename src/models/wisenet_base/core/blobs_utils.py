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
# from skimage.morphology import watershed
# from pycocotools import mask as maskUtils
# from scipy import ndimage
# from datasets import base_dataset

# def se_pairwise(fi, fj):
#     return (fi - fj).pow(2).sum(1)

# def log_pairwise(fi, fj):
#     diff = se_pairwise(fi, fj)
#     return  (2./ (1+torch.exp(diff))).clamp(min=1e-6, max=(1.-1e-6))


# def get_random_indices(mask, n_indices=10):
#     mask_ind = np.where(mask.squeeze())
#     n_pixels = mask_ind[0].shape[0]
#     P_ind = np.random.randint(0, n_pixels, n_indices)
#     yList = mask_ind[0][P_ind]
#     xList = mask_ind[1][P_ind]

#     return {"yList":yList, "xList":xList}



# def mask2seed(mask, n_indices=10):
#     return get_random_indices(mask, n_indices=n_indices)






# def CombineSeeds(seedList, ind=None):
#     yList = []
#     xList = []
#     categoryList = []

#     if ind is None:
#         ind = range(len(seedList))

#     for i in ind:
#         yList += seedList[i]["yList"]
#         xList += seedList[i]["xList"]
#         categoryList += seedList[i]["category_id"]

#     assert len(categoryList) == len(yList) 
#     return {"yList":yList, "xList":xList, "categoryList":categoryList}


# def propDict2seedList(propDict, n_neighbors=100, random_proposal=False):
#     seedList = []
#     for prop in propDict["propDict"]:
#         if len(prop["annList"]) == 0:
#             seedList += [{"category_id":[prop["point"]["category_id"]],
#                            "yList":[prop["point"]["y"]],   
#                           "xList":[prop["point"]["x"]],   
#                           "neigh":{"yList":[prop["point"]["y"]],
#                                     "xList":[prop["point"]["x"]]}}]

#         else:
#             if random_proposal:
#                 i = np.random.randint(0, len(prop["annList"]))
#                 mask = prop["annList"][i]["mask"]
#             else:
#                 mask = prop["annList"][0]["mask"]
                
#             seedList += [{"category_id":[prop["point"]["category_id"]],
#                            "yList":[prop["point"]["y"]],   
#                           "xList":[prop["point"]["x"]],   
#                           "neigh":get_random_indices(mask, n_indices=100)}]

#     # Background
#     background = propDict["background"]
#     if background.sum() == 0:
#         y_axis = np.random.randint(0, background.shape[1],100)
#         x_axis = np.random.randint(0, background.shape[2],100)
#         background[0,y_axis, x_axis] = 1
#     bg_seeds = get_random_indices(background, n_indices=len(propDict["propDict"]))
#     seedList += [{"category_id":[0]*len(bg_seeds["yList"]),
#                     "yList":bg_seeds["yList"].tolist(), 
#                   "xList":bg_seeds["xList"].tolist(), 
#                   "neigh":get_random_indices(background, n_indices=100)}] 

#     return seedList

# def pointList2points(pointList):
#     return pointList2mask(pointList)
# def pointList2BestObjectness(pointList, batch):

#     propDict = pointList2propDict(pointList, batch, thresh=0.5)
    
#     annList = []
#     h,w = propDict["background"].squeeze().shape
#     blobs = np.zeros((h,w), int)
#     categoryDict = {}
#     maskVoid = batch["maskVoid"]
#     for i, prop in enumerate(propDict['propDict']):
#         if len(prop["annList"]) == 0:
#             continue
#         blobs[prop["annList"][0]["mask"] !=0] = i+1

        
#         categoryDict[i+1] = prop["category_id"]

#         if maskVoid is not None:
#             binmask = prop["annList"][0]["mask"] * (ms.t2n(maskVoid).squeeze())
#         else:
#             binmask = prop["annList"][0]["mask"]

#         seg = maskUtils.encode(np.asfortranarray(ms.t2n(binmask)).astype("uint8")) 
#         seg["counts"] = seg["counts"].decode("utf-8")
#         score = prop["annList"][0]["score"]

#         annList += [{"segmentation":seg,
#               "iscrowd":0,
#               "area":int(maskUtils.area(seg)),
#              "image_id":batch["name"][0],
#              "category_id":int(prop['category_id']),
#              "height":h,
#              "width":w,
#              "score":score}]


#     return {"annList":annList, "blobs": blobs, "categoryDict":categoryDict}

# def blobs2BestDice(blobs, categoryDict, propDict, batch):
#     h, w = blobs.shape
#     annList = []
#     blobs_copy = np.zeros(blobs.shape, int)
#     maskVoid = batch["maskVoid"]
#     for u in np.unique(blobs):
#         if u == 0:
#             continue
#         binmask = (blobs == u)
#         best_dice = 0.
#         best_mask = None
#         for ann in propDict['propDict'][u-1]["annList"]:

#             score = sf.dice(ann["mask"], binmask)
#             if score > best_dice:
#                 best_dice = score
#                 best_mask = ann["mask"]
#                 prop_score = ann["score"]

#         if best_mask is None:
#             best_mask = (blobs==u).astype(int)


#         if maskVoid is not None:
#             binmask = best_mask * (ms.t2n(maskVoid).squeeze())
#         else:
#             binmask = best_mask

#         if best_mask is None:
#             blobs_copy[blobs==u] = u 
#         else:
#             blobs_copy[best_mask==1] = u


#         seg = maskUtils.encode(np.asfortranarray(ms.t2n(binmask)).astype("uint8")) 
#         seg["counts"] = seg["counts"].decode("utf-8")
#         score = best_dice


#         annList += [{"segmentation":seg,
#               "iscrowd":0,
#               "area":int(maskUtils.area(seg)),
#              "image_id":batch["name"][0],
#              "category_id":int(categoryDict[u]),
#              "height":h,
#              "width":w,
#              "score":score}]
        
#     return {"blobs":blobs_copy, "annList":annList}

# def blobs2annList(blobs, categoryDict, batch):
#     maskVoid = batch["maskVoid"]
#     h,w = blobs.shape
#     annList = []
#     blobs = ms.t2n(blobs)

#     for u in np.unique(blobs):
#         if u == 0:
#             continue
#         binmask = (blobs == u)
#         if maskVoid is not None:
#             binmask = binmask * (ms.t2n(maskVoid).squeeze())

#         seg = maskUtils.encode(np.asfortranarray(ms.t2n(binmask)).astype("uint8")) 
#         seg["counts"] = seg["counts"].decode("utf-8")
#         score = 1

#         annList += [{"segmentation":seg,
#               "iscrowd":0,
#               "area":int(maskUtils.area(seg)),
#              "image_id":batch["name"][0],
#              "category_id":int(categoryDict[u]),
#              "height":h,
#              "width":w,
#              "score":score}]

#     return annList

# def json2blobs(imgAnn):
#     pass

# def annList2counts(annList):
#     counts = np.zeros(21)
#     for a in annList:
#        counts[a["category_id"]] += 1


#     # mask[mask==1] = ann["category_id"]
#     return {"counts":counts[1:][None]}





# def annList2points(annList):
#     n_anns = len(annList)
#     if n_anns == 0:
#         return {"points":None}

#     points =  np.zeros((annList[0]["height"], annList[0]["width"]), int)

#     for i in range(n_anns):
#         ann = annList[i]
#         p = ann["point"]
#         points[p["y"], p["x"]] = p["category_id"]

#     return {"points":points}

# def annList2mask(annList, box=False):
#     n_anns = len(annList)
#     if n_anns == 0:
#         return {"mask":None}

#     ann = annList[0]
#     if "mask" in ann:
#         mask = ann["mask"].copy()
#     else:
#         mask = maskUtils.decode(ann["segmentation"])

#     for i in range(1, n_anns):
#         ann = annList[i]
#         if "mask" in ann:
#             mask += ann["mask"]
#         else:
#             mask += maskUtils.decode(ann["segmentation"])

#     for i in range(n_anns):
#         if box:
#             box_mask = ann2bbox(annList[i])["mask"]
#             mask[box_mask!=0] = 255

#     # mask[mask==1] = ann["category_id"]
#     return {"mask":mask}

# def ann2mask(ann):
#     mask =  maskUtils.decode(ann["segmentation"])
#     # mask[mask==1] = ann["category_id"]
#     return {"mask":mask}



# def pointList2mask(pointList):
    
#     mask = np.zeros(pointList[0]["shape"], int)
#     for p in pointList:
#         mask[:, p["y"], p["x"]] = p["category_id"]

#     return {"mask":mask}

# # from core import proposals as prp
# def batch2propDict(batch):
#     pointList = mask2pointList(batch["points"])["pointList"]
#     propList = base_dataset.SharpProposals(batch["name"])

#     propDict = pointList2propDict(pointList, propList, 0.5)
#     return propDict

# def pointList2pairList(pointList):
#     pairList = []
#     n_points = len(pointList)

#     for i in range(n_points):
#         p = pointList[i]
#         category_id = p["category_id"]

#         p_closest = None
#         dist = np.inf

#         for j in range(i+1, n_points): 
#             p_tmp = pointList[j]
#             if p_tmp["category_id"] == category_id:
#                 p_dist = np.abs(p_tmp["y"] - p["y"])
#                 p_dist += np.abs(p_tmp["x"] - p["x"])

#                 if p_dist < dist: 
#                     p_closest = p_tmp
#                     dist = p_dist

#         if p_closest is not None:
#             pairList += [{"p1":p, "p2":p_closest, 
#                           "category_id":category_id,
#                           "distance":dist}]

#     return pairList

# from skimage.segmentation import find_boundaries
# def probs2splitMask_pairs(probs, pointList=None):
#     pairList = pointList2pairList(pointList)
#     probs = ms.t2n(probs)

#     if pointList is None:
#         pointList = probs2blobs(probs)["pointList"]

#     maskList = []
#     for pair in pairList:
#         probs_mask = probs[:, pair["category_id"]]
#         point_mask = np.zeros(probs_mask.shape).squeeze()

#         p1, p2 = pair["p1"], pair["p2"]
#         point_mask[p1["y"], p1["x"]] = 1
#         point_mask[p2["y"], p2["x"]] = 2

#         distance = ndimage.black_tophat(probs_mask.squeeze(), 7)
#         mask =  1 - find_boundaries(watershed(distance, point_mask))

#         maskList += [{"mask":mask, 
#                       "category_id":pair["category_id"]}]

#     return {"maskList":maskList}

# def probs2GtAnnList(probs, points):
#     points = ms.t2n(points.squeeze())
#     annList = probs2annList(probs)["annList"]

#     for ann in annList:
#         ann["gt_pointList"] = []
#         mask = ann2mask(ann)["mask"]
#         binmask = mask * (points == ann['category_id'])

#         n_points = binmask.sum()
#         if n_points == 1:
#             ann["status"] = "TP"
#             ann_points = np.vstack(np.where(binmask)).T
#             p = ann_points[0]
#             ann["gt_pointList"] += [{"y":p[0], "x":p[1], "category_id":ann["category_id"]}]

#         if n_points > 1: 
#             ann["status"] = "SP"

#             ann_points = np.vstack(np.where(binmask)).T
#             for i in range(ann_points.shape[0]):
#                 p = ann_points[i]
#                 ann["gt_pointList"] += [{"y":p[0], "x":p[1], "category_id":ann["category_id"]}]

#         if n_points == 0: 
#             ann["status"] = "FP"
            

#     return annList

# def probs2splitMask_category(probs, pointList=None):
#     probs = ms.t2n(probs)

#     if pointList is None:
#         pointList = probs2blobs(probs)["pointList"]

#     categories = []
#     for p in pointList:
#         categories += [int(p["category_id"])]
#     categories = set(categories)

#     maskList = []
#     n,c,h,w = probs.shape
#     background = np.zeros((n,h,w), int)
#     for category_id in categories:
#         probs_mask = probs[:, category_id]
#         point_mask = np.zeros(probs_mask.shape).squeeze()

#         point_id = 0
#         for p in pointList:
#             if p["category_id"] != category_id:
#                 continue
            
#             point_id += 1
#             # ewr
#             point_mask[p["y"], p["x"]] = point_id

#         distance = ndimage.black_tophat(probs_mask.squeeze(), 7)    
#         mask = find_boundaries(watershed(distance, point_mask))
        
#         background += mask

#         maskList += [{"mask":1-mask, 
#                       "category_id":category_id}]
#     background = background.clip(0,1)
#     return {"maskList":maskList, "background":1-background}


# def probs2splitMask_all(probs, pointList=None):
#     probs = ms.t2n(probs)

#     if pointList is None:
#         pointList = probs2blobs(probs)["pointList"]

#     categories = []
#     for p in pointList:
#         categories += [p["category_id"]]
#     categories = set(categories)

#     maskList = []
#     n,c,h,w = probs.shape
#     background = np.zeros((n,h,w), int)

#     point_id = 0
#     point_mask = np.zeros( probs[:, 0].shape).squeeze()
#     probs_mask = probs[:,1:].max(1)


#     for p in pointList:
#         point_id += 1
#         point_mask[p["y"], p["x"]] = point_id

#     distance = ndimage.black_tophat(probs_mask.squeeze(), 7)    
#     mask = find_boundaries(watershed(distance, point_mask))
    
#     background += mask

#     maskList += [{"mask":1-mask}]

#     background = background.clip(0,1)
#     return {"maskList":maskList, "background":1-background}

# def get_batches(n_pixels, size=500000):
#     batches = []
#     for i in range(0, n_pixels, size):
#         batches +=[(i, i+size)]
#     return batches

# @torch.no_grad()
# def get_embedding_blobs(O, fg_bg_seeds):
#     n, c, h, w = O.shape
#     # seeds = torch.cat([fg_seeds, bg_seeds], 2)[:,:,None]
#     fA = O.view(1,c,-1)
#     fS = O[:,:, fg_bg_seeds["yList"], fg_bg_seeds["xList"]]

#     n_pixels = h*w
#     blobs = torch.zeros(h*w)

#     n_seeds =  fS.shape[-1]

#     maximum = 5000000
#     n_loops = int(np.ceil((n_pixels * n_seeds) / maximum))
#     for (s,e) in get_batches(n_pixels, size=n_pixels//n_loops):
#         # s,e = map(int, (s,e))
#         diff = log_pairwise(fS[:,:,None], fA[:,:,s:e,None]) 
#         blobs[s:e] = diff.max(2)[1] + 1 
    
#     bg_min_index = np.where(np.array(fg_bg_seeds["categoryList"])==0)[0].min()
#     assert len(fg_bg_seeds["yList"])//2 == bg_min_index
#     blobs[blobs > int(bg_min_index)] = 0
#     blobs = blobs.squeeze().reshape(h,w).long()

#     categoryDict = {}
#     for i, category_id in enumerate(fg_bg_seeds["categoryList"]):
#         if category_id == 0:
#              continue

#         categoryDict[i+1] = category_id 

#     return {"blobs":ms.t2n(blobs), "categoryDict":categoryDict}

# # @torch.no_grad()
# # def pointList2propDict_old(pointList, propList, single_point=False, thresh=0.5):
# #     propDict = []
# #     shape = pointList[0]["shape"]
# #     foreground = np.zeros(shape, int)

# #     selected = []

# #     for i, p in enumerate(pointList):
# #         idList = []
# #         annList = []
# #         for k in range(len(propList)):
# #             proposal_ann = propList[k]

# #             if not (proposal_ann["score"] > thresh):
# #                 continue

# #             proposal_mask =  proposal_ann["mask"]

# #             #########
# #             if proposal_mask[p["y"], p["x"]]==0:
# #                 continue

# #             if k in selected:
# #                 continue

# #             if (single_point and 
# #                 np.sum([proposal_mask[p_tmp["y"], p_tmp["x"]] for p_tmp in 
# #                     pointList]) > 1):
# #                 continue

# #             # if (proposal_mask*foreground).sum() > 0:
# #             #     continue


# #             # score = proposal_ann["score"]

# #             annList += [proposal_ann]
# #             idList += [k]
# #             selected += [k]

# #         mask = annList2mask(annList)["mask"]

# #         if mask is not None:
# #             foreground = foreground + mask

# #         #foreground[foreground<2]=0
# #         propDict += [{"annList":annList,"point":p, "idList":idList, 
# #                       "category_id":int(p["category_id"])}]
# #         #########   
# #     return {"propDict":propDict,"foreground":foreground, "background":(foreground==0).astype(int)}

# @torch.no_grad()
# def pointList2propDict(pointList, batch, single_point=False, thresh=0.5):
#     sharp_proposals = base_dataset.SharpProposals(batch)
#     propDict = []
#     shape = pointList[0]["shape"]
#     foreground = np.zeros(shape, int)

#     if single_point:
#         points = pointList2mask(pointList)["mask"]

#     idDict= {}
#     annDict = {}
#     for i, p in enumerate(pointList):
#         annDict[i] = []
#         idDict[i] = []

#     for k in range(len(sharp_proposals)):
#         proposal_ann = sharp_proposals[k]
#         if not (proposal_ann["score"] > thresh):
#             continue
#         proposal_mask =  proposal_ann["mask"]

#         for i, p in enumerate(pointList):
#             if proposal_mask[p["y"], p["x"]]==0:
#                 continue
            
#             if single_point and (points * proposal_mask).sum() > 1:
#                     continue

#             # score = proposal_ann["score"]
           
#             annDict[i] += [proposal_ann]
#             idDict[i] += [k]

#     for i in annDict:
#         annList = annDict[i]
#         idList = idDict[i]
#         p = pointList[i]

#         mask = annList2mask(annList)["mask"]
#         if mask is not None:
#             foreground = foreground + mask

#         #foreground[foreground<2]=0
#         propDict += [{"annList":annList,"point":p, "idList":idList, 
#                       "category_id":int(p["category_id"])}]
#         #########  

#     return {"propDict":propDict,"foreground":foreground, "background":(foreground==0).astype(int)}


# def point2propList(point, propList):
#     annList = []
#     idList = []
#     for k in range(len(propList)):
#         proposal_ann = propList[k]

#         if proposal_ann["score"] < 0.5:
#             continue

#         proposal_mask =  proposal_ann["mask"]

#         #########
#         if proposal_mask[point["y"], point["x"]]==0:
#             continue

#         # score = proposal_ann["score"]
#         annList += [proposal_ann]
#         idList += [k]
#         #########   
#     return {"annList":annList, "idList":idList}
        

# def points2pointList(points):
#     return mask2pointList(points)
# def mask2pointList(mask):
#     pointList = []
#     mask = ms.t2n(mask)
#     pointInd = np.where(mask.squeeze())
#     n_points = pointInd[0].size

#     for p in range(n_points):

#         p_y, p_x = pointInd[0][p], pointInd[1][p]
#         point_category = mask[0, p_y,p_x]

#         pointList += [{"y":p_y,"x":p_x, "category_id":int(point_category), 
#                        "shape":mask.shape}]

#     return {"pointList":pointList}

# def ann2points(ann):
#     point_matrix =  np.zeros((ann["height"], ann["width"]))
#     point_matrix[ann['point']['y'], ann['point']['x']] = 1
#     # mask[mask==1] = ann["category_id"]
#     return point_matrix

# def ann2bbox(ann):
#     bbox = maskUtils.toBbox(ann["segmentation"])
#     r, c = ann["segmentation"]["size"]
    
#     x, y, w, h = bbox

#     mask = bbox2mask(r, c, x, y, w, h)
#     ye = min(r-1, y+h)
#     xe = min(c-1, x+w) 

#     return {"mask":mask, "shape":(x, y, xe, ye)}

# def bbox2mask(r,c, x,y,w,h):
#     x,y,w,h = map(int, (x,y,w,h))    
#     mask = np.zeros((r, c), int) 

#     ye = min(r-1, y+h)
#     xe = min(c-1, x+w) 

#     mask[y:ye, x] = 1
#     mask[y:ye, xe] = 1
#     mask[y, x:xe] = 1
#     mask[ye, x:xe] = 1

#     return mask

# def mask2ann(binmask, category_id, image_id, 
#              height, width, maskVoid=None, score=None, point=None):
#     binmask = binmask.squeeze().astype("uint8")

#     if maskVoid is not None:
#         binmask = binmask * maskVoid

#     segmentation = maskUtils.encode(np.asfortranarray(ms.t2n(binmask)).astype("uint8")) 
#     segmentation["counts"] = segmentation["counts"].decode("utf-8")
#     # print(segmentation)
#     ann = {"segmentation":segmentation,
#                   "iscrowd":0,
#                   "area":int(maskUtils.area(segmentation)),
#                  "image_id":image_id,
#                  "category_id":int(category_id),
#                  "height":height,
#                  "width":width,
#                  "score":score,
#                  "point":point}

#     return ann

# # from core import score_functions as sf
# def ann2proposalRandom(ann, batch, matching_method="dice"):
#     sharp_proposals = base_dataset.SharpProposals(batch["name"])
#     pred_mask = ann2mask(ann)["mask"]
#     best_score = 0
#     best_mask = None
#     py, px = ann["point"]["y"], ann["point"]["x"]

#     probs = np.zeros(len(sharp_proposals))
#     for k in range(len(sharp_proposals)):
#         proposal_ann = sharp_proposals[k]

#         if proposal_ann["score"] < 0.5:
#             continue

#         proposal_mask =  proposal_ann["mask"]
#         if proposal_mask[py, px] != 1:
#             continue

#         # if (proposal_mask * counter_points.squeeze().clip(0,1)).sum() > 1:
#         #     continue
#         # print("lol")
#         #########
#         # print(np.unique(pred_mask))
#         probs[k] = sf.dice(pred_mask, proposal_mask)
    
#     if probs.sum() == 0:
#         k_best = np.random.choice(len(sharp_proposals))
#     else:
#         k_best = np.random.choice(len(sharp_proposals), p=probs/probs.sum())
        
#     proposal_ann = sharp_proposals[k_best]
#     best_mask =  proposal_ann["mask"]
#     return best_mask

# def ann2proposal(ann, batch, matching_method="dice"):
#     sharp_proposals = base_dataset.SharpProposals(batch["name"])
#     pred_mask = ann2mask(ann)["mask"]
#     best_score = 0
#     best_mask = None
#     py, px = ann["point"]["y"], ann["point"]["x"]
#     for k in range(len(sharp_proposals)):
#         proposal_ann = sharp_proposals[k]

#         if proposal_ann["score"] < 0.5:
#             continue

#         proposal_mask =  proposal_ann["mask"]
#         if proposal_mask[py, px] != 1:
#             continue

#         # if (proposal_mask * counter_points.squeeze().clip(0,1)).sum() > 1:
#         #     continue
#         # print("lol")
#         #########
#         # print(np.unique(pred_mask))
#         if matching_method == "dice":
#             score = sf.dice(pred_mask, proposal_mask)

#         if matching_method == "objectness":
#             score = proposal_ann["score"] 
#         #########   
#         if score > best_score:
#             best_mask = proposal_mask
#             best_score = score
#             # best_score = proposal_ann["score"] 

#     return best_mask

# @torch.no_grad()
# def probs2annList(probs, image_id=None):
#     probs = ms.t2n(probs).squeeze()
#     n_classes, h, w = probs.shape
    
#     mask_labels = ms.t2n(probs.argmax(0))
#     annList = []
#     # print(np.unique(mask_labels))
#     for category_id in np.unique(mask_labels):
#         if category_id == 0:
#             continue
#         # print("class", category_id)
#         # ms.images(mask_labels)
#         class_blobs = morph.label(mask_labels==category_id).squeeze()
#         # print(np.unique(class_blobs))
#         for u in np.unique(class_blobs):
#             if u == 0:
#                 continue
            
#             binmask = (class_blobs == u)
#             # ms.images(binmask)
#             # asdsa
#             ann = mask2ann(binmask, category_id, image_id, height=h, width=w)

#             # try:


#             sub_probs =  probs[category_id]*binmask

#             # except:
#             #     print(c)
#             #     print(probs.shape, binmask.shape)
#             #     sdasdasds
#             ind = np.where(sub_probs==sub_probs.max())
#             # print(ind)
#             r = ind[0][0]
#             c = ind[1][0]
            
#             ann["point"] = {"y":r, "x":c}
#             annList += [ann]
            
#     #         blobs[i,l-1] = morph.label(mask_labels==l)
#     #         counts[i, l-1] = (np.unique(blobs[i,l-1]) != 0).sum()

#     # blobs = blobs.astype(int)

#     # if return_counts:
#     #     return blobs, counts

#     return {"annList":annList}


# @torch.no_grad()
# def points2annList(points):
#     n, h, w = points.shape
#     pointList = points2pointList(points)["pointList"]
    
#     annList = []
#     # print(np.unique(mask_labels))
#     for p in pointList:
#         point_mask = np.zeros((h,w), int)
#         point_mask[p["y"], p["x"]] = p["category_id"]
#         ann = mask2ann(point_mask, p["category_id"], None, h, w)
#         ann["point"] = {"shape":(h,w),"y":p["y"], "x":p["x"], "category_id":p["category_id"]}
#         ann["shape"] = (h,w)
#         annList += [ann]
            
#     return {"annList":annList}

# def points2single(points, which, points_probs=None, 
#                   blobs=None, return_blobs=False):
#     points_single = np.zeros(points.shape, dtype=int) 
#     points_ind = np.where(points!=0)
#     points_loc = np.hstack([points_ind]).T


#     if points_probs is None:
#         order_ind = np.arange(points_loc.shape[0])
#     else:
#         order_ind = np.argsort(points_probs[points_ind])[::-1]

#     which = order_ind[which]
#     i,j,k = points_loc[which]

#     point_class = points[i,j,k]
#     points_single[i,j,k] = int(point_class)

#     if return_blobs:

#         return points_single, point_class, blobs_single

#     return points_single, point_class



# def probs2blobs(probs):
#     annList = []

#     probs = ms.t2n(probs)
#     n, n_classes, h, w = probs.shape
  
#     counts = np.zeros((n, n_classes-1))
    
#     # Binary case
#     pred_mask = ms.t2n(probs.argmax(1))
#     blobs = np.zeros(pred_mask.shape)
#     points = np.zeros(pred_mask.shape)

#     max_id = 0
#     for i in range(n):        
#         for category_id in np.unique(pred_mask[i]):
#             if category_id == 0:
#                 continue          

#             ind = pred_mask==category_id

#             connected_components = morph.label(ind)

#             uniques = np.unique(connected_components)

#             blobs[ind] = connected_components[ind] + max_id
#             max_id = uniques.max() + max_id

#             n_blobs = (uniques != 0).sum()

#             counts[i, category_id-1] = n_blobs

#             for j in range(1, n_blobs+1):
#                 binmask = connected_components == j
#                 blob_probs = probs[i, category_id] * binmask
#                 y, x = np.unravel_index(blob_probs.squeeze().argmax(), blob_probs.squeeze().shape)

#                 points[i, y, x] = category_id
#                 annList += [mask2ann(binmask, category_id, image_id=-1, 
#                         height=binmask.shape[1], 
#                         width=binmask.shape[2], maskVoid=None, 
#                         score=None, point={"y":y,"x":x, 
#                         "prob":float(blob_probs[blob_probs!=0].max()),
#                         "category_id":int(category_id)})]
                


#     blobs = blobs.astype(int)

#     # Get points

#     return {"blobs":blobs, "annList":annList, "probs":probs,
#             "counts":counts, "points":points,
#             "pointList":mask2pointList(points)["pointList"],
#             "pred_mask":pred_mask,
#             "n_blobs":len(annList)}

# # def get_blobs(probs, return_counts=False):
# #     probs = ms.t2n(probs)
# #     n, n_classes, h, w = probs.shape
  
# #     blobs = np.zeros((n, n_classes-1, h, w))
# #     counts = np.zeros((n, n_classes-1))
    
# #     # Binary case

# #     mask_labels = ms.t2n(probs.argmax(1))

# #     for i in range(n):
# #         for l in np.unique(mask_labels[i]):
# #             if l == 0:
# #                 continue
            
# #             blobs[i,l-1] = morph.label(mask_labels==l)
# #             counts[i, l-1] = (np.unique(blobs[i,l-1]) != 0).sum()

# #     blobs = blobs.astype(int)

# #     if return_counts:
# #         return blobs, counts

# #     return blobs

# def blobs2points(probs, return_pointprobs=False):

#     blobs = probs2blobs(probs)["blobs"].squeeze()
#     probs = probs.squeeze()
#     c, h, w = blobs.shape
#     points = np.zeros((h, w), int)
#     point_probs = np.zeros((h, w))
#     for k, n_blobs in enumerate(blobs.max((1,2)).squeeze()):
#         if n_blobs == 0:
#             continue

#         probs_class = probs[k+1]

        
#         for j in range(1, n_blobs+1):
#             blob_probs = probs_class * (blobs[k] == j)

#             r, c = np.unravel_index(blob_probs.argmax(), blob_probs.shape)

#             points[r, c] = k + 1

#             point_probs[r, c] = probs_class[r,c]

#     if return_pointprobs:
#         return points, point_probs

#     return {"points":points, "point_probs":point_probs}




# ########### Helpers
# def isPointInBBox(p, ann):
#     bbox = maskUtils.toBbox(ann["segmentation"])
#     x,y,w,h = map(int, bbox)   
#     if (p["y"] <= (y+h) and p["y"] >= y and 
#         p["x"] <= (x+w) and p["x"] >= x): 
#         return True

#     return False

