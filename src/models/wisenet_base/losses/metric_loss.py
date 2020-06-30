# import torch
# import numpy as np
# from addons.pycocotools import mask as maskUtils
# import os
# import json

# def OneHeadLoss_new(model, batch, visualize=False):
#     n,c,h,w = batch["images"].shape

#     model.train()
#     O_dict = model(batch["images"].cuda())
#     O = O_dict["embedding_mask"]

#     base_dict = metric_base(O, batch)
#     points = batch["points"]
#     yList = base_dict["yList"]
#     xList = base_dict["xList"]
#     propDict = base_dict["propDict"]
#     yList = base_dict["yList"]
#     background = base_dict["background"]

#     ###################################
#     n,c,h,w = O.shape
#     fg_seeds = O[:, :, yList, xList]
#     n_seeds = fg_seeds.shape[-1]
#     prop_mask = np.zeros((h, w))

#     loss = torch.tensor(0.).cuda()

#     for i in range(n_seeds):
#         annList = propDict[i]["annList"]

#         if len(annList) == 0:
#             mask = np.zeros(points.squeeze().shape)
#             mask[propDict[i]["point"]["y"], propDict[i]["point"]["x"]] = 1
#         else:
#             mask = annList[0]["mask"]


       
#         mask_ind = np.where(mask)
#         prop_mask[mask!=0] = (i+1)

        
#         f_A = fg_seeds[:,:,[i]]
        
#         # Positive Embeddings
#         n_pixels = mask_ind[0].shape[0]
#         P_ind = np.random.randint(0, n_pixels, 100)
#         yList = mask_ind[0][P_ind]
#         xList = mask_ind[1][P_ind]
#         fg_P = O[:,:,yList, xList]
        
#         ap = - torch.log(log_pairwise(f_A, fg_P)) 
#         loss += ap.mean()

#         # Get Negatives
#         if n_seeds > 1:
#             N_ind = [j for j in range(n_seeds) if j != i]
#             f_N = fg_seeds[:,:,N_ind]
#             an = - torch.log(1. - log_pairwise(f_A, f_N)) 
#             loss += an.mean()

#     # Extract background seeds
#     bg = np.where(background.squeeze())

#     n_pixels = bg[0].shape[0]
#     bg_ind = np.random.randint(0, n_pixels, n_seeds)
#     yList = bg[0][bg_ind]
#     xList = bg[1][bg_ind]
#     f_A = O[:,:,yList, xList]


#     bg_ind = np.random.randint(0, n_pixels, 100)
#     yList = bg[0][bg_ind]
#     xList = bg[1][bg_ind]
#     f_P = O[:,:,yList, xList]


#     # BG seeds towards BG pixels, BG seeds away from FG seeds
#     ap = - torch.log(log_pairwise(f_A[:,:,None], f_P[:,:,:,None])) 
#     an = - torch.log(1. - log_pairwise(f_A[:,:,None], fg_seeds[:,:,:,None])) 

#     loss += ap.mean()
#     loss += an.mean()

#     if batch["dataset"][0] == "cityscapes" or batch["dataset"][0] == "coco2014":        
#         n_max = 6
#     else:
#         n_max = 12

#     if f_A.shape[2] < n_max:
#         with torch.no_grad():

#             diff = log_pairwise(O.view(1,c,-1)[:,:,:,None], 
#                                    torch.cat([fg_seeds, f_A], 2)[:,:,None])    
#             labels = diff.max(2)[1] + 1 
#             labels = labels <= n_seeds
#             labels = labels.squeeze().reshape(h,w)
#             bg = labels.cpu().long()*torch.from_numpy(background)        
#             # ms.images(labels.cpu().long()*torch.from_numpy(background))


#         # Extract false positive pixels
#         bg_ind = np.where(bg.squeeze())
#         n_P = bg_ind[0].shape[0]
#         if n_P != 0:
#             A_ind = np.random.randint(0, n_P, n_seeds)
#             f_P = O[:,:, bg_ind[0][A_ind], bg_ind[1][A_ind]]

#             ap = - torch.log(log_pairwise(f_A[:,:,None], f_P[:,:,:,None])) 
#             an = - torch.log(1. - log_pairwise(f_P[:,:,None], fg_seeds[:,:,:,None])) 

#             # if i < 3:
#             loss += ap.mean()
#             loss += an.mean()


#     return loss / max(n_seeds, 1)


# def se_pairwise(fi, fj):
#     return (fi - fj).pow(2).sum(1)

# def log_pairwise(fi, fj):
#     diff = se_pairwise(fi, fj)
#     return  (2./ (1+torch.exp(diff))).clamp(min=1e-6, max=(1.-1e-6))

# def metric_base(O, batch):
#     n,c,h,w = O.shape

#     points = batch["points"]
#     batch["maskObjects"] = None 
#     batch['maskClasses'] = None
#     batch["maskVoid"] = None

#     pointList = mask2pointList(points)["pointList"]

    
#     if len(pointList) == 0:
#         return None

#     if "single_point" in batch:
#         single_point = True
#     else:
#         single_point = False


#     propDict = pointList2propDict(pointList, batch, 
#                                      single_point=single_point,
#                                      thresh=0.5)
#     background = propDict["background"]

#     propDict = propDict["propDict"]

#     yList = []
#     xList = []
#     for p in pointList:
#         yList += [p["y"]]
#         xList += [p["x"]]

#     return {"xList":xList, "yList":yList, "background":background, "propDict":propDict}


# def pointList2propDict(pointList, batch, single_point=False, thresh=0.5, 
#                         proposal_type="sharp"):

#     proposals = batch2proposals(batch, proposal_type=proposal_type)

#     propDict = []
#     shape = pointList[0]["shape"]
#     foreground = np.zeros(shape, int)


#     idDict= {}
#     annDict = {}
#     for i, p in enumerate(pointList):
#         annDict[i] = []
#         idDict[i] = []

#     n_points = len(annDict)
#     for k in range(len(proposals)):
#         proposal_ann = proposals[k]

#         if not (proposal_ann["score"] > thresh):
#             continue

#         proposal_mask =  proposal_ann["mask"]

#         for i, p in enumerate(pointList):
#             if proposal_mask[p["y"], p["x"]]==0:
#                 continue
            
#             # score = proposal_ann["score"]
           
#             annDict[i] += [proposal_ann]
#             idDict[i] += [k]

#     for i in range(n_points):
#         point_annList = annDict[i]
#         point_idList = idDict[i]
#         p = pointList[i]

#         mask = annList2mask(point_annList)["mask"]

#         if mask is not None:
#             foreground = foreground + mask

#         #foreground[foreground<2]=0
#         propDict += [{"annList":point_annList,"point":p, "idList":point_idList, 
#                       "category_id":int(p["category_id"])}]
#         #########  

#     return {"propDict":propDict,"foreground":foreground, "background":(foreground==0).astype(int)}


# #### MISC 
# def batch2proposals(batch, proposal_type):
#     if proposal_type == "sharp":
#         print("Sharp used")
#         proposals = SharpProposals(batch)
#     else:
#         import ipdb; ipdb.set_trace()  # breakpoint 8e909a15 //


#     return proposals

# class SharpProposals:
#     def __init__(self, batch):
#         # if dataset_name == "pascal":
#         self.proposals_path = batch["proposals_path"][0]

#         if "SharpProposals_name" in batch:
#             batch_name = batch["SharpProposals_name"][0]
#         else:
#             batch_name = batch["name"][0]
#         name_jpg = self.proposals_path + "{}.jpg.json".format(batch_name)
#         name_png = self.proposals_path + "{}.json".format(batch_name)
        
#         if os.path.exists(name_jpg):
#             name = name_jpg
#         else:
#             name = name_png

            
#         _, _, self.h, self.w = batch["images"].shape

#         if "resized" in batch and batch["resized"].item() == 1:
#             name_resized = self.proposals_path + "{}_{}_{}.json".format(batch["name"][0], 
#                                                                         self.h, self.w)
  
#         else:
#             name_resized = name
#         # name_resized = name         
#         proposals = load_json(name_resized)
#         self.proposals = sorted(proposals, key=lambda x:x["score"], 
#                                 reverse=True)         

#     def __getitem__(self, i):
#         encoded = self.proposals[i]["segmentation"]
#         proposal_mask = maskUtils.decode(encoded)
        
#         return {"mask":proposal_mask, 
#                 "score":self.proposals[i]["score"]}


#     def __len__(self):
#         return len(self.proposals)

# def annList2mask(annList):
#     n_anns = len(annList)
#     if n_anns == 0:
#         return {"mask":None}

#     ann = annList[0]
#     try:
#         h, w = ann["mask"].shape
#     except:
#         h, w = ann["height"], ann["width"]
#     mask = np.zeros((h, w), int)

#     for i in range(n_anns):
#         ann = annList[i]

#         if "mask" in ann:
#             ann_mask = ann["mask"]
#         else:
#             ann_mask = maskUtils.decode(ann["segmentation"])

#         assert ann_mask.max() <= 1
#         mask += ann_mask

#     # mask[mask==1] = ann["category_id"]
#     return {"mask":mask}


# def mask2pointList(mask):
#     pointList = []
#     mask = t2n(mask)
#     pointInd = np.where(mask.squeeze())
#     n_points = pointInd[0].size

#     for p in range(n_points):

#         p_y, p_x = pointInd[0][p], pointInd[1][p]
#         point_category = mask[0, p_y,p_x]

#         pointList += [{"y":p_y,"x":p_x, "category_id":int(point_category), 
#                        "shape":mask.shape}]

#     return {"pointList":pointList}


# def t2n(x):
#     if isinstance(x, (int, float)):
#         return x
#     if isinstance(x, torch.autograd.Variable):
#         x = x.cpu().data.numpy()
        
#     if isinstance(x, (torch.cuda.FloatTensor, torch.cuda.IntTensor, torch.cuda.LongTensor, torch.cuda.DoubleTensor )):
#         x = x.cpu().numpy()

#     if isinstance(x, (torch.FloatTensor, torch.IntTensor, torch.LongTensor, torch.DoubleTensor )):
#         x = x.numpy()

#     return x

# def load_json(fname, decode=None):

#     with open(fname, "r") as json_file:
#         d = json.load(json_file)
        
#     return d