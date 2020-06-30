import os

import numpy as np
import torch
from PIL import Image
from torch.utils import data
from .. import misc as ms
from torchvision import transforms
from scipy.ndimage.morphology import distance_transform_edt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
from pycocotools import mask as maskUtils
from skimage.transform import resize

def batch2annList(batch):
    annList = []
    image_id = int(batch["name"][0].replace("_",""))
    #image_id = batch["image_id"][0]
    height, width = batch["images"].shape[-2:]

    maskObjects = batch["maskObjects"]
    maskClasses = batch["maskClasses"]
    n_objects = maskObjects[maskObjects!=255].max()
    id = 1
    for obj_id in range(1, n_objects+1):
        if obj_id == 0:
            continue

        binmask = (maskObjects == obj_id)

        segmentation = maskUtils.encode(np.asfortranarray(ms.t2n(binmask).squeeze())) 
        
        segmentation["counts"] = segmentation["counts"].decode("utf-8")
        uniques = (binmask.long()*maskClasses).unique()
        uniques = uniques[uniques!=0]
        assert len(uniques) == 1

        category_id = uniques[0].item()
        
        annList += [{"segmentation":segmentation,
                      "iscrowd":0,
                      # "bbox":maskUtils.toBbox(segmentation).tolist(),
                      "area":int(maskUtils.area(segmentation)),
                     "id":id,
                     "height":height,
                     "width":width,
                     "image_id":image_id,
                     "category_id":category_id}]
        id += 1

    return annList

class BaseDataset(data.Dataset):
    def __init__(self):
        dataset_name = type(self).__name__
        base = "/mnt/projects/counting/Saves/main/"

        if "Pascal" in dataset_name:
            self.lcfcn_path = base + "dataset:Pascal2007_model:Res50FCN_metric:mRMSE_loss:water_loss_B_config:basic/"
        
        elif "CityScapes" in dataset_name:
            self.lcfcn_path = base + "dataset:CityScapes_model:Res50FCN_metric:mRMSE_loss:water_loss_B_config:basic/"
        
        elif "CocoDetection2014" in dataset_name:
            self.lcfcn_path = base + "dataset:CocoDetection2014_model:Res50FCN_metric:mRMSE_loss:water_loss_B_config:sample3000/"

        elif "Kitti" in dataset_name:
            self.lcfcn_path = base + "dataset:Kitti_model:Res50FCN_metric:mRMSE_loss:water_loss_B_config:basic/"
            self.proposals_path = "/mnt/datasets/public/issam/kitti/ProposalsSharp/"

        elif "Plants" in dataset_name:
            self.lcfcn_path = base + "dataset:Plants_model:Res50FCN_metric:mRMSE_loss:water_loss_B_config:basic/"

        else:
            raise

        # fname = base + "lcfcn_points/{}.pkl".format(dataset_name)
        
        # if os.path.exists(fname):
        #     history = ms.load_pkl(self.lcfcn_path + "history.pkl")
        #     self.pointDict = ms.load_pkl(fname)

        #     if self.pointDict["best_model"]["epoch"] != history["best_model"]["epoch"]:            
        #         reset = "reset"
        # else:
        #     if dataset_name == "PascalSmall":
        #         fname = base + "lcfcn_points/{}.pkl".format(dataset_name)
        #         self.pointDict = ms.load_pkl(fname.replace("PascalSmall", "Pascal2012"))
        #     elif dataset_name == "PascalOriginal":
        #         self.pointDict = ms.load_pkl(fname.replace("PascalOriginal", "Pascal2012"))
        #     else:  
        #         import ipdb; ipdb.set_trace()  # breakpoint 5f76e230 //




    def get_lcfcn_pointList(self, name):
        if self.split == "val":
            try:
                return self.pointDict[name]
            except:
                return []
        else:
            return -1

    def get_sm_propList(self, name):
        propList = SharpProposals({"name":[name+".png"], 
                                   "proposals_path":[self.proposals_path] })
        return propList


from .. import ann_utils as au

class MCGProposals:
    def __init__(self, batch):
        path = "/mnt/datasets/public/issam/VOCdevkit/proposals/MCG_2012/"
        fname = path+"{}.mat".format(batch["name"][0])
        fname_pkl = fname.replace(".mat", ".pkl")

        
        if not os.path.exists(fname_pkl):
            self.proposals = ms.loadmat(fname)

            self.n_annList = self.proposals["scores"].shape[0]
            self.superpixel = self.proposals["superpixels"]
            self.min_score = abs(np.min(self.proposals["scores"]))
            self.max_score = np.max(self.proposals["scores"]+self.min_score)
            annList = []
            for i in range(len(self)):
                print(i, "/", len(self))
                prp = self.proposals["labels"][i][0].ravel()
                proposal_mask = np.zeros(self.superpixel.shape, int)
                proposal_mask[np.isin(self.superpixel, prp)] = 1

                score = self.proposals["scores"][i][0] + self.min_score
                score /= self.max_score

                ann = au.mask2ann(proposal_mask, category_id=1,
                                image_id=batch["name"][0],
                                height=self.superpixel.shape[0],
                                width=self.superpixel.shape[1], 
                                maskVoid=None, score=score)

                annList += [ann]

            ms.save_pkl(fname_pkl, annList)


        self.annList = ms.load_pkl(fname_pkl)
        self.n_annList = len(self.annList)


    def __getitem__(self, i):     
        encoded = self.annList[i]["segmentation"]
        proposal_mask = maskUtils.decode(encoded)
        score = 1
        return {"mask":proposal_mask, 
                "score":score}
        

    def __len__(self):
        return min(1500, self.n_annList)

class SharpProposals:
    def __init__(self, batch):
        # if dataset_name == "pascal":
        self.proposals_path = batch["proposals_path"][0]

        if "SharpProposals_name" in batch:
            batch_name = batch["SharpProposals_name"][0]
        else:
            batch_name = batch["name"][0]
        name_jpg = self.proposals_path + "{}.jpg.json".format(batch_name)
        name_png = self.proposals_path + "{}.json".format(batch_name)
        
        if os.path.exists(name_jpg):
            name = name_jpg
        else:
            name = name_png

            
        _, _, self.h, self.w = batch["images"].shape

        if "resized" in batch and batch["resized"].item() == 1:
            name_resized = self.proposals_path + "{}_{}_{}.json".format(batch["name"][0], 
                                                                        self.h, self.w)
            
            if not os.path.exists(name_resized):
                proposals = ms.load_json(name)
                json_file = loop_and_resize(self.h, self.w, proposals)
                ms.save_json(name_resized, json_file)
        else:
            name_resized = name
        # name_resized = name         
        proposals = ms.load_json(name_resized)
        self.proposals = sorted(proposals, key=lambda x:x["score"], 
                                reverse=True)         

    def __getitem__(self, i):
        encoded = self.proposals[i]["segmentation"]
        proposal_mask = maskUtils.decode(encoded)
        
        return {"mask":proposal_mask, 
                "score":self.proposals[i]["score"]}


    def __len__(self):
        return len(self.proposals)


    def sharpmask2psfcn_proposals(self):
        import ipdb; ipdb.set_trace()  # breakpoint 102ed333 //

        pass

def loop_and_resize(h, w, proposals):
    proposals_resized = []
    n_proposals = len(proposals)
    for i in range(n_proposals):
        print("{}/{}".format(i, n_proposals))
        prop = proposals[i]
        seg = prop["segmentation"]
        proposal_mask = maskUtils.decode(seg)
        # proposal_mask = resize(proposal_mask*255, (h, w), order=0).astype("uint8")

        if not proposal_mask.shape == (h, w):
            proposal_mask = (resize(proposal_mask*255, (h, w), order=0)>0).astype(int)
            seg = maskUtils.encode(np.asfortranarray(proposal_mask).astype("uint8"))
            seg["counts"] = seg["counts"].decode("utf-8")

            prop["segmentation"] = seg 
            proposals_resized += [proposals[i]]

        else:
            proposals_resized += [proposals[i]]
        
    return proposals_resized

