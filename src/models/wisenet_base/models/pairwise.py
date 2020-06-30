import torch
import torch.nn as nn
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch.autograd import Variable
import numpy as np 
import misc as ms
from skimage import morphology as morph
from torch.autograd import Function
from losses import saliency
from core import blobs_utils as bu
# from core import proposals as prp
from . import base_model as bm
from addons.pycocotools import mask as maskUtils
# from core import score_functions as sf
import ann_utils as au

from losses import helpers

# def single_blob():
#     pass

def apply_trans_probs(blobs_probs, trans_mat_dict, h, w):
    if 1:
        probs = (ms.pad_image(torch.FloatTensor(blobs_probs)).squeeze()).clamp(0, 1)
        probs[0] = probs[0]/10.
        ht, wt = int(trans_mat_dict["h"]), int(trans_mat_dict["w"])
        
        probs_small = F.avg_pool2d(probs[None], 8, 8)
        n_classes = probs_small.shape[1]
        probs_small = probs_small.view(n_classes, -1)
        trans_mat = trans_mat_dict["trans_mat"]

        probs_trans = torch.matmul(probs_small.cuda(), trans_mat)
        probs_trans = probs_trans.view(1, n_classes, ht, wt)
        probs_trans = F.interpolate(probs_trans, (h, w), mode='bilinear')
        _, blobs_trans = torch.max(probs_trans, 1)
        ms.images(blobs_trans)
    import ipdb; ipdb.set_trace()  # breakpoint ebb51bef //

    return blobs_trans

def apply_trans(blobs_sims_tmp, h,w, n_seeds, bg_min_index, trans_mat_dict):
    blobs_sims = blobs_sims_tmp.squeeze().reshape(h,w, n_seeds)
    blobs_sims = blobs_sims.transpose(1,2).transpose(0,1)
    blobs_sims = ms.pad_image(blobs_sims[None]).squeeze()

    ht, wt = int(trans_mat_dict["h"]), int(trans_mat_dict["w"])
    
    blobs_sims = F.avg_pool2d(blobs_sims[None], 8, 8)

    blobs_sims = blobs_sims.view(n_seeds,-1)
    blobs_sims[bg_min_index:] = blobs_sims[bg_min_index:]**2
    blobs_sims = blobs_sims / blobs_sims.sum(0)
    trans_mat = trans_mat_dict["trans_mat"]

    blobs_sims_new = torch.matmul(blobs_sims.cuda(), trans_mat)
    blobs_sims_new = blobs_sims_new.view(1, n_seeds, ht, wt)
    blobs_sims_new = F.interpolate(blobs_sims_new, (h, w), mode='bilinear')
    _, blobs_trans = torch.max(blobs_sims_new, 1)

    return blobs_trans

def apply_trans_binary(blobs_sims_tmp, h,w, n_seeds, bg_min_index, trans_mat_dict, fg_bg_seeds):

    bs = blobs_sims_tmp.squeeze().reshape(h,w, n_seeds)
    bs = bs.transpose(1,2).transpose(0,1)

    blobs = torch.zeros((h, w))
    for i in range(bg_min_index):
        bs_i = bs[i]
        # blobs_sims_bg = bs[j for j in range(bs.shape[0])

        blobs_sims_bg = bs[np.setdiff1d(np.arange(bs.shape[0]), [i])].max(0)[0]
        blobs_sims_bg = blobs_sims_bg**n_seeds

        blobs_sims = torch.cat([blobs_sims_bg[None], bs_i[None]], 0)
        
        blobs_sims = blobs_sims / blobs_sims.sum(0)

        blobs_sims = ms.pad_image(blobs_sims[None]).squeeze()


        ht, wt = int(trans_mat_dict["h"]), int(trans_mat_dict["w"])
        blobs_sims = F.avg_pool2d(blobs_sims[None], 8, 8)
        blobs_sims = blobs_sims.view(2,-1)
        trans_mat = trans_mat_dict["trans_mat"]

        blobs_sims_new = torch.matmul(blobs_sims.cuda(), trans_mat)
        blobs_sims_new = blobs_sims_new.view(1, 2, ht, wt)
        blobs_sims_new = F.interpolate(blobs_sims_new, (h, w), mode='bilinear')
        _, blobs_trans = torch.max(blobs_sims_new, 1)
        # ms.images(torch.max(torch.cat([(1 - blobs_sims)[None], blobs_sims[None]], 0)))
        y, x = fg_bg_seeds["yList"][i], fg_bg_seeds["xList"][i]

        labeled = morph.label(ms.t2n(blobs_trans))
        kk = labeled[0, y, x]

        blobs_trans[torch.ByteTensor((labeled!=kk).astype("uint8"))] = 0
        blobs[(blobs_trans==1).squeeze()] = i + 1
        # ms.images(blobs.long())
        

    return blobs.long().squeeze()

@torch.no_grad()
def get_embedding_blobs(self, O, fg_bg_seeds, trans_mat_dict=None, blobs_probs=None):
    n, c, h, w = O.shape
    # seeds = torch.cat([fg_seeds, bg_seeds], 2)[:,:,None]
    fA = O.view(1,c,-1)
    fS = O[:,:, fg_bg_seeds["yList"], fg_bg_seeds["xList"]]

    n_pixels = h*w
    blobs = torch.zeros(h*w)
    
    n_seeds =  fS.shape[-1]

    maximum = 5000000
    n_loops = int(np.ceil((n_pixels * n_seeds) / maximum))
    if trans_mat_dict:
        blobs_sims_tmp =  torch.zeros(h*w, n_seeds)




    for (s,e) in get_batches(n_pixels, size=n_pixels//n_loops):
        # s,e = map(int, (s,e))
        sim = self.similarity_function(fS[:,:,None], fA[:,:,s:e,None]) 
        blobs[s:e] = sim.max(2)[1]
        if trans_mat_dict:
            blobs_sims_tmp[s:e] = sim

    # Filter output
    bg_min_index = np.where(np.array(fg_bg_seeds["categoryList"])==0)[0].min()

    blobs = blobs.squeeze().reshape(h,w).long()
    blobs = threshold(blobs + 1, fg_bg_seeds)
    blobs = ms.t2n(blobs)
    # ms.images(blobs)
    # ms.images(blobs_trans)
    # blobs_sims = blobs_sims.squeeze().reshape(h,w, n_seeds)
    #ms.images(torch.max(torch.FloatTensor(blobs_probs), 1)[1])
    #ms.images(apply_trans_probs(blobs_probs, trans_mat_dict, h, w))
    if trans_mat_dict is not None:
        blobs_trans = apply_trans_binary(blobs_sims_tmp, h,w, n_seeds, bg_min_index, trans_mat_dict, fg_bg_seeds)
    # if trans_mat_dict is not None:
    #     blobs_trans = apply_trans(blobs_sims_tmp, h,w, n_seeds, bg_min_index, trans_mat_dict)

    #     blobs_trans = threshold(blobs_trans + 1, fg_bg_seeds)
    #     blobs_trans = blobs_trans.squeeze()
        # ms.images(blobs_trans)


    categoryDict = {}
    for i, category_id in enumerate(fg_bg_seeds["categoryList"]):
        if category_id == 0:
             continue

        categoryDict[i+1] = category_id 

    if trans_mat_dict is None:
        return {"blobs":ms.t2n(blobs), 
            "categoryDict":categoryDict}
    else:
        return {"blobs":ms.t2n(blobs), "blobs_trans":ms.t2n(blobs_trans), 
                "categoryDict":categoryDict}

def threshold(blobs, fg_bg_seeds):
    bg_min_index = np.where(np.array(fg_bg_seeds["categoryList"])==0)[0].min()
    assert len(fg_bg_seeds["yList"])//2 == bg_min_index
    blobs[blobs > int(bg_min_index)] = 0

    return blobs

class OneHead(bm.BaseModel):
    def __init__(self, train_set, **model_options):
        super().__init__(train_set, **model_options)

        loss_name = model_options["main_dict"]["loss_name"]
        if loss_name == "OneHeadL1RBFLoss":
            similarity_function = pairwise_l1_rbf
        elif loss_name in ["OneHeadRBFLoss_multiproposals_noFP",
                    "OneHeadRBFLoss_multiproposals", 
                    "OneHeadRBFLoss", "OneHeadRBFLoss_random",
                    "OneHeadRBFLoss", "OneHeadRBFLoss_noFP", 
                    "OneHeadRBFLoss_withSim","OneHeadRBFLoss_withSim_noFP"]:
            similarity_function = pairwise_rbf
        elif loss_name == "OneHeadSumLoss":
            similarity_function = pairwise_sum
        elif loss_name == "OneHeadL1SumLoss":
            similarity_function = pairwise_l1_sum
        elif loss_name == "OneHeadLoss":
            similarity_function = pairwise_mean
        else:
            similarity_function = pairwise_squared
        
        self.similarity_function = similarity_function
        self.feature_extracter = bm.FeatureExtracter()
        self.embedding_head = bm.Upsampler(self.feature_extracter.expansion_rate, 64)
        
        # self.pointDict = au.load_LCFCNPoints({"dataset_name":type(train_set).__name__})


    def forward(self, x_input):
        x_8s, x_16s, x_32s = self.feature_extracter.extract_features(x_input)
        embedding_mask = self.embedding_head.upsample(x_input, x_8s, x_16s, x_32s)
        ms.assert_no_nans(embedding_mask)
        return {"embedding_mask":embedding_mask}

    @torch.no_grad()
    def visualize(self, batch, proposal_type="sharp", predict_method="blobs"):
        pred_dict = self.predict(batch, proposal_type=proposal_type, predict_method=predict_method)
        ms.images(ms.pretty_vis(batch["images"], pred_dict["annList"]))

    @torch.no_grad()
    def predict(self, batch, proposal_type="sharp", predict_method="blobs"):
        self.sanity_checks(batch)
        model = self
        self.eval()

        n,c,h,w = batch["images"].shape
        model.eval()

        lcfcn_pointList = batch["lcfcn_pointList"]
        counts = np.zeros(model.n_classes-1)
        if len(lcfcn_pointList) == 0:
            return {"blobs": np.zeros((h,w), int), "annList":[], "counts":counts}

        if predict_method == "loc":
            return {"annList":[au.mask2ann(au.pointList2mask(
                                lcfcn_pointList)["mask"],1,1,h,w)]}

        propDict = au.pointList2propDict(lcfcn_pointList, batch, proposal_type=proposal_type,
                                             thresh=0.5)
    
        # Segmenter
        O = model(batch["images"].cuda())["embedding_mask"]
        seedList = propDict2seedList(propDict)

        fg_bg_seeds = CombineSeeds(seedList)
        
        blobs_categoryDict = get_embedding_blobs(model, O, fg_bg_seeds)
        blobs = blobs_categoryDict["blobs"]
        categoryDict = blobs_categoryDict["categoryDict"]

        # blobs, categoryDict, propDict = prototype_predict(model, batch, visualize=False)
        
        if predict_method == "BestDice":
            blob_dict = au.blobs2BestDice(blobs, categoryDict, 
                                          propDict, batch)
            blobs = blob_dict["blobs"]
            annList = blob_dict["annList"]
            

        else:
            annList = au.blobs2annList(blobs, categoryDict, batch)

        return {"blobs":blobs, "annList":annList, "counts":counts}





class TwoHeads_Base(OneHead):
    def __init__(self, train_set, **model_options):
        super().__init__(train_set, **model_options)
        self.aff_model = None
        
    def forward(self, x_input):

        x_8s, x_16s, x_32s = self.feature_extracter.extract_features(x_input)
        # blob_mask = self.blob_head.upsample(x_input, x_8s, x_16s, x_32s)
        embedding_mask = self.embedding_head.upsample(x_input, x_8s, x_16s, x_32s)
        points = self.PRM.get_points({"images":x_input})
        pointList = au.mask2pointList(points[None])["pointList"]

        return {"embedding_mask":embedding_mask, 
                "pointList":pointList}

    @torch.no_grad()
    def predict(self, batch, proposal_type="sharp", predict_method="blobs", sim_func=None, use_trans=0):
        self.sanity_checks(batch)

        # if self.aff_model is None:
        #     from models.affinitynet import AffinityHead
        #     self.aff_model =  AffinityHead().cuda()
        #     self.aff_model.load_state_dict(torch.load("/mnt/datasets/public/issam/res38_aff.pth"))

            # batch["images"] = ms.pad_image(batch["images"])
            

        model = self
        model.eval()
        self.eval()

        n,c,h,w = batch["images"].shape

        if self.aff_model is not None:
            trans_mat = self.aff_model.get_trans_mat(batch)

        O_dict = model(batch["images"].cuda())
        try:
          lcfcn_pointList = O_dict["pointList"]["pointList"]
        except:
            lcfcn_pointList = O_dict["pointList"]
        counts = np.zeros(model.n_classes-1)
        if len(lcfcn_pointList) == 0:
            return {"blobs": np.zeros((h,w), int), "annList":[], "counts":counts}

        if predict_method == "loc":
            return {"annList":[au.mask2ann(au.pointList2mask(
                                lcfcn_pointList)["mask"],1,1,h,w)]}
        if "single_point" in batch and batch["single_point"].item() == 1:
            propDict = au.pointList2propDict(lcfcn_pointList, batch, single_point=True, proposal_type=proposal_type,
                                                 thresh=0.5)
        else:
            propDict = au.pointList2propDict(lcfcn_pointList, batch, proposal_type=proposal_type,
                                                 thresh=0.5)
    
        # Segmenter
        O = O_dict["embedding_mask"]

        seedList = propDict2seedList(propDict)

        fg_bg_seeds = CombineSeeds(seedList)

        if not use_trans:
            trans_mat = None
        

        blobs_categoryDict = get_embedding_blobs(model, O, fg_bg_seeds)

        if use_trans:
            blobs = blobs_categoryDict["blobs_trans"]
        else:
            blobs = blobs_categoryDict["blobs"]

        categoryDict = blobs_categoryDict["categoryDict"]
        # blobs, categoryDict, propDict = prototype_predict(model, batch, visualize=False)

        if predict_method == "BestDice":
            blob_dict = au.blobs2BestDice(blobs, categoryDict, 
                                          propDict, batch, 
                                          sim_func=sim_func)
            blobs = blob_dict["blobs"]
            annList = blob_dict["annList"]
            

        else:
            annList = au.blobs2annList(blobs, categoryDict, batch)

        return {"blobs":blobs, "annList":annList, "counts":counts}

def get_batches(n_pixels, size=500000):
    batches = []
    for i in range(0, n_pixels, size):
        batches +=[(i, i+size)]
    return batches

def pairwise_l1_sum(fi, fj):
    diff = (fi - fj).abs().sum(1)
    return  (2./ (1+torch.exp(diff))).clamp(min=1e-6, max=(1.-1e-6))

def pairwise_rbf(fi, fj):
    diff = (fi - fj).pow(2).sum(1)

    # print(diff.max().item())
    return  torch.exp(-diff/64).clamp(min=1e-6, max=(1.-1e-6))

def pairwise_l1_rbf(fi, fj):
    diff = (fi - fj).abs().sum(1)
    # print(diff.max().item())
    return  torch.exp(-diff/64).clamp(min=1e-6, max=(1.-1e-6))


def pairwise_sum(fi, fj):
    diff = (fi - fj).pow(2).sum(1).clamp(min=0, max=50)
    # print(diff.max().item())
    return  (2./ (1+torch.exp(diff))).clamp(min=1e-6, max=(1.-1e-6))
    

def pairwise_mean(fi, fj):
    diff = (fi - fj).pow(2).mean(1)

    return  (2./ (1+torch.exp(diff))).clamp(min=1e-6, max=(1.-1e-6))


def pairwise_squared(fi, fj):
    diff = (fi - fj).pow(2).sum(1)
    # print(diff.max().item())
    return  -diff


class OneHead_32(OneHead):
    def __init__(self, train_set, **model_options):
        super().__init__(train_set, **model_options)
        self.embedding_head = bm.Upsampler(self.feature_extracter.expansion_rate, 32)

class OneHead_128(OneHead):
    def __init__(self, train_set, **model_options):
        super().__init__(train_set, **model_options)
        self.embedding_head = bm.Upsampler(self.feature_extracter.expansion_rate, 128)

class OneHead_256(OneHead):
    def __init__(self, train_set, **model_options):
        super().__init__(train_set, **model_options)
        self.embedding_head = bm.Upsampler(self.feature_extracter.expansion_rate, 256)

class OneHead_Pyramid(OneHead):
    def __init__(self, train_set, **model_options):
        super().__init__(train_set, **model_options)
        self.feature_extracter = bm.FeatureExtracterPyramid()

class OneHead_Dilated(OneHead):
    def __init__(self, train_set, **model_options):
        super().__init__(train_set, **model_options)
        self.feature_extracter = bm.FeatureExtracterDilated()

class OneHeadLoc(OneHead_32):
    def __init__(self, train_set, **model_options):
        super().__init__(train_set, **model_options)


    def forward(self, x_input):
        x_8s, x_16s, x_32s = self.feature_extracter.extract_features(x_input)
        O = self.embedding_head.upsample(x_input, x_8s, x_16s, x_32s)
        n,c,h,w = O.shape
        
        G = np.mgrid[:h,:w].astype(float)
        G[0] = G[0] / float(h)
        G[1] = G[1] / float(w)
        G = torch.FloatTensor(G).cuda()
        O = torch.cat([O,G[None]], dim=1)
        
        return {"embedding_mask":O}




    # @torch.no_grad()
    # def get_twoHeads_annList(model, batch, predict_method=None, proposal_type="sharp"):
    #     n,c,h,w = batch["images"].shape
    #     model.eval()

    #     O_dict = model(batch["images"].cuda())
    #     O = O_dict["blob_mask"]

    #     probs = F.softmax(O, dim=1)
    #     blob_dict = au.probs2blobs(probs)

    #     if predict_method == 'original':
    #         return {"blobs":blob_dict['blobs'], 
    #                 "probs":blob_dict['probs'], 
    #                 "annList":blob_dict['annList'], 
    #                 "counts":blob_dict['counts']}

    #     head_pointList = blob_dict["pointList"]
    #     counts = np.zeros(model.n_classes-1)
    #     if len(head_pointList) == 0:
    #         return {"blobs": np.zeros((h,w), int), "annList":[], "probs":probs,
    #                 "counts":counts}

    #     propDict = au.pointList2propDict(head_pointList, batch, thresh=0.5,
    #         proposal_type=proposal_type)
        
    #     # Segmenter
    #     O = O_dict["embedding_mask"]
    #     seedList = au.propDict2seedList(propDict)

    #     fg_bg_seeds = au.CombineSeeds(seedList)

    #     blobs_categoryDict = model.get_embedding_blobs(O, fg_bg_seeds)
    #     blobs = blobs_categoryDict["blobs"]
    #     categoryDict = blobs_categoryDict["categoryDict"]
        
    #     if predict_method == "BestDice":
    #         blob_dict = au.blobs2BestDice(blobs, categoryDict, 
    #                     propDict, batch)
    #         blobs = blob_dict["blobs"]
    #         annList = blob_dict["annList"]
    #     else:
    #         annList = au.blobs2annList(blobs, categoryDict, batch)

    #     return {"blobs":blobs, "probs":probs, "annList":annList, "counts":counts}



# class TwoHeads(bm.BaseModel):
#     def __init__(self, train_set, **model_options):
#         super().__init__(train_set, **model_options)

#         self.feature_extracter = bm.FeatureExtracter()
#         self.blob_head = bm.Upsampler(self.feature_extracter.expansion_rate, 
#                                     train_set.n_classes)
#         self.embedding_head = bm.Upsampler(self.feature_extracter.expansion_rate, 
#                                   64)

#     def forward(self, x_input):
#         x_8s, x_16s, x_32s = self.feature_extracter.extract_features(x_input)
#         blob_mask = self.blob_head.upsample(x_input, x_8s, x_16s, x_32s)
#         embedding_mask = self.embedding_head.upsample(x_input, x_8s, x_16s, x_32s)

#         return {"embedding_mask":embedding_mask, 
#                 "blob_mask":blob_mask}

#     def predict(self, batch, predict_method="blobs", proposal_type="sharp"):
#         self.sanity_checks(batch)
#         self.eval()
#         blob_dict = self.get_twoHeads_annList(batch, predict_method=predict_method,
#             proposal_type=proposal_type)

#         return {"blobs":blob_dict['blobs'], 
#                 "probs":blob_dict["probs"],
#                 "annList":blob_dict["annList"],
#                 "counts":blob_dict["counts"]}


#     @torch.no_grad()
#     def get_twoHeads_annList(model, batch, predict_method=None, proposal_type="sharp"):
#         n,c,h,w = batch["images"].shape
#         model.eval()

#         O_dict = model(batch["images"].cuda())
#         O = O_dict["blob_mask"]

#         probs = F.softmax(O, dim=1)
#         blob_dict = au.probs2blobs(probs)

#         if predict_method == 'original':
#             return {"blobs":blob_dict['blobs'], 
#                     "probs":blob_dict['probs'], 
#                     "annList":blob_dict['annList'], 
#                     "counts":blob_dict['counts']}

#         head_pointList = blob_dict["pointList"]
#         counts = np.zeros(model.n_classes-1)
#         if len(head_pointList) == 0:
#             return {"blobs": np.zeros((h,w), int), "annList":[], "probs":probs,
#                     "counts":counts}

#         propDict = au.pointList2propDict(head_pointList, batch, thresh=0.5,
#             proposal_type=proposal_type)
        
#         # Segmenter
#         O = O_dict["embedding_mask"]
#         seedList = propDict2seedList(propDict)

#         fg_bg_seeds = CombineSeeds(seedList)

#         blobs_categoryDict = get_embedding_blobs(model, O, fg_bg_seeds)
#         blobs = blobs_categoryDict["blobs"]
#         categoryDict = blobs_categoryDict["categoryDict"]
        
#         if predict_method == "BestDice":
#             blob_dict = au.blobs2BestDice(blobs, categoryDict, 
#                         propDict, batch)
#             blobs = blob_dict["blobs"]
#             annList = blob_dict["annList"]
#         else:
#             annList = au.blobs2annList(blobs, categoryDict, batch)

#         return {"blobs":blobs, "probs":probs, "annList":annList, "counts":counts}

from models import iprm
from models import lcfcn

class TwoHeads_mIoU(OneHead):
    def __init__(self, train_set, **model_options):
        super().__init__(train_set, **model_options)
        
    def forward(self, x_input):
        x_8s, x_16s, x_32s = self.feature_extracter.extract_features(x_input)
        # blob_mask = self.blob_head.upsample(x_input, x_8s, x_16s, x_32s)
        embedding_mask = self.embedding_head.upsample(x_input, x_8s, x_16s, x_32s)


        return {"embedding_mask":embedding_mask}

    @torch.no_grad()
    def predict(self, batch, proposal_type="sharp", predict_method="blobs"):

        self.sanity_checks(batch)
        model = self
        self.eval()

        n,c,h,w = batch["images"].shape
        model.eval()
        O_dict = model(batch["images"].cuda())

        lcfcn_pointList = au.mask2pointList(batch["points"])["pointList"]
        
        counts = np.zeros(model.n_classes-1)
        if len(lcfcn_pointList) == 0:
            return {"blobs": np.zeros((h,w), int), "annList":[], "counts":counts}

        if predict_method == "loc":
            return {"annList":[au.mask2ann(au.pointList2mask(
                                lcfcn_pointList)["mask"],1,1,h,w)]}

        propDict = au.pointList2propDict(lcfcn_pointList, batch, proposal_type=proposal_type,
                                             thresh=0.5)
    
        # Segmenter
        O = O_dict["embedding_mask"]
        seedList = propDict2seedList(propDict)

        fg_bg_seeds = CombineSeeds(seedList)
        
        blobs_categoryDict = get_embedding_blobs(model, O, fg_bg_seeds)
        blobs = blobs_categoryDict["blobs"]
        categoryDict = blobs_categoryDict["categoryDict"]

        # blobs, categoryDict, propDict = prototype_predict(model, batch, visualize=False)

        if predict_method == "BestDice":
            blob_dict = au.blobs2BestDice(blobs, categoryDict, 
                                          propDict, batch)
            blobs = blob_dict["blobs"]
            annList = blob_dict["annList"]
            

        else:
            annList = au.blobs2annList(blobs, categoryDict, batch)


        return {"blobs":blobs, "annList":annList, "counts":counts}


class LCFCN_BO_mIoU(OneHead):
    def __init__(self, train_set, **model_options):
        super().__init__(train_set, **model_options)
        
    def forward(self, x_input):
        x_8s, x_16s, x_32s = self.feature_extracter.extract_features(x_input)
        # blob_mask = self.blob_head.upsample(x_input, x_8s, x_16s, x_32s)
        embedding_mask = self.embedding_head.upsample(x_input, x_8s, x_16s, x_32s)


        return {"embedding_mask":embedding_mask}

    @torch.no_grad()
    def predict(self, batch, proposal_type="sharp", predict_method="blobs"):

        self.sanity_checks(batch)
        model = self
        self.eval()

        n,c,h,w = batch["images"].shape
        lcfcn_pointList = au.mask2pointList(batch["points"])["pointList"]
        
        counts = np.zeros(model.n_classes-1)
        if len(lcfcn_pointList) == 0:
            return {"blobs": np.zeros((h,w), int), "annList":[], "counts":counts}

        if predict_method == "loc":
            return {"annList":[au.mask2ann(au.pointList2mask(
                                lcfcn_pointList)["mask"],1,1,h,w)]}

        pred_dict = au.pointList2BestObjectness(lcfcn_pointList, batch)


        return pred_dict




class TwoHeads_PRM(TwoHeads_Base):
    def __init__(self, train_set, **model_options):
        super().__init__(train_set, **model_options)
        path = "/mnt/projects/counting/Saves/main//dataset:PascalOriginal_model:TwoHeads_Pascal_metric:MAE_loss:OneHeadRBFLoss_config:noFlip//State_Dicts/best_model.pth"
        self.load_state_dict(torch.load(path), strict=False)
        # self.PRM.load_state_dict(torch.load(path))
        self.PRM = iprm.PRM(train_set, **model_options)
        path = ("/mnt/projects/counting/Saves/main//"+
        "dataset:PascalOriginal_model:PRM_metric:MAE_loss:PRMLoss_config:noFlip//State_Dicts/best_model.pth")
        self.PRM.load_state_dict(torch.load(path))


    def forward(self, x_input):
        x_8s, x_16s, x_32s = self.feature_extracter.extract_features(x_input)
        # blob_mask = self.blob_head.upsample(x_input, x_8s, x_16s, x_32s)
        embedding_mask = self.embedding_head.upsample(x_input, x_8s, x_16s, x_32s)

        points = self.PRM.get_points({"images":x_input})

        pointList = au.mask2pointList(points[None])["pointList"]

        return {"embedding_mask":embedding_mask, 
                "pointList":pointList}


class TwoHeads_Kitti(TwoHeads_Base):
    def __init__(self, train_set, **model_options):
        super().__init__(train_set, **model_options)
        self.LCFCN = lcfcn.LCFCN_BO(train_set, **model_options)
        
        path = ("/mnt/projects/counting/Saves/main//"+
            "dataset:Kitti_model:LCFCN_BO_metric:MAE_loss:lcfcnLoss_config:noFlip//State_Dicts/best_model.pth")
        self.LCFCN.load_state_dict(torch.load(path))


    def forward(self, x_input):
        x_8s, x_16s, x_32s = self.feature_extracter.extract_features(x_input)
        # blob_mask = self.blob_head.upsample(x_input, x_8s, x_16s, x_32s)
        embedding_mask = self.embedding_head.upsample(x_input, x_8s, x_16s, x_32s)
        pointList = self.LCFCN.predict({"images":x_input}, predict_method="pointList")
     

        return {"embedding_mask":embedding_mask, 
                "pointList":pointList}



class TwoHeads_COCO(TwoHeads_Base):
    def __init__(self, train_set, **model_options):
        super().__init__(train_set, **model_options)
        self.LCFCN = lcfcn.LCFCN_BO(train_set, **model_options)
        
        path = ("/mnt/projects/counting/Saves/main//"+
            "dataset:CocoDetection2014_model:LCFCN_BO_metric:MAE_loss:lcfcnLoss_config:noFlip//State_Dicts/best_model.pth")
        self.LCFCN.load_state_dict(torch.load(path))

    def forward(self, x_input):
        x_8s, x_16s, x_32s = self.feature_extracter.extract_features(x_input)
        # blob_mask = self.blob_head.upsample(x_input, x_8s, x_16s, x_32s)
        embedding_mask = self.embedding_head.upsample(x_input, x_8s, x_16s, x_32s)
        pointList = self.LCFCN.predict({"images":x_input}, predict_method="pointList")
     

        return {"embedding_mask":embedding_mask, 
                "pointList":pointList}




class TwoHeads_CityScapes(TwoHeads_Base):
    def __init__(self, train_set, **model_options):
        super().__init__(train_set, **model_options)
        self.LCFCN = lcfcn.LCFCN_BO(train_set, **model_options)
        path = ("/mnt/projects/counting/Saves/main//"+
            "dataset:CityScapes_model:LCFCN_BO_metric:MAE_loss:lcfcnLoss_config:noFlip//State_Dicts/best_model.pth")
        self.LCFCN.load_state_dict(torch.load(path))

    def forward(self, x_input):
        x_8s, x_16s, x_32s = self.feature_extracter.extract_features(x_input)
        # blob_mask = self.blob_head.upsample(x_input, x_8s, x_16s, x_32s)
        embedding_mask = self.embedding_head.upsample(x_input, x_8s, x_16s, x_32s)
        pointList = self.LCFCN.predict({"images":x_input}, predict_method="pointList")
     

        return {"embedding_mask":embedding_mask, 
                "pointList":pointList}


class TwoHeads_Plants(TwoHeads_Base):
    def __init__(self, train_set, **model_options):
        super().__init__(train_set, **model_options)

        self.LCFCN = lcfcn.LCFCN_BO_Expanded(train_set, **model_options)
        
        path = ('/mnt/projects/counting/Saves/main//'+
            'dataset:Plants_model:Res50FCN_metric:'+
            'mRMSE_loss:water_loss_config:basic//'+
            'State_Dicts/best_model.pth')
        self.LCFCN.load_state_dict(torch.load(path))

    def forward(self, x_input):
        x_8s, x_16s, x_32s = self.feature_extracter.extract_features(x_input)
        # blob_mask = self.blob_head.upsample(x_input, x_8s, x_16s, x_32s)
        embedding_mask = self.embedding_head.upsample(x_input, x_8s, x_16s, x_32s)
        pointList_dict = self.LCFCN.predict({"images":x_input},
                                        predict_method="pointList")
     
        return {"embedding_mask":embedding_mask, 
                "pointList":pointList_dict["pointList"],
                "blobs_probs":pointList_dict["probs"]}



class LCFCN_BO_Plants(TwoHeads_Base):
    def __init__(self, train_set, **model_options):
        super().__init__(train_set, **model_options)

        self.LCFCN = lcfcn.LCFCN_BO_Expanded(train_set, **model_options)
        
        path = ('/mnt/projects/counting/Saves/main//'+
            'dataset:Plants_model:Res50FCN_metric:'+
            'mRMSE_loss:water_loss_config:basic//'+
            'State_Dicts/best_model.pth')
        self.LCFCN.load_state_dict(torch.load(path))

    def forward(self, x_input):
        blob_mask = self.LCFCN(x_input)
        return blob_mask

    def get_blobs(self, p_labels, return_counts=False):
        from skimage import morphology as morph
        p_labels = ms.t2n(p_labels)
        n,h,w = p_labels.shape
      
        blobs = np.zeros((n, self.n_classes-1, h, w))
        counts = np.zeros((n, self.n_classes-1))
        
        # Binary case
        for i in range(n):
            for l in np.unique(p_labels[i]):
                if l == 0:
                    continue
                
                blobs[i,l-1] = morph.label(p_labels==l)
                counts[i, l-1] = (np.unique(blobs[i,l-1]) != 0).sum()

        blobs = blobs.astype(int)

        if return_counts:
            return blobs, counts

        return blobs

    def predict(self, batch, predict_method="blobs", proposal_type="sharp"):
        # self.sanity_checks(batch)
        self.eval()

        n,c,h,w = batch["images"].shape
        
        O = self(batch["images"].cuda())
        probs = F.softmax(O, dim=1)
        blob_dict = au.probs2blobs(probs)

        if predict_method == 'pointList':

            return {"pointList":blob_dict["pointList"],
                    "blobs":blob_dict['blobs'],
                    "probs":blob_dict["probs"]}

        if predict_method == 'blobs_probs':
            blobs = self.get_blobs(O.max(1)[1])
            return blobs, probs

        if predict_method == 'original':
            return {"blobs":blob_dict['blobs'], 
                    "probs":blob_dict['probs'], 
                    "annList":blob_dict['annList'], 
                    "counts":blob_dict['counts']}

        head_pointList = blob_dict["pointList"]


        if len(head_pointList) == 0:
            return {"blobs": np.zeros((h,w), int), "annList":[]}

        pred_dict = au.pointList2BestObjectness(head_pointList, batch)
        return pred_dict

class TwoHeads_Pascal(TwoHeads_Base):
    def __init__(self, train_set, **model_options):
        super().__init__(train_set, **model_options)
        self.LCFCN = lcfcn.LCFCN_BO(train_set, **model_options)
        path = ("/mnt/projects/counting/Saves/main//"+
            "dataset:Pascal2012_model:LCFCN_BO_metric:MAE_loss:lcfcnLoss_config:wtp//State_Dicts/best_model.pth")
        self.LCFCN.load_state_dict(torch.load(path))

    def forward(self, x_input):
        x_8s, x_16s, x_32s = self.feature_extracter.extract_features(x_input)
        # blob_mask = self.blob_head.upsample(x_input, x_8s, x_16s, x_32s)
        embedding_mask = self.embedding_head.upsample(x_input, x_8s, x_16s, x_32s)
        pointList_dict = self.LCFCN.predict({"images":x_input},
                                        predict_method="pointList")
     
        return {"embedding_mask":embedding_mask, 
                "pointList":pointList_dict["pointList"],
                "blobs_probs":pointList_dict["probs"]}



def get_random_indices(mask, n_indices=10):
    mask_ind = np.where(mask.squeeze())
    n_pixels = mask_ind[0].shape[0]
    P_ind = np.random.randint(0, n_pixels, n_indices)
    yList = mask_ind[0][P_ind]
    xList = mask_ind[1][P_ind]

    return {"yList":yList, "xList":xList}

def propDict2seedList(propDict, n_neighbors=100, random_proposal=False):
    seedList = []
    for prop in propDict["propDict"]:
        if len(prop["annList"]) == 0:
            seedList += [{"category_id":[prop["point"]["category_id"]],
                          "yList":[prop["point"]["y"]],   
                          "xList":[prop["point"]["x"]],   
                          "neigh":{"yList":[prop["point"]["y"]],
                                    "xList":[prop["point"]["x"]]}}]

        else:
            if random_proposal:
                i = np.random.randint(0, len(prop["annList"]))
                mask = prop["annList"][i]["mask"]
            else:
                mask = prop["annList"][0]["mask"]
                
            seedList += [{"category_id":[prop["point"]["category_id"]],
                           "yList":[prop["point"]["y"]],   
                          "xList":[prop["point"]["x"]],   
                          "neigh":get_random_indices(mask, n_indices=100)}]

    # Background
    background = propDict["background"]
    if background.sum() == 0:
        y_axis = np.random.randint(0, background.shape[1],100)
        x_axis = np.random.randint(0, background.shape[2],100)
        background[0,y_axis, x_axis] = 1
    bg_seeds = get_random_indices(background, n_indices=len(propDict["propDict"]))
    seedList += [{"category_id":[0]*len(bg_seeds["yList"]),
                    "yList":bg_seeds["yList"].tolist(), 
                  "xList":bg_seeds["xList"].tolist(), 
                  "neigh":get_random_indices(background, n_indices=100)}] 

    return seedList

def CombineSeeds(seedList, ind=None):
    yList = []
    xList = []
    categoryList = []

    if ind is None:
        ind = range(len(seedList))

    for i in ind:
        yList += seedList[i]["yList"]
        xList += seedList[i]["xList"]
        categoryList += seedList[i]["category_id"]

    assert len(categoryList) == len(yList) 
    return {"yList":yList, "xList":xList, "categoryList":categoryList}


class OneHeadStrong(OneHead):
    def __init__(self, train_set, **model_options):
        super().__init__(train_set, **model_options)

        self.feature_extracter = bm.FeatureExtracterStrong()
        self.embedding_head = bm.UpsamplerStrong(self.feature_extracter.expansion_rate, 64)
        
        # self.pointDict = au.load_LCFCNPoints({"dataset_name":type(train_set).__name__})


    def forward(self, x_input):

        x_4s, x_8s, x_16s, x_32s = self.feature_extracter.extract_features(x_input)
        embedding_mask = self.embedding_head.upsample(x_input, x_4s, x_8s, x_16s, x_32s)
        ms.assert_no_nans(embedding_mask)
        return {"embedding_mask":embedding_mask}


# class TwoHeadsStrong(TwoHeads):
#     def __init__(self, train_set, **model_options):
#         super().__init__(train_set, **model_options)

#         self.feature_extracter = bm.FeatureExtracterStrong()
#         self.blob_head = bm.UpsamplerStrong(self.feature_extracter.expansion_rate, 
#                                     train_set.n_classes)
#         self.embedding_head = bm.UpsamplerStrong(self.feature_extracter.expansion_rate, 
#                                   64)
#         # self.pointDict = au.load_LCFCNPoints({"dataset_name":type(train_set).__name__})


#     def forward(self, x_input):

#         x_4s, x_8s, x_16s, x_32s = self.feature_extracter.extract_features(x_input)
#         embedding_mask = self.embedding_head.upsample(x_input, x_4s, x_8s, x_16s, x_32s)
#         blob_mask = self.blob_head.upsample(x_input, x_4s, x_8s, x_16s, x_32s)
#         ms.assert_no_nans(embedding_mask)
#         return {"embedding_mask":embedding_mask, 
#                 "blob_mask":blob_mask}

# class TwoHeadsStrong101(TwoHeads):
#     def __init__(self, train_set, **model_options):
#         super().__init__(train_set, **model_options)

#         self.feature_extracter = bm.FeatureExtracterStrong101()
#         self.blob_head = bm.UpsamplerStrong(self.feature_extracter.expansion_rate, 
#                                     train_set.n_classes)
#         self.embedding_head = bm.UpsamplerStrong(self.feature_extracter.expansion_rate, 
#                                   64)
#         # self.pointDict = au.load_LCFCNPoints({"dataset_name":type(train_set).__name__})


#     def forward(self, x_input):

#         x_4s, x_8s, x_16s, x_32s = self.feature_extracter.extract_features(x_input)
#         embedding_mask = self.embedding_head.upsample(x_input, x_4s, x_8s, x_16s, x_32s)
#         blob_mask = self.blob_head.upsample(x_input, x_4s, x_8s, x_16s, x_32s)
#         ms.assert_no_nans(embedding_mask)
#         return {"embedding_mask":embedding_mask, 
#                 "blob_mask":blob_mask}





class OneHeadProto_original(OneHead):
    def __init__(self, train_set, **model_options):
        super().__init__(train_set, **model_options)

    @torch.no_grad()
    def predict(self, batch, proposal_type="sharp", predict_method="blobs"):
        self.sanity_checks(batch)
        model = self
        self.eval()

        n,c,h,w = batch["images"].shape
        model.eval()

        lcfcn_pointList = batch["lcfcn_pointList"]
        counts = np.zeros(model.n_classes-1)
        if len(lcfcn_pointList) == 0:
            return {"blobs": np.zeros((h,w), int), "annList":[], "counts":counts}


        blobs, categoryDict, propDict = prototype_predict(model, batch, visualize=False)
        
        if predict_method == "BestDice":
            blob_dict = au.blobs2BestDice(blobs, categoryDict, propDict, batch)
            blobs = blob_dict["blobs"]
            annList = blob_dict["annList"]
        else:
            annList = au.blobs2annList(blobs, categoryDict, batch)

        return {"blobs":blobs, "annList":annList, "counts":counts}



class OneHeadProto(OneHeadStrong):
    def __init__(self, train_set, **model_options):
        super().__init__(train_set, **model_options)

    @torch.no_grad()
    def predict(self, batch, proposal_type="sharp", predict_method="blobs"):
        self.sanity_checks(batch)
        model = self
        self.eval()

        n,c,h,w = batch["images"].shape
        model.eval()

        lcfcn_pointList = batch["lcfcn_pointList"]
        counts = np.zeros(model.n_classes-1)
        if len(lcfcn_pointList) == 0:
            return {"blobs": np.zeros((h,w), int), "annList":[], "counts":counts}


        blobs, categoryDict, propDict = prototype_predict(model, batch, visualize=False)
        
        if predict_method == "BestDice":
            blob_dict = au.blobs2BestDice(blobs, categoryDict, propDict, batch)
            blobs = blob_dict["blobs"]
            annList = blob_dict["annList"]
        else:
            annList = au.blobs2annList(blobs, categoryDict, batch)

        return {"blobs":blobs, "annList":annList, "counts":counts}



def prototype_predict(model, batch, pointList=None, visualize=False):
    n,c,h,w = batch["images"].shape

    model.eval()
    O_dict = model(batch["images"].cuda())
    O = O_dict["embedding_mask"]

    if pointList is None:
        lcfcn_pointList = batch["lcfcn_pointList"]
    else:
        lcfcn_pointList = pointList

    base_dict = helpers.metric_base(O, batch, lcfcn_pointList)
    yList = base_dict["yList"]
    xList = base_dict["xList"]
    propDict = base_dict["propDict"]
    # yList = base_dict["yList"]

    bg_dict = helpers.get_bg_dict(base_dict["background"])
    # foreground = distance_transform_cdt(1 - background)
    ###################################
    n,c,h,w = O.shape
    
    fS = O[:, :, yList+bg_dict["yList"], 
                 xList+bg_dict["xList"]]
    # for y, x in zip(yList, xList):
    #     pass
    
    sim =  pairwise_sum(fS[:,:,None], O.view(1,64,-1)[:,:,:,None])


    blobs = sim.max(2)[1] + 1
    blobs[blobs > len(yList)] = 0
    blobs = blobs.squeeze().reshape(h,w).long()

    categoryDict = {}
    for i in range(len(lcfcn_pointList)):
        categoryDict[i+1] = int(lcfcn_pointList[i]["category_id"])


    return ms.t2n(blobs), categoryDict, {"propDict":propDict}

