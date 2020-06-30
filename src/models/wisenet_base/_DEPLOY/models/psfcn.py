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




class OneHead(bm.BaseModel):
    def __init__(self, train_set, **model_options):
        super().__init__(train_set, **model_options)

        self.feature_extracter = bm.FeatureExtracter()
        self.embedding_head = bm.Upsampler(self.feature_extracter.expansion_rate, 64)
        
        # self.pointDict = au.load_LCFCNPoints({"dataset_name":type(train_set).__name__})


    def forward(self, x_input):
        x_8s, x_16s, x_32s = self.feature_extracter.extract_features(x_input)
        embedding_mask = self.embedding_head.upsample(x_input, x_8s, x_16s, x_32s)

        return {"embedding_mask":embedding_mask}

    @torch.no_grad()
    def visualize(self, batch, proposal_type="sharp", predict_method="blobs"):
        pred_dict = self.predict(batch, proposal_type=proposal_type, predict_method=predict_method)
        ms.images(batch["images"], pred_dict["blobs"], denorm=1)

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

        propDict = au.pointList2propDict(lcfcn_pointList, batch, proposal_type=proposal_type,
                                             thresh=0.5)
    
        # Segmenter
        O = model(batch["images"].cuda())["embedding_mask"]
        seedList = au.propDict2seedList(propDict)

        fg_bg_seeds = au.CombineSeeds(seedList)
        
        blobs_categoryDict = model.get_embedding_blobs(O, fg_bg_seeds)
        blobs = blobs_categoryDict["blobs"]
        categoryDict = blobs_categoryDict["categoryDict"]
        
        if predict_method == "BestDice":
            blob_dict = au.blobs2BestDice(blobs, categoryDict, 
                                          propDict, batch)
            blobs = blob_dict["blobs"]
            annList = blob_dict["annList"]

        else:
            annList = au.blobs2annList(blobs, categoryDict, batch)

        return {"blobs":blobs, "annList":annList, "counts":counts}




class OneHeadLoc(OneHead):
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

