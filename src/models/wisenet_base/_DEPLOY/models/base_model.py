import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch.autograd import Variable
import numpy as np 
import misc as ms
from skimage import morphology as morph
from torch.autograd import Function
import torchvision
# from core import proposals as prp
# from core import predict_methods as pm 
# from core import score_functions as sf 
import ann_utils as au

def get_batches(n_pixels, size=500000):
    batches = []
    for i in range(0, n_pixels, size):
        batches +=[(i, i+size)]
    return batches

def se_pairwise(fi, fj):
    return (fi - fj).pow(2).sum(1)


def log_pairwise(fi, fj):
    diff = se_pairwise(fi, fj)
    return  (2./ (1+torch.exp(diff))).clamp(min=1e-6, max=(1.-1e-6))

class Upsampler(nn.Module):
    def __init__(self, expansion_rate, n_output):
        super().__init__()
        
        self.score_32s = nn.Conv2d(512 *  expansion_rate,
                                   n_output,
                                   kernel_size=1)
        
        self.score_16s = nn.Conv2d(256 *  expansion_rate,
                                   n_output,
                                   kernel_size=1)
        
        self.score_8s = nn.Conv2d(128 *  expansion_rate,
                                   n_output,
                                   kernel_size=1)
    
    def upsample(self, x_input, x_8s, x_16s, x_32s):
        input_spatial_dim = x_input.size()[2:]
        
        logits_8s = self.score_8s(x_8s)
        logits_16s = self.score_16s(x_16s)
        logits_32s = self.score_32s(x_32s)
        
        logits_16s_spatial_dim = logits_16s.size()[2:]
        logits_8s_spatial_dim = logits_8s.size()[2:]
                
        logits_16s += nn.functional.interpolate(logits_32s,
                                        size=logits_16s_spatial_dim,
                                        mode="bilinear",
                                        align_corners=True)
        
        logits_8s += nn.functional.interpolate(logits_16s,
                                        size=logits_8s_spatial_dim,
                                        mode="bilinear",
                                        align_corners=True)
        
        logits_upsampled = nn.functional.interpolate(logits_8s,
                                       size=input_spatial_dim,
                                        mode="bilinear",
                                        align_corners=True)
        return logits_upsampled

class BaseModel(nn.Module):
    def __init__(self, train_set, **model_options):
        super().__init__()
        self.options = model_options
        # self.predict_dict = ms.get_functions(pm)

        if hasattr(train_set, "n_classes"):
            self.n_classes = train_set.n_classes
        else:
            self.n_classes = train_set["n_classes"]  

        if hasattr(train_set, "ignore_index"):
            self.ignore_index = train_set.ignore_index
        else:
            self.ignore_index = -100

        self.blob_mode = None
        self.trained_batch_names = set()

    def sanity_checks(self, batch):
        if batch["split"][0] != "train":
            assert batch["name"][0] not in self.trained_batch_names 
        
    @torch.no_grad()
    def predict(self, batch, predict_method="probs"):
        self.sanity_checks(batch)
        self.eval()
        # ms.reload(pm)
        # self.predict_dict = ms.get_functions(pm)
        if predict_method == "counts":
            probs = F.softmax(self(batch["images"].cuda()),dim=1).data
            blob_dict = au.probs2blobs(probs)

            return {"counts":blob_dict["counts"]}

        elif predict_method == "probs":
            probs = F.softmax(self(batch["images"].cuda()),dim=1).data
            return {"probs":probs}

        elif predict_method == "points":
            probs = F.softmax(self(batch["images"].cuda()),dim=1).data
            blob_dict = au.probs2blobs(probs)

            return {"points":blob_dict["points"], 
                    "pointList":blob_dict["pointList"],
                    "probs":probs}
            

        elif predict_method == "blobs":
            probs = F.softmax(self(batch["images"].cuda()),dim=1).data
            blob_dict = au.probs2blobs(probs)
            
            return blob_dict

        else:
            print("Used predict method {}".format(predict_method))
            return self.predict_dict[predict_method](self, batch)

    @torch.no_grad()
    def get_embedding_blobs(self, O, fg_bg_seeds):
        n, c, h, w = O.shape
        # seeds = torch.cat([fg_seeds, bg_seeds], 2)[:,:,None]
        fA = O.view(1,c,-1)
        fS = O[:,:, fg_bg_seeds["yList"], fg_bg_seeds["xList"]]

        n_pixels = h*w
        blobs = torch.zeros(h*w)

        n_seeds =  fS.shape[-1]

        maximum = 5000000
        n_loops = int(np.ceil((n_pixels * n_seeds) / maximum))
        
        for (s,e) in get_batches(n_pixels, size=n_pixels//n_loops):
            # s,e = map(int, (s,e))
            diff = log_pairwise(fS[:,:,None], fA[:,:,s:e,None]) 
            blobs[s:e] = diff.max(2)[1] + 1 
        
        bg_min_index = np.where(np.array(fg_bg_seeds["categoryList"])==0)[0].min()
        assert len(fg_bg_seeds["yList"])//2 == bg_min_index
        blobs[blobs > int(bg_min_index)] = 0
        blobs = blobs.squeeze().reshape(h,w).long()

        categoryDict = {}
        for i, category_id in enumerate(fg_bg_seeds["categoryList"]):
            if category_id == 0:
                 continue

            categoryDict[i+1] = category_id 

        return {"blobs":ms.t2n(blobs), "categoryDict":categoryDict}






class FeatureExtracter(nn.Module):
    def __init__(self):
        super().__init__()
        
        resnet50_32s = torchvision.models.resnet50(pretrained=True)
        resnet50_32s.fc = nn.Sequential()
        self.resnet50_32s = resnet50_32s
        self.expansion_rate = resnet50_32s.layer1[0].expansion

        # # FREEZE BATCH NORMS
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    
    def extract_features(self, x_input):
        self.resnet50_32s.eval()
        x = self.resnet50_32s.conv1(x_input)
        x = self.resnet50_32s.bn1(x)
        x = self.resnet50_32s.relu(x)
        x = self.resnet50_32s.maxpool(x)

        x = self.resnet50_32s.layer1(x)
        
        x_8s = self.resnet50_32s.layer2(x)
        x_16s = self.resnet50_32s.layer3(x_8s)
        x_32s = self.resnet50_32s.layer4(x_16s)

        return x_8s, x_16s, x_32s