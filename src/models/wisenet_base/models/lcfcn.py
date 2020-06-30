import torch
import torch.nn as nn
import torchvision
import numpy as np
from .. import misc as ms
from .. import ann_utils as au
import torch.nn.functional as F
from . import base_model as bm
from skimage import morphology as morph

class LCFCN_BO(bm.BaseModel):
    def __init__(self, train_set, **model_options):
        super().__init__(train_set, **model_options)

        self.feature_extracter = bm.FeatureExtracter()
        self.blob_head = bm.Upsampler(self.feature_extracter.expansion_rate, 
                                    train_set.n_classes)

    def forward(self, x_input):
        x_8s, x_16s, x_32s = self.feature_extracter.extract_features(x_input)
        blob_mask = self.blob_head.upsample(x_input, x_8s, x_16s, x_32s)

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
        predict_method = "blob_annList"
        n,c,h,w = batch["images"].shape
        
        O = self(batch["images"].cuda())
        probs = F.softmax(O, dim=1)
        blob_dict = au.probs2blobs(probs)

        if predict_method == 'pointList':

            return {"pointList":blob_dict["pointList"],
                    "blobs":blob_dict['blobs'],
                    "probs":blob_dict["probs"]}

        ###
        if predict_method == "blob_annList":
            annList = blob_dict["annList"]
            for ann in annList:
                ann["image_id"] = batch["name"][0]
                ann["score"] = 1.0

            return {"annList":annList}

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
            return {"blobs": np.zeros((h,w), int), 
                    "annList":[]}

        pred_dict = au.pointList2BestObjectness(head_pointList, batch)
        return pred_dict

class LCFCN_Pyramid(LCFCN_BO):
    def __init__(self, train_set, **model_options):
        super().__init__(train_set, **model_options)
        self.feature_extracter = bm.FeatureExtracterPyramid()

from ..models import gam
class LCFCN_Regularized(LCFCN_BO):
    def __init__(self, train_set, **model_options):
        super().__init__(train_set, **model_options)

        backbone = gam.vgg16(pretrained=True)

        self.tmp = list(backbone.features.children());
        for i in range(8) :
            self.tmp.pop();

        # replace relu layers with prelu
        # self.replace_relu_with_prelu();

        self.feature_extracter = nn.Sequential(*self.tmp);

        self.regressor = nn.Linear(in_features=512,  out_features=self.n_classes);

    def replace_relu_with_prelu(self) :
        id_relu = [1,3,6,8,11,13,15,18,20,22];
        for i in id_relu :
            self.tmp[i] = nn.PReLU(self.tmp[i-1].out_channels);

    def forward(self, x):
        n,c,h,w = x.shape
        x_feat = self.feature_extracter(x)
        # O = self.blob_head.upsample(x, x_8s, x_16s, x_32s)
        x_feat = x_feat.abs();
        input_size = (x_feat.size(2), x_feat.size(3))

        count = F.avg_pool2d(x_feat, kernel_size=input_size, stride=input_size)
        count = count.view(count.size(0), -1);
        count = self.regressor(count);

        shape = [self.n_classes] + list(x_feat.shape[-2:])

        x_feat = x_feat.view(x_feat.size(0), x_feat.size(1), -1);
        x_feat = x_feat.mul(self.regressor.weight.data.unsqueeze(2));

        x_feat = x_feat.sum(1);
        x_feat = x_feat.abs();
        max_, _ = x_feat.data.max(1);
        x_feat.data.div_(max_.unsqueeze(1).expand_as(x_feat));
        x_feat = x_feat.reshape(shape)
        x_feat = F.interpolate(x_feat[None], size=(h,w), mode="bilinear")
        return {"count":count, "cam":x_feat}

    def predict(self, batch, predict_method="blobs", proposal_type="sharp"):
        self.sanity_checks(batch)
        self.eval()

        n,c,h,w = batch["images"].shape
        
        O = self(batch["images"].cuda())["cam"]
        probs = F.softmax(O, dim=1)
        blob_dict = au.probs2blobs(probs)

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


class LCFCN_Dilated(LCFCN_BO):
    def __init__(self, train_set, **model_options):
        super().__init__(train_set, **model_options)
        self.feature_extracter = bm.FeatureExtracterDilated()
        # self.feature_extracter = bm.FeatureExtracterPyramid()    

class Res50FCN(bm.BaseModel):
    
    def __init__(self, train_set):
        super().__init__(train_set)
        
        num_classes = train_set.n_classes
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet50_32s = torchvision.models.resnet50(pretrained=True)
        
        resnet_block_expansion_rate = resnet50_32s.layer1[0].expansion
        
        # Create a linear layer -- we don't need logits in this case
        resnet50_32s.fc = nn.Sequential()
        
        self.resnet50_32s = resnet50_32s
        
        self.score_32s = nn.Conv2d(512 *  resnet_block_expansion_rate,
                                   num_classes,
                                   kernel_size=1)
        
        self.score_16s = nn.Conv2d(256 *  resnet_block_expansion_rate,
                                   num_classes,
                                   kernel_size=1)
        
        self.score_8s = nn.Conv2d(128 *  resnet_block_expansion_rate,
                                   num_classes,
                                   kernel_size=1)

        # # FREEZE BATCH NORMS
        
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad = False
                m.bias.requires_grad = False

        
 
    def forward(self, x):

        self.resnet50_32s.eval()
        input_spatial_dim = x.size()[2:]
        
        x = self.resnet50_32s.conv1(x)
        x = self.resnet50_32s.bn1(x)
        x = self.resnet50_32s.relu(x)
        x = self.resnet50_32s.maxpool(x)

        x = self.resnet50_32s.layer1(x)
        
        x = self.resnet50_32s.layer2(x)
        logits_8s = self.score_8s(x)
        
        x = self.resnet50_32s.layer3(x)
        logits_16s = self.score_16s(x)
        
        x = self.resnet50_32s.layer4(x)
        logits_32s = self.score_32s(x)
        
        logits_16s_spatial_dim = logits_16s.size()[2:]
        logits_8s_spatial_dim = logits_8s.size()[2:]
        
        logits_16s += nn.functional.upsample(logits_32s,
                                        size=logits_16s_spatial_dim,
                                        mode="bilinear",
                                        align_corners=True)
        
        logits_8s += nn.functional.upsample(logits_16s,
                                        size=logits_8s_spatial_dim,
                                        mode="bilinear",
                                        align_corners=True)
        
        logits_upsampled = nn.functional.upsample(logits_8s,
                                       size=input_spatial_dim,
                                        mode="bilinear",
                                        align_corners=True)
        
        return logits_upsampled

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
        self.sanity_checks(batch)
        self.eval()

        n,c,h,w = batch["images"].shape
        
        O = self(batch["images"].cuda())
        probs = F.softmax(O, dim=1)
        blob_dict = au.probs2blobs(probs)

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

    def visualize(self, batch):
        pred_dict = self.predict(batch, "blobs")
        ms.images(batch["images"], pred_dict["blobs"].astype(int), denorm=1)


# class LCFCN_BO(bm.BaseModel):
#     def __init__(self, train_set, **model_options):
#         super().__init__(train_set, **model_options)

#         self.feature_extracter = bm.FeatureExtracter()
#         self.blob_head = bm.Upsampler(self.feature_extracter.expansion_rate, train_set.n_classes)
        
#         # self.pointDict = au.load_LCFCNPoints({"dataset_name":type(train_set).__name__})


#     def forward(self, x_input):
#         x_8s, x_16s, x_32s = self.feature_extracter.extract_features(x_input)
#         blob_mask = self.blob_head.upsample(x_input, x_8s, x_16s, x_32s)
        
#         return blob_mask

class LCFCN(bm.BaseModel):
    def __init__(self, train_set, **model_options):
        super().__init__(train_set, **model_options)

        self.feature_extracter = bm.FeatureExtracter()
        self.blob_head = bm.Upsampler(self.feature_extracter.expansion_rate, train_set.n_classes)
        
        # self.pointDict = au.load_LCFCNPoints({"dataset_name":type(train_set).__name__})


    def forward(self, x_input):
        x_8s, x_16s, x_32s = self.feature_extracter.extract_features(x_input)
        blob_mask = self.blob_head.upsample(x_input, x_8s, x_16s, x_32s)
        

        return blob_mask


 

class LCFCN_Strong(bm.BaseModel):
    def __init__(self, train_set, **model_options):
        super().__init__(train_set, **model_options)

        self.feature_extracter = bm.FeatureExtracterStrong()
        self.blob_head = bm.UpsamplerStrong(self.feature_extracter.expansion_rate, 21)
        
        # self.pointDict = au.load_LCFCNPoints({"dataset_name":type(train_set).__name__})


    def forward(self, x_input):
        x_8s, x_16s, x_32s = self.feature_extracter.extract_features(x_input)
        blob_mask = self.blob_head.upsample(x_input, x_8s, x_16s, x_32s)
        

        return blob_mask


class LCFCN_BO_Expanded(bm.BaseModel):
    
    def __init__(self, train_set, **model_options):
        super().__init__(train_set, **model_options)
        
        num_classes = train_set.n_classes
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet50_32s = torchvision.models.resnet50(pretrained=True)
        
        resnet_block_expansion_rate = resnet50_32s.layer1[0].expansion
        
        # Create a linear layer -- we don't need logits in this case
        resnet50_32s.fc = nn.Sequential()
        
        self.resnet50_32s = resnet50_32s
        
        self.score_32s = nn.Conv2d(512 *  resnet_block_expansion_rate,
                                   num_classes,
                                   kernel_size=1)
        
        self.score_16s = nn.Conv2d(256 *  resnet_block_expansion_rate,
                                   num_classes,
                                   kernel_size=1)
        
        self.score_8s = nn.Conv2d(128 *  resnet_block_expansion_rate,
                                   num_classes,
                                   kernel_size=1)

        # # FREEZE BATCH NORMS
        
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad = False
                m.bias.requires_grad = False

        
 
    def forward(self, x):

        self.resnet50_32s.eval()
        input_spatial_dim = x.size()[2:]
        
        x = self.resnet50_32s.conv1(x)
        x = self.resnet50_32s.bn1(x)
        x = self.resnet50_32s.relu(x)
        x = self.resnet50_32s.maxpool(x)

        x = self.resnet50_32s.layer1(x)
        
        x = self.resnet50_32s.layer2(x)
        logits_8s = self.score_8s(x)
        
        x = self.resnet50_32s.layer3(x)
        logits_16s = self.score_16s(x)
        
        x = self.resnet50_32s.layer4(x)
        logits_32s = self.score_32s(x)
        
        logits_16s_spatial_dim = logits_16s.size()[2:]
        logits_8s_spatial_dim = logits_8s.size()[2:]
        
        logits_16s += nn.functional.upsample(logits_32s,
                                        size=logits_16s_spatial_dim,
                                        mode="bilinear",
                                        align_corners=True)
        
        logits_8s += nn.functional.upsample(logits_16s,
                                        size=logits_8s_spatial_dim,
                                        mode="bilinear",
                                        align_corners=True)
        
        logits_upsampled = nn.functional.upsample(logits_8s,
                                       size=input_spatial_dim,
                                        mode="bilinear",
                                        align_corners=True)
        
        return logits_upsampled

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
