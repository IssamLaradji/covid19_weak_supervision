from sklearn.metrics import confusion_matrix

import numpy as np
import misc as ms
##from pycocotools.cocoeval import COCOeval
# from models.helpers import score_functions as sf
from metrics import mAP_eval
#from pycocotools import mask as maskUtils
#from pycocotools.coco import COCO
class ABO:
    def __init__(self, n_classes=20):
        self.iou = []
        self.n_objects = []
        self.n_classes = n_classes

        self.iou = np.zeros(n_classes)
        self.n_objects = np.zeros(n_classes)

    def _average(self, iou, n_objects):

        return np.nanmean(np.divide(iou, n_objects))

    def scoreBatch(self, model, batch): 
        model.eval()

        # Pred
        maskObjects, maskProbs = ms.t2n(model.predict(batch, 
                                        "blobs_probs"))
        maskObjects = maskObjects.squeeze()
        maskProbs = maskProbs.squeeze()
        
        pred_masks, pred_labels, pred_scores = pred2masks(maskObjects, maskProbs)

        # Probs
        maskClasses = ms.t2n(batch["maskClasses"].squeeze())
        maskObjects = ms.t2n(batch["maskObjects"].squeeze())

        gt_masks, gt_labels = gt2mask(maskClasses, maskObjects)
        import ipdb; ipdb.set_trace()  # breakpoint 0fd24f32 //
        
        bestOverlapDict = np.zeros(self.n_classes)
        n_objectDict =  np.zeros(self.n_classes)


        bestOverlap = 0

        gt_labels = gt_labels[0]
        pred_labels = pred_labels[0]
        gt_masks = gt_masks[0]
        pred_masks = pred_masks[0]

        for i in range(len(gt_labels)):
            c = gt_labels[i]

            for j in range(len(pred_labels)):

                if pred_labels[j] != c:
                    continue 

                overlap = sf.dice(pred_masks[j], gt_masks[i], smooth=0.)

                if overlap > bestOverlap:
                    bestOverlap = overlap

                bestOverlapDict[c-1] += bestOverlap
                n_objectDict[c-1] += 1

        # COMPUTE SCORE
        return {"score": self._average(bestOverlapDict,
                                       n_objectDict), 
                "bestOverlapDict":bestOverlapDict,
                "n_objectDict":n_objectDict}

    ### RUNNING AVERAGE ####
    def get_running_average(self):
        return self._average(self.iou, self.n_objects)

    def update_running_average(self, model, batch):

        score_dict = self.scoreBatch(model, batch)
    

        self.iou += score_dict["bestOverlapDict"]
        self.n_objects += score_dict["n_objectDict"]

        
        return score_dict
        

class mAP_base:
    def __init__(self, iou_threshold=0.5):
        self.n_pos = None
        self.score = None
        self.match = None

        self.thresh = iou_threshold

    def _average(self, n_pos, score, match):
        prec, rec = mAP_eval.calc_instance_segmentation_voc_prec_rec(n_pos, score, match)
        ap = mAP_eval.calc_detection_voc_ap(prec, rec)
        return np.nanmean(ap)

    def scoreBatch(self, model, batch, n_pos=None, 
                   map_scores=None, match=None): 
        model.eval()

        # Pred
        maskObjects, maskProbs = ms.t2n(model.predict(batch,
         "blobs_probs",return_annList=False))
        
        maskObjects = maskObjects.squeeze()
        maskProbs = maskProbs.squeeze()
        
        pred_masks, pred_labels, pred_scores = pred2masks(maskObjects, maskProbs)
        
        # Probs
        maskClasses = ms.t2n(batch["maskClasses"].squeeze())
        maskObjects = ms.t2n(batch["maskObjects"].squeeze())

        gt_masks, gt_labels = gt2mask(maskClasses, maskObjects)
        


        score_dict = mAP_eval.eval_instance_segmentation_voc(
            pred_masks, pred_labels, pred_scores,
            gt_masks, gt_labels, iou_thresh=self.thresh, 
            use_07_metric=False,
            n_pos=n_pos, 
            score=map_scores, 
            match=match)


        # COMPUTE SCORE
        return {"score":score_dict["map"], 
                "n_pos":score_dict['n_pos'], 
                "map_scores":score_dict["score"],
                "match":score_dict["match"]}

    ### RUNNING AVERAGE ####
    def get_running_average(self):
        return self._average(self.n_pos, self.score, self.match)

    def update_running_average(self, model, batch):

        score_dict = self.scoreBatch(model, batch, 
                                     n_pos=self.n_pos, 
                                     map_scores=self.score, 
                                     match=self.match)
        
        self.n_pos = score_dict["n_pos"]
        self.score = score_dict["map_scores"]
        self.match = score_dict["match"]

        # if self.n_pos is not None and self._average(self.n_pos, self.score, self.match) <1:
        #     import ipdb; ipdb.set_trace()  # breakpoint 76ea7e6d //
        
        return score_dict
        
class map10(mAP_base):
    def __init__(self):
        super().__init__(iou_threshold=0.1)  

class map25(mAP_base):
    def __init__(self):
        super().__init__(iou_threshold=0.25)

class map50(mAP_base):
    def __init__(self):
        super().__init__(iou_threshold=0.5)      

class map75(mAP_base):
    def __init__(self):
        super().__init__(iou_threshold=0.75)



def calc_bd(A, B):
    A = ms.label2hot(A.ravel(), 
                               np.max(A) + 1)[:, 1:]
    A_area = np.sum(A, axis=0)

    B = ms.label2hot(B.ravel(), 
                           np.max(B) + 1)[:,1:]
    B_area = np.sum(B, axis=0)

    nom = 2 * np.dot(A.T, B)
    denom = A_area[:, None] + B_area[None, :] 

    dice = nom / denom
    M = len(dice)

    if 0 in dice.shape:
        bd = 0.
    else:
        bd = dice.max(1).sum()
        
    return {"bd": bd, "M":float(M)}



class SBD:
    def __init__(self):
        self.bd1 = None
        self.bd2 = None

        self.M1 = 0.
        self.M2 = 0.

    def _average(self, bd1, M1, bd2, M2):
        if M1 == 0 or M2 == 0:
            total = 0.
        else:
            total = min(bd1/M1, bd2/M2)

        return 1. - total

    def scoreBatch(self, model, batch):    
        model.eval()

        p_blobs = ms.t2n(model.predict(batch, metric="blobs"))
        labels = ms.t2n(batch["labels"]).astype(int)

        bd1_dict = calc_bd(p_blobs, labels)
        bd2_dict = calc_bd(labels, p_blobs)


        bd1 = bd1_dict["bd"]
        M1 = bd1_dict["M"]

        bd2 = bd2_dict["bd"]
        M2 = bd2_dict["M"]

        return {"score": self._average(bd1, M1, bd2, M2), 
                "bd1":bd1,
                "bd2":bd2,
                "M1":M1,
                "M2":M2}


    def update_running_average(self, model, batch):
        score_dict = self.scoreBatch(model, batch)
        if self.bd1 is None:
            self.bd1 = score_dict["bd1"]
            self.bd2 = score_dict["bd2"]

            self.M1 = score_dict["M1"]
            self.M2 = score_dict["M2"]

        else:
            self.bd1 += score_dict["bd1"]
            self.bd2 += score_dict["bd2"]

            self.M1 += score_dict["M1"]
            self.M2 += score_dict["M2"]

    def get_running_average(self):
        return self._average(self.bd1, self.M1, self.bd2, self.M2)
        

def pred2masks(maskBlobs, maskProbs):

    labels = np.where(maskBlobs.max((1,2)) != 0)[0]
    _, h, w = maskBlobs.shape
    n_objects = 0

    n_object_list = {}
    for l in labels:
        uniques = np.unique(maskBlobs[l])
        uniques = uniques[uniques!=0]

        n_objects += (uniques != 0).sum()
        n_object_list[l] = uniques

    pred_masks = np.zeros((n_objects, h, w), int)
    pred_labels = np.zeros(n_objects, int)
    pred_scores = np.zeros(n_objects, int)
    i = 0
    for l in labels:
        for o in n_object_list[l]:
            pred_masks[i, maskBlobs[l]==o] = 1

            pred_labels[i] = l+1
            pred_scores[i] = 1.0

            i+= 1

    return [pred_masks], [pred_labels], [pred_scores]

def gt2mask(maskClasses, maskObjects):
    labels = np.unique(maskObjects)  
    labels = labels[labels!=0]
    h, w = maskObjects.shape

    gt_masks = np.zeros((labels.size, h, w), int)
    gt_labels = np.zeros(labels.size, int)

    for i, l in enumerate(labels):
        ind = maskObjects==l
        gt_masks[i, ind] = 1

        gt_labels[i] = maskClasses[ind][0]

    return [gt_masks], [gt_labels]


class mIoU:
    def __init__(self):
        self.cf = None

    def _average(self, cf):
        Inter = np.diag(cf)
        G = cf.sum(axis=1)
        P = cf.sum(axis=0)
        union =  G + P - Inter

        nz = union != 0
        mIoU = Inter[nz] / union[nz]
        mIoU = np.mean(mIoU)

        return 1.-mIoU

    def scoreBatch(self, model, batch):    
        model.eval()
        

        preds = ms.t2n(model.predict(batch, metric="maskClasses"))
        maskClasses = ms.t2n(batch["maskClasses"])

        cf = confusion_matrix(y_true=preds.ravel(),
                         y_pred=maskClasses.ravel(),
                         labels=np.arange(max(model.n_classes,2)))

        return {"score": self._average(cf), "cf":cf}


    def update_running_average(self, model, batch):
        cf = self.scoreBatch(model, batch)["cf"]
        if self.cf is None:
            self.cf = cf
        else:
            self.cf += cf

    def get_running_average(self):
        return self._average(self.cf) 



class MAE:
    def __init__(self):
        self.n = 0
        self.se = 0

    def _average(self, se, n):
        return se / float(n)
    
    @staticmethod
    def compute_squared_error(true, pred):
        se = (abs(true.ravel() - pred.ravel())).sum()
        return se 

    @staticmethod
    def compute_score(true, pred):
        se = MAE.compute_squared_error(true, pred)
        n = float(true.size)
        return se / n

    def scoreBatch(self, model, batch):    
        model.eval()

        n = batch["images"].shape[0]
        # Predict pixel labels and get true labels
        pred = ms.t2n(model.predict(batch, predict_method="counts")["counts"]).ravel()
        true = ms.t2n(batch["counts"]).ravel()

        # COMPUTE SCORE
        se = MAE.compute_squared_error(true, pred)

        return {"score": self._average(se, n), 
                "se":se, "n":n}

    ### RUNNING AVERAGE ####
    def update_running_average(self, model, batch):
        score_dict = self.scoreBatch(model, batch)
        
        self.se += score_dict["se"]
        self.n += score_dict["n"]

        return score_dict
        

    def get_running_average(self):
        return self._average(self.se, self.n)
