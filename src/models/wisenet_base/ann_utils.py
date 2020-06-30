from . import misc as ms
import numpy as np
import os
import torch
from skimage import morphology as morph
from .addons import pycocotools
from .addons.pycocotools.cocoeval import COCOeval 
from .addons.pycocotools import mask as maskUtils
from torch.utils import data
from .datasets import base_dataset
import pandas as pd

def probs2GtAnnList(probs, points):
    points = ms.t2n(points.squeeze())
    annList = probs2annList(probs)["annList"]

    for ann in annList:
        ann["gt_pointList"] = []
        mask = ann2mask(ann)["mask"]
        binmask = mask * (points == ann['category_id'])

        n_points = binmask.sum()
        if n_points == 1:
            ann["status"] = "TP"
            ann_points = np.vstack(np.where(binmask)).T
            p = ann_points[0]
            ann["gt_pointList"] += [{"y":p[0], "x":p[1], "category_id":ann["category_id"]}]

        if n_points > 1: 
            ann["status"] = "SP"

            ann_points = np.vstack(np.where(binmask)).T
            for i in range(ann_points.shape[0]):
                p = ann_points[i]
                ann["gt_pointList"] += [{"y":p[0], "x":p[1], "category_id":ann["category_id"]}]

        if n_points == 0: 
            ann["status"] = "FP"
            

    return annList


# 
from skimage.morphology import watershed
from skimage.segmentation import find_boundaries
from scipy import ndimage
def probs2splitMask_all(probs, pointList=None):
    probs = ms.t2n(probs)

    if pointList is None:
        pointList = probs2blobs(probs)["pointList"]

    categories = []
    for p in pointList:
        categories += [p["category_id"]]
    categories = set(categories)

    maskList = []
    n,c,h,w = probs.shape
    background = np.zeros((n,h,w), int)

    point_id = 0
    point_mask = np.zeros( probs[:, 0].shape).squeeze()
    probs_mask = probs[:,1:].max(1)


    for p in pointList:
        point_id += 1
        point_mask[p["y"], p["x"]] = point_id

    distance = ndimage.black_tophat(probs_mask.squeeze(), 7)    
    mask = find_boundaries(watershed(distance, point_mask))
    
    background += mask

    maskList += [{"mask":1-mask}]

    background = background.clip(0,1)
    return {"maskList":maskList, "background":1-background}

@torch.no_grad()
def probs2annList(probs, image_id=None):
    probs = ms.t2n(probs).squeeze()
    n_classes, h, w = probs.shape
    
    mask_labels = ms.t2n(probs.argmax(0))
    annList = []
    # print(np.unique(mask_labels))
    for category_id in np.unique(mask_labels):
        if category_id == 0:
            continue
        # print("class", category_id)
        # ms.images(mask_labels)
        class_blobs = morph.label(mask_labels==category_id).squeeze()
        # print(np.unique(class_blobs))
        for u in np.unique(class_blobs):
            if u == 0:
                continue
            
            binmask = (class_blobs == u)
            # ms.images(binmask)
            # asdsa
            ann = mask2ann(binmask, category_id, image_id, height=h, width=w)

            # try:


            sub_probs =  probs[category_id]*binmask

            # except:
            #     print(c)
            #     print(probs.shape, binmask.shape)
            #     sdasdasds
            ind = np.where(sub_probs==sub_probs.max())
            # print(ind)
            r = ind[0][0]
            c = ind[1][0]
            
            ann["point"] = {"y":r, "x":c}
            annList += [ann]
            
    #         blobs[i,l-1] = morph.label(mask_labels==l)
    #         counts[i, l-1] = (np.unique(blobs[i,l-1]) != 0).sum()

    # blobs = blobs.astype(int)

    # if return_counts:
    #     return blobs, counts

    return {"annList":annList}


def batch2BestObjectnessAnnList(batch):
    points = batch["points"]
    pointList = mask2pointList(points)["pointList"]

    loss = torch.tensor(0.).cuda()
    if len(pointList) == 0:
        return loss

    if "single_point" in batch:
        single_point = True
    else:
        single_point = False
      

    propDict = pointList2propDict(pointList, batch, 
                                     single_point=single_point,
                                     thresh=0.5)

    propDict = propDict["propDict"]

    yList = []
    xList = []
    for p in pointList:
        yList += [p["y"]]
        xList += [p["x"]]


    for i in range(len(pointList)):
        annList = propDict[i]["annList"]

        if len(annList) == 0:
            mask = np.zeros(points.squeeze().shape)
            mask[propDict[i]["point"]["y"], propDict[i]["point"]["x"]] = 1
        else:
            mask = annList[0]["mask"]


def load_predAnnList(main_dict, predict_method, imageList=None, 
                     proposal_type="sharp", reset=None):
    predictList = ["BestObjectness", "UpperBound", "BestDice"]

    
    if predict_method not in predictList:
        raise ValueError("predict method should be in {}".format(predictList))
    dataset_name = main_dict["dataset_name"]
    base = "/mnt/projects/counting/Saves/main/"

    fname = base + "lcfcn_points/{}_{}_{}_annList.json".format(dataset_name, 
                                predict_method, proposal_type)




    if os.path.exists(fname) and reset != "reset":
        return ms.load_json(fname)

    else:
        if predict_method == "BestDice":
            model =  ms.load_best_model(main_dict)

        _, val_set = load_trainval(main_dict)

        loader = data.DataLoader(val_set, 
                       batch_size=1, 
                       num_workers=0, 
                       drop_last=False)

        # pointDict = load_LCFCNPoints(main_dict)

        annList = []
        for i, batch in enumerate(loader):
            print(i, "/", len(loader), " - annList")

            pointList = batch["lcfcn_pointList"]
            if len(pointList) == 0:
                continue

            if predict_method == "BestDice":
                pred_dict = model.predict(batch, predict_method="BestDice",
                                            proposal_type=proposal_type)
            else:
                pred_dict = eval("pointList2{}".format(predict_method))(pointList, batch, proposal_type)
            
            annList += pred_dict["annList"]

        ms.save_json(fname, annList)
        return annList

def get_image_dict(annList):
    imageDict = {}
    for a in annList:
        image_id = a["image_id"]
        if image_id in imageDict:
            imageDict[image_id] += [a]
        else:
            imageDict[image_id] = [a]

    return imageDict


def compute_BD(A, B):
    bd = 0.
    for i in range(len(A)):
        gt_mask = maskUtils.decode(A[i]["segmentation"])
        max_dice = 0.
        for j in range(len(B)):
            pred_mask = maskUtils.decode(B[j]["segmentation"])
            dice = compute_dice(pred_mask, gt_mask)
            max_dice = max(max_dice, dice)

        bd += max_dice

    return bd / len(A)

def get_SBD_DIC(gt_annDict, pred_annList):
    predDict = get_image_dict(pred_annList)
    gtDict = get_image_dict(gt_annDict["annotations"])

    sbd_array = np.zeros(len(predDict))
    dic_array = np.zeros(len(predDict))

    for m, k in enumerate(predDict.keys()):
        print(m,"/",len(predDict))

        pred_anns = predDict[k]
        gt_anns = gtDict[k]

        bd = min(compute_BD(gt_anns, pred_anns), compute_BD(pred_anns, gt_anns))


        sbd_array[m] = bd
        dic_array[m] = abs(len(gt_anns) - len(pred_anns))


    print("SBD:", sbd_array.mean(), "dic_array", dic_array.mean())


def compute_coverage(A, B):
    mucov = 0.
    mwcov = 0.
    gt_size = 0.
    for i in range( len(A)):
        gt_mask = maskUtils.decode(A[i]["segmentation"])
        best_iou = 0.
        for j in range(len(B)):
            pred_mask = maskUtils.decode(B[j]["segmentation"])
            iou=compute_iou(pred_mask, gt_mask)
            best_iou = max(best_iou, iou)

        mucov += best_iou  
        mwcov += best_iou * gt_mask.sum()
        gt_size += gt_mask.sum()

    return (mucov /  len(A)), (mwcov  / gt_size), gt_size

def get_MUCov(gt_annDict, pred_annList):

    predDict = get_image_dict(pred_annList)
    gtDict = get_image_dict(gt_annDict["annotations"])
    
    mucov_array = np.zeros(len(predDict))
    mwcov_array = np.zeros(len(predDict))
    for m, k in enumerate(predDict.keys()):
        print(m,"/",len(predDict))

        pred_anns = gtDict[k]
        gt_anns = predDict[k]

        mucov, mwcov, gt_size = compute_coverage(gt_anns, pred_anns)

        mucov_array[m] = mucov 
        mwcov_array[m] = mwcov

    print("MUCov:", mucov_array.mean(), "MWCov", mwcov_array.mean())



def get_perSizeResults(gt_annDict, pred_annList, pred_images_only=False):
    cocoGt = pycocotools.coco.COCO(gt_annDict)
    # pred_annList2 = []

    # In case of no predictions
    if len(pred_annList) == 0:
        pred_annList = [gt_annDict["annotations"][0]]
        pred_annList[0]["score"] = 1
        
    cocoDt = cocoGt.loadRes(pred_annList)
    
    cocoEval = COCOeval(cocoGt, cocoDt, "segm")
    cocoEval.params.iouThrs = np.array([.25, .5, .75])
    if pred_images_only:
        cocoEval.params.imgIds = get_image_ids(pred_annList)
    
    cocoEval.evaluate()
    cocoEval.accumulate()

    results = cocoEval.summarize()

    result_dict = {}
    for i in ["0.25", "0.5", "0.75"]:
        score = results["{}_all".format(i)]
        result_dict[i] = score

    for i in ["small", "medium", "large"]:
        score = results["0.5_{}".format(i)]
        result_dict[i] = score

    return {"results": results, "result_dict": result_dict}

def get_perCategoryResults(gt_annDict, pred_annDict):
    cocoGt = pycocotools.coco.COCO(gt_annDict)
    cocoDt = cocoGt.loadRes(pred_annDict)
    
    cocoEval = COCOeval(cocoGt, cocoDt, "segm")
    results = {}
    for i in cocoEval.params.catIds:
        cocoEval = COCOeval(cocoGt, cocoDt, "segm")
        cocoEval.params.iouThrs = np.array([.5])
        cocoEval.params.catIds = [i]
        cocoEval.params.areaRngLbl = ["all"]

        cocoEval.evaluate()
        cocoEval.accumulate()

        stat = list(cocoEval.summarize().values())
        assert len(stat) == 1
        results[i] = stat[0]

    # result_dict = {}
    # for i in ["0.25", "0.5", "0.75"]:
    #     score = results["{}_all".format(i)]
    #     result_dict[i] = score

    # for i in ["small", "medium", "large"]:
    #     score = results["0.5_{}".format(i)]
    #     result_dict[i] = score

    return {"results": results, "result_dict": results}

def get_image_ids(pred_annList):
    idList = set()
    for p in pred_annList:
        idList.add(p["image_id"])

    return list(idList)


# def pred_for_coco2014(main_dict, pred_annList):
#     if main_dict["dataset_name"] == "CocoDetection2014":
#         train_set,_ = ms.load_trainval(main_dict)
#         for p in pred_annList:
#             p["image_id"] = int(p["image_id"])
#             p["category_id"] = train_set.label2category[p["category_id"]]

#     return pred_annList

def test_upperboundmasks(main_dict, reset=None):
    # pointDict = load_LCFCNPoints(main_dict)
    fname = main_dict["path_baselines"].replace("baselines","upperboundmasks")

    if reset == "reset":
        pred_annList = load_predAnnList(main_dict, predict_method="UpperBoundMask",
                                            reset=reset)
        gt_annDict = load_gtAnnDict(main_dict)

        results = get_perSizeResults(gt_annDict, pred_annList)

        result_dict = results["result_dict"]

        result_dict["Model"] = "UpperBoundMasks"
        result_list = [result_dict]
        ms.save_pkl(fname, result_list)
    else:
        result_list = ms.load_pkl(fname)

    return result_list

def test_upperbound(main_dict, reset=None):
    # pointDict = load_LCFCNPoints(main_dict)
    fname = main_dict["path_baselines"].replace("baselines","upperbound")

    if reset == "reset":
        pred_annList = load_predAnnList(main_dict, reset=reset)
        gt_annDict = load_gtAnnDict(main_dict)

        results = get_perSizeResults(gt_annDict, pred_annList)

        result_dict = results["result_dict"]

        result_dict["Model"] = "UpperBound"
        result_list = [result_dict]
        ms.save_pkl(fname, result_list)
    else:
        result_list = ms.load_pkl(fname)

    return result_list

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

def test_model(main_dict, reset=None):
    # pointDict = load_LCFCNPoints(main_dict)
    _, val_set = load_trainval(main_dict)
    
    model = ms.load_best_model(main_dict)   
    gt_annDict = load_gtAnnDict(main_dict)
    # for i in range(50):
    import ipdb; ipdb.set_trace()  # breakpoint 887ad390 //

    if 1:
        b_list = [23]
        for i in b_list:
            batch = ms.get_batch(val_set, [i])
            annList_ub = pointList2UpperBoundMask(batch["lcfcn_pointList"], batch)["annList"]
            annList_bo = pointList2BestObjectness(batch["lcfcn_pointList"], batch)["annList"]
            annList = model.predict(batch, predict_method="BestDice")["annList"]
            results = get_perSizeResults(gt_annDict, annList)
            print(i,"Counts:", batch["counts"].item(),
                    " - BestObjectness:", len(annList_bo),
                    " - Model:", len(annList), 
                    " - UpperBound", len(annList_ub))
            print(i, 
                     get_perSizeResults(gt_annDict, annList_bo, pred_images_only=1)["result_dict"]["0.25"], 
                    get_perSizeResults(gt_annDict, annList, pred_images_only=1)["result_dict"]["0.25"], 
                    get_perSizeResults(gt_annDict, annList_ub, pred_images_only=1)["result_dict"]["0.25"])
        import ipdb; ipdb.set_trace()  # breakpoint 98d0193a //
        image_points = ms.get_image(batch["images"], batch["points"], enlarge=1,denorm=1)
        ms.images(image_points, annList2mask(annList)["mask"], 
                        win="model prediction")
        ms.images(batch["images"], annList2mask(annList_bo)["mask"],win="2",  denorm=1)
        ms.images(batch["images"], annList2mask(annList_ub)["mask"], win="3", denorm=1)
        ms.images(batch["images"], batch["points"], win="4", enlarge=1,denorm=1)
        ms.images(batch["images"],  model.predict(batch, predict_method="points")["blobs"], 
                        win="5", enlarge=1,denorm=1)
        ms.images(batch["images"], pointList2points(batch["lcfcn_pointList"])["mask"],
 
                              win="predicted_points", enlarge=1,denorm=1)
    fname = main_dict["path_baselines"].replace("baselines", main_dict["model_name"])

    if reset == "reset":
        _, val_set = load_trainval(main_dict)
        history = ms.load_history(main_dict)
        import ipdb; ipdb.set_trace()  # breakpoint a769ce6e //

        model = ms.load_best_model(main_dict)
        pred_annList = dataset2annList(model, val_set, 
                 predict_method="BestDice", 
                 n_val=None)

        pred_annList_up = load_predAnnList(main_dict, predict_method="UpperBoundMask")
        pred_annList_up = load_predAnnList(main_dict, predict_method="UpperBound")
        gt_annDict = load_gtAnnDict(main_dict)

        results = get_perSizeResults(gt_annDict, pred_annList)

        result_dict = results["result_dict"]

        result_dict["Model"] = main_dict["model_name"]
        result_list = [result_dict]
        ms.save_pkl(fname, result_list)
    else:
        result_list = ms.load_pkl(fname)

    return result_list


def test_baselines(main_dict, reset=None):
    #### Best Objectness
    if os.path.exists(main_dict["path_baselines"]) and reset!="reset":
        result_list = ms.load_pkl(main_dict["path_baselines"])
        return result_list

    else:
        gt_annDict = load_gtAnnDict(main_dict)
        pred_annList = load_predAnnList(main_dict,
                                        predict_method="BestObjectness",
                                        reset=reset)

        # idList1 = get_image_ids(pred_annList)
        # idList2 = get_image_ids(gt_annDict["annotations"])

        # results = get_perCategoryResults(gt_annDict, pred_annList)
        results = get_perSizeResults(gt_annDict, pred_annList)

        result_dict = results["result_dict"]

        result_dict["Model"] = "BestObjectness"
        result_list = [result_dict]

        #### Upper bound

        pred_annList = load_predAnnList(main_dict, predict_method="UpperBound", 
                                        reset=reset)
        # results = get_perCategoryResults(gt_annDict, pred_annList)
        results = get_perSizeResults(gt_annDict, pred_annList)


        result_dict = results["result_dict"]

        result_dict["Model"] = "UpperBound"
        result_list += [result_dict]
        ms.save_pkl(main_dict["path_baselines"], result_list)

    print(pd.DataFrame(result_list))
    return result_list


def validate(model, dataset, predict_method, n_val=None, return_annList=False):
    
    pred_annList = dataset2annList(model, dataset, 
                                   predict_method=predict_method, 
                                   n_val=n_val)

    
    gt_annDict = load_gtAnnDict(dataset)

    results = get_perSizeResults(gt_annDict, pred_annList)
    
    result_dict = results["result_dict"]
    result_dict["predict_method"] = predict_method

    if return_annList:
        return result_dict, pred_annList

    return result_dict


def validate_mIoU(model, dataset, predict_method, n_val=None, return_annList=False):
    
    
    iou_sum = 0
    n_objects = 0
    n_images = len(dataset)
    for i in range(n_images):
        
        batch = ms.get_batch(dataset, [i])
        maskObjects = batch["maskObjects"].squeeze()
        pred_dict = model.predict(batch, predict_method="BestDice")
    
        for ann in pred_dict["annList"]:
            point = ann["point"]
            label = maskObjects[point["y"], point["x"]].item()
            assert label != 0
            A = maskObjects == label
            B = ann2mask(ann)["mask"]
            iou_sum += compute_iou(ms.t2n(A), ms.t2n(B))
            n_objects += 1

            ms.images(batch["images"], A, denorm=1, win="GT")
            ms.images(batch["images"], B, denorm=1,  win="Pred")
            das
        print("{}/{} - {}".format(i, n_images, iou_sum / n_objects))


    iou_mean = iou_sum / n_objects


    # if return_annList:
    #     return result_dict, pred_annList

    # return result_dict

def qualitative(main_dict):
    pass

def test_best(main_dict, reset=None):
    _, val_set = load_trainval(main_dict)

    history = ms.load_history(main_dict)

    # if reset == "reset":    
    try:
        pred_annList = ms.load_best_annList(main_dict)
    except:
        model = ms.load_best_model(main_dict)
        pred_annList = dataset2annList(model, val_set, 
                 predict_method="BestDice", 
                 n_val=None)
        ms.save_pkl(main_dict["path_best_annList"], pred_annList)
    # else:
        # pred_annList = ms.load_best_annList(main_dict)

    gt_annDict = load_gtAnnDict(val_set)
    results = get_perCategoryResults(gt_annDict, pred_annList)
    
    result_dict = results["result_dict"]
    # result_dict[] = 
    # result_dict[] = 
    result_dict["Model"] = main_dict["model_name"]
    result_dict["epoch"] = history["best_model"]["epoch"]
    result_list = test_baselines(main_dict)
    result_list += [result_dict]

    print(pd.DataFrame(result_list))




# 0. load val
def load_trainval(main_dict):
    path_datasets = "datasets"
    path_transforms = 'addons/transforms.py'
    dataset_dict = ms.get_module_classes(path_datasets)
    transform_dict = ms.get_functions(path_transforms)
    dataset_name = main_dict["dataset_name"]
    train_set, val_set = ms.load_trainval({"dataset_name":dataset_name,
                               "path_datasets":path_datasets,
                               "trainTransformer":"Tr_WTP_NoFlip",
                               "testTransformer":"Te_WTP",
                               "dataset_options":{},
                               "dataset_dict":dataset_dict,
                               "transform_dict":transform_dict})

    annList_path = val_set.path + "/annotations/{}_gt_annList.json".format(val_set.split)
    val_set.annList_path = annList_path

    return train_set, val_set

def batch2annList(batch):
    annList = []
    image_id = int(batch["name"][0].replace("_",""))

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
    
# 1. Load gtAnnDict
def load_gtAnnDict(val_set, reset=None):
    reset = None
    annList_path = val_set.annList_path

    if os.path.exists(annList_path) and reset != "reset":
        return ms.load_json(annList_path)

    else:        
        ann_json = {}
        ann_json["categories"] = val_set.categories
        ann_json["type"] = "instances"


        # Images
        imageList = []
        annList = []
        id = 1

        for i in range(len(val_set)):
            print("{}/{}".format(i, len(val_set)))
            batch = val_set[i]

            image_id = batch["name"]

            height, width = batch["images"].shape[-2:]
            imageList += [{"file_name":batch["name"],
                          "height":height,
                          "width":width,
                          "id":batch["name"]}]

            maskObjects = batch["maskObjects"]
            maskClasses = batch["maskClasses"]
            n_objects = maskObjects[maskObjects!=255].max().item()
            
            for obj_id in range(1, n_objects+1):
                if obj_id == 0:
                    continue

                binmask = (maskObjects == obj_id)
                segmentation = maskUtils.encode(np.asfortranarray(ms.t2n(binmask))) 
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
                             "image_id":image_id,
                             "category_id":category_id}]
                id += 1

        ann_json["annotations"] = annList
        ann_json["images"] = imageList

        ms.save_json(annList_path, ann_json)

        # Save dummy results
        anns = ms.load_json(annList_path)
        fname_dummy = annList_path.replace(".json","_best.json")
        annList = anns["annotations"]
        for a in annList:
            a["score"] = 1

        ms.save_json(fname_dummy, annList)

# 1. Load dummyAnnDict
def assert_gtAnnDict(main_dict, reset=None):
    _, val_set = load_trainval(main_dict)
    annList_path = val_set.annList_path

    fname_dummy = annList_path.replace(".json","_best.json")

    # Test should be 100
    cocoGt = pycocotools.coco.COCO(annList_path)

    imgIds= sorted(cocoGt.getImgIds())
    assert len(imgIds) == len(val_set)
    assert len(ms.load_json(fname_dummy)) == len(ms.load_json(annList_path)["annotations"])

    assert len(ms.load_json(fname_dummy)) == len(cocoGt.anns)
    imgIds = imgIds[0:100]
    imgIds = np.random.choice(imgIds, min(100, len(imgIds)), replace=False)
    cocoDt = cocoGt.loadRes(fname_dummy)
    
    cocoEval = COCOeval(cocoGt, cocoDt, "segm")
    # cocoEval.params.imgIds  = imgIds.tolist()
    cocoEval.params.iouThrs = np.array([.25, .5, .75])
    
    cocoEval.evaluate()
    cocoEval.accumulate()
    stats = cocoEval.summarize()

    assert stats["0.25_all"] == 1
    assert stats["0.5_all"] == 1
    assert stats["0.75_all"] == 1


def maskList2annList(maskList, categoryList, image_id, scoreList=None):
    annList = []
    _, h, w = maskList.shape
    
    for i in range(maskList.shape[0]):
        binmask = maskList[i]

        seg = maskUtils.encode(np.asfortranarray(ms.t2n(binmask)).astype("uint8")) 
        seg["counts"] = seg["counts"].decode("utf-8")
        if scoreList is not None:
            score = scoreList[i]

        annList += [{"segmentation":seg,
              "iscrowd":0,
              "area":int(maskUtils.area(seg)),
             "image_id":image_id,
             "category_id":int(categoryList[i]),
             "height":h,
             "width":w,
             "score":score}]

    return annList

def blobs2annList(blobs, categoryDict, batch):
    if "maskVoid" not in batch:
        maskVoid = None
    else:
        maskVoid = batch["maskVoid"]
    h,w = blobs.shape
    annList = []
    blobs = ms.t2n(blobs)

    for u in np.unique(blobs):
        if u == 0:
            continue
        binmask = (blobs == u)
        if maskVoid is not None:
            binmask = binmask * (ms.t2n(maskVoid).squeeze())

        seg = maskUtils.encode(np.asfortranarray(ms.t2n(binmask)).astype("uint8")) 
        seg["counts"] = seg["counts"].decode("utf-8")
        score = 1

        annList += [{"segmentation":seg,
              "iscrowd":0,
              "area":int(maskUtils.area(seg)),
             "image_id":batch["name"][0],
             "category_id":int(categoryDict[u]),
             "height":h,
             "width":w,
             "score":score}]

    return annList

# def load_LCFCNPoints(main_dict, reset=None):
#     dataset_name = main_dict["dataset_name"]
#     base = "/mnt/projects/counting/Saves/main/"

#     if "Pascal" in dataset_name:
#         path = base + "dataset:Pascal2007_model:Res50FCN_metric:mRMSE_loss:water_loss_B_config:basic/"
    
#     elif "CityScapes" in dataset_name:
#         path = base + "dataset:CityScapes_model:Res50FCN_metric:mRMSE_loss:water_loss_B_config:basic/"
    
#     elif "CocoDetection2014" in dataset_name:
#         path = base + "dataset:CocoDetection2014_model:Res50FCN_metric:mRMSE_loss:water_loss_B_config:sample3000/"

#     elif "Kitti" in dataset_name:
#         path = base + "dataset:Kitti_model:Res50FCN_metric:mRMSE_loss:water_loss_B_config:basic/"


#     elif "Plants" in dataset_name:
#         path = base + "dataset:Plants_model:Res50FCN_metric:mRMSE_loss:water_loss_B_config:basic/"

#     else:
#         raise
    
#     fname = base + "lcfcn_points/{}.pkl".format(dataset_name)

#     if os.path.exists(fname):
#         history = ms.load_pkl(path + "history.pkl")
#         pointDict = ms.load_pkl(fname)

#         if pointDict["best_model"]["epoch"] != history["best_model"]["epoch"]:            
#             reset = "reset"

#     if os.path.exists(fname) and reset != "reset":
#         return pointDict

#     else:
#         train_set, val_set = load_trainval(main_dict)

#         # Create Model
#         model = main_dict["model_dict"]["Res50FCN"](train_set)
#         model.load_state_dict(torch.load(path + 
#                                          "/State_Dicts/best_model.pth"))
#         history = ms.load_pkl(path + "history.pkl")
#         model.cuda()

#         loader = data.DataLoader(val_set, 
#                        batch_size=1, 
#                        num_workers=1, 
#                        drop_last=False)
#         pointDict = {}
#         model.eval()
#         for i, batch in enumerate(loader):
#             print(i, "/", len(loader), " - pointDict")
#             pointList = model.predict(batch, predict_method="points")["pointList"]
#             pointDict[batch["name"][0]] = pointList

#         pointDict["best_model"] = history['best_model']
#         pointDict['main_dict'] = history['main_dict']

#         ms.save_pkl(fname, pointDict)

#         return pointDict


def eval_MUCov():
    pass

def eval_MWCov():
    pass

def annList2BestDice(annList, batch):
    sharp_proposals = batch2proposals(batch, proposal_type="sharp")

  
    new_annList = []
    

    if "maskVoid" in batch and batch["maskVoid"] is not None:
        maskVoid = batch["maskVoid"]
    else:
        maskVoid = None

    for ann in annList:

        binmask = ann2mask(ann)["mask"]
        best_dice = 0.
        best_mask = None
        for sharp_ann in sharp_proposals:
            if ann["score"] < 0.5:
                continue
            score = dice(sharp_ann["mask"], binmask)
            if score > best_dice:
                best_dice = score
                best_mask = sharp_ann["mask"]
                # prop_score = ann["score"]

        if best_mask is None:
            best_mask = binmask

        if maskVoid is not None:
            binmask = best_mask * (ms.t2n(maskVoid).squeeze())
        else:
            binmask = best_mask

        seg = maskUtils.encode(np.asfortranarray(ms.t2n(binmask)).astype("uint8")) 
        seg["counts"] = seg["counts"].decode("utf-8")
        ann["score"] = best_dice
        ann["segmentation"] = seg

        new_annList += [ann]
    
    return {"annList":new_annList}


def compute_intersection(proposal, mask, bg, alpha, beta, gamma):
    overlap = A*B # Logical AND
    iou = overlap.sum()
    return iou

def compute_sum(A, B):
    overlap = A*B # Logical AND
    iou = overlap.sum()
    return iou


def compute_iou(A, B):
    overlap = A*B # Logical AND
    union = A + B # Logical OR
    union = union.clip(0, 1)
    iou = overlap.sum()/float(union.sum())
    return iou

def compute_dice(a, b):
    num = 2 *  (a * b).sum()
    denom = a.sum() + b.sum()

    return num / denom


def blobs2BestDice(blobs, categoryDict, propDict, batch, blobs_sims=None, sim_func=None):
    if sim_func is None:
        sim_func = dice

    h, w = blobs.shape
    annList = []
    blobs_copy = np.zeros(blobs.shape, int)

    if "maskVoid" in batch:
        maskVoid = batch["maskVoid"]
    else:
        maskVoid = None

    for u in np.unique(blobs):
        u = int(u)
        if u == 0:
            continue
        binmask = (blobs == u)
        best_dice = 0.
        best_mask = None
        point = propDict['propDict'][u-1]["point"]

        for ann in propDict['propDict'][u-1]["annList"]:
            score = sim_func(ann["mask"], binmask)
            if score > best_dice:
                best_dice = score
                best_mask = ann["mask"]
                prop_score = ann["score"]

        if best_mask is None:
            best_mask = (blobs==u).astype(int)

        
        if maskVoid is not None:
            binmask = best_mask * (ms.t2n(maskVoid).squeeze())
        else:
            binmask = best_mask

        if best_mask is None:
            blobs_copy[blobs==u] = u 
        else:
            blobs_copy[binmask==1] = u


        seg = maskUtils.encode(np.asfortranarray(ms.t2n(binmask)).astype("uint8")) 
        seg["counts"] = seg["counts"].decode("utf-8")
        score = best_dice

        # if batch["dataset"] == "coco2014":
        #     image_id = int(batch["name"][0])
        # else:
        image_id = batch["name"][0]

        annList += [{"segmentation":seg,
              "iscrowd":0,
              "area":int(maskUtils.area(seg)),
             "image_id":image_id,
             "category_id":int(categoryDict[u]),
             "height":h,
             "width":w,
             "score":score,
             "point":point}]
        
    return {"blobs":blobs_copy, "annList":annList}

@torch.no_grad()
def dataset2annList(model, dataset, 
             predict_method="BestObjectness", 
             n_val=None):


    loader = data.DataLoader(dataset, 
                   batch_size=1, 
                   num_workers=0, 
                   drop_last=False)

    n_batches = len(loader)
    n_images = len(dataset)

    annList = []
    for i, batch in enumerate(loader):
        print(i, "/", n_batches)
        pred_dict = model.predict(batch, predict_method=predict_method)
        
        assert batch["name"][0] not in model.trained_batch_names
        
        if n_val is None or n_val == n_images:
            pass
        elif i > n_val:
            break

        annList += pred_dict["annList"]

    return annList

def pointList2mask(pointList):
    
    mask = np.zeros(pointList[0]["shape"], int)
    for p in pointList:
        mask[:, p["y"], p["x"]] = p["category_id"]

    return {"mask":mask}


def pointList2points(pointList):
    return pointList2mask(pointList)




def print_results(results):
    pass


def probs2blobs(probs):
    annList = []

    probs = ms.t2n(probs)
    n, n_classes, h, w = probs.shape
  
    counts = np.zeros((n, n_classes-1))
    
    # Binary case
    pred_mask = ms.t2n(probs.argmax(1))
    blobs = np.zeros(pred_mask.shape)
    points = np.zeros(pred_mask.shape)

    max_id = 0
    for i in range(n):        
        for category_id in np.unique(pred_mask[i]):
            if category_id == 0:
                continue          

            ind = pred_mask==category_id

            connected_components = morph.label(ind)

            uniques = np.unique(connected_components)

            blobs[ind] = connected_components[ind] + max_id
            max_id = uniques.max() + max_id

            n_blobs = (uniques != 0).sum()

            counts[i, category_id-1] = n_blobs

            for j in range(1, n_blobs+1):
                binmask = connected_components == j
                blob_probs = probs[i, category_id] * binmask
                y, x = np.unravel_index(blob_probs.squeeze().argmax(), blob_probs.squeeze().shape)

                points[i, y, x] = category_id
                annList += [mask2ann(binmask, category_id, image_id=-1, 
                        height=binmask.shape[1], 
                        width=binmask.shape[2], maskVoid=None, 
                        score=None, point={"y":y,"x":x, 
                        "prob":float(blob_probs[blob_probs!=0].max()),
                        "category_id":int(category_id)})]
                


    blobs = blobs.astype(int)

    # Get points

    return {"blobs":blobs, "annList":annList, "probs":probs,
            "counts":counts, "points":points,
            "pointList":mask2pointList(points)["pointList"],
            "pred_mask":pred_mask,
            "n_blobs":len(annList)}


def mask2pointList(mask):
    pointList = []
    mask = ms.t2n(mask)
    pointInd = np.where(mask.squeeze())
    n_points = pointInd[0].size

    for p in range(n_points):

        p_y, p_x = pointInd[0][p], pointInd[1][p]
        point_category = mask[0, p_y,p_x]

        pointList += [{"y":p_y,"x":p_x, "category_id":int(point_category), 
                       "shape":mask.shape}]

    return {"pointList":pointList}

# def se_pairwise(fi, fj):

#     return (fi - fj).pow(2).mean(1)



    
@torch.no_grad()
def pointList2BestObjectness(pointList, batch, proposal_type="sharp"):
    if "single_point" in batch:
        single_point = True
    else:
        single_point = False

    propDict = pointList2propDict(pointList, batch, thresh=0.5, 
                                  single_point=single_point,
                                  proposal_type=proposal_type)
    
    
    h,w = propDict["background"].squeeze().shape
    blobs = np.zeros((h,w), int)
    categoryDict = {}
    if "maskVoid" in batch and batch["maskVoid"] is not None:
        maskVoid = ms.t2n(batch["maskVoid"].squeeze())
    else:
        maskVoid = None

    annList = []
    for i, prop in enumerate(propDict['propDict']):
        if len(prop["annList"]) == 0:
            continue
        blobs[prop["annList"][0]["mask"] !=0] = i+1

        
        categoryDict[i+1] = prop["category_id"]

        if maskVoid is not None:
            binmask = prop["annList"][0]["mask"] * (ms.t2n(maskVoid).squeeze())
        else:
            binmask = prop["annList"][0]["mask"]

        seg = maskUtils.encode(np.asfortranarray(ms.t2n(binmask)).astype("uint8")) 
        seg["counts"] = seg["counts"].decode("utf-8")
        score = prop["annList"][0]["score"]
        point = prop["point"]
        annList += [{"segmentation":seg,
              "iscrowd":0,
              "area":int(maskUtils.area(seg)),
             "image_id":batch["name"][0],
             "category_id":int(prop['category_id']),
             "height":h,
             "width":w,
             "score":score,
             "point":point}]


    return {"annList":annList, "blobs": blobs, "categoryDict":categoryDict}


def pointList2UpperBoundMask(pointList, batch, proposal_type="sharp"):
    # propDict = pointList2propDict(pointList, batch, thresh=0.5)

    n, c = batch["counts"].shape
    n, _, h, w = batch["images"].shape

    n_objects = batch["maskObjects"].max()
    if "maskVoid" in batch:
        maskVoid = ms.t2n(batch["maskVoid"].squeeze())
    else:
        maskVoid = None

    ###########
    annList = []
    visited = set()
    for p_index, p in enumerate(pointList):
        category_id = p["category_id"]
        best_score = 1.0

        gt_object_found = False
        cls_mask = (batch["maskClasses"] == category_id).long().squeeze()

        for k in range(n_objects):
            if (k+1) in visited:
                continue
            gt_mask = (batch["maskObjects"] == (k+1)).long().squeeze()
            
            if (gt_mask[p["y"],p["x"]].item() != 0 and 
                cls_mask[p["y"],p["x"]].item()==1):
                gt_object_found = True
                visited.add(k+1)
                break

        if gt_object_found == False:
            continue

        gt_mask = ms.t2n(gt_mask)

        ann = mask2ann(gt_mask, p["category_id"],
                        image_id=batch["name"][0],
                        height=batch["images"].shape[2],
                        width=batch["images"].shape[3], 
                        maskVoid=maskVoid, score=best_score)

        annList += [ann]

    return {"annList":annList}


def BOLCFCN2UpperBoundMask(model, batch, proposal_type="sharp"):
    # propDict = pointList2propDict(pointList, batch, thresh=0.5)

    n, c = batch["counts"].shape
    n, _, h, w = batch["images"].shape

    n_objects = batch["maskObjects"].max()
    if "maskVoid" in batch:
        maskVoid = ms.t2n(batch["maskVoid"].squeeze())
    else:
        maskVoid = None

    ###########
    annList = []
    visited = set()
    for p_index, p in enumerate(pointList):
        category_id = p["category_id"]
        best_score = 1.0

        gt_object_found = False
        cls_mask = (batch["maskClasses"] == category_id).long().squeeze()

        for k in range(n_objects):
            if (k+1) in visited:
                continue
            gt_mask = (batch["maskObjects"] == (k+1)).long().squeeze()
            
            if (gt_mask[p["y"],p["x"]].item() != 0 and 
                cls_mask[p["y"],p["x"]].item()==1):
                gt_object_found = True
                visited.add(k+1)
                break

        if gt_object_found == False:
            continue

        gt_mask = ms.t2n(gt_mask)

        ann = mask2ann(gt_mask, p["category_id"],
                        image_id=batch["name"][0],
                        height=batch["images"].shape[2],
                        width=batch["images"].shape[3], 
                        maskVoid=maskVoid, score=best_score)

        annList += [ann]

    return {"annList":annList}


def pointList2UpperBound(pointList, batch, proposal_type="sharp"):
    propDict = pointList2propDict(pointList, batch, proposal_type=proposal_type,
                                  thresh=0.5)

    n, c = batch["counts"].shape
    n, _, h, w = batch["images"].shape

    n_objects = batch["maskObjects"].max()
    if "maskVoid" in batch:
        maskVoid = ms.t2n(batch["maskVoid"].squeeze())
    else:
        maskVoid = None

    ###########
    annList = []
    for p_index, p in enumerate(pointList):
        # category_id = p["category_id"]
        best_score = 0
        best_mask = None

        gt_object_found = False
        # cls_mask = (batch["maskClasses"] == category_id).long().squeeze()

        for k in range(n_objects):
            gt_mask = (batch["maskObjects"] == (k+1)).long().squeeze()
            
            if (gt_mask[p["y"],p["x"]].item() != 0):
                gt_object_found = True

                break

        if gt_object_found == False:
            continue

        gt_mask = ms.t2n(gt_mask)
        #########################################
        best_score = 0
        best_mask = None

        for proposal_ann in propDict["propDict"][p_index]["annList"]:
            proposal_mask =  proposal_ann["mask"]
            #########
            # proposal_mask = resize(proposal_mask, (h, w), order=0)

            score = dice(gt_mask, proposal_mask)

            if score > best_score:
                best_mask = proposal_mask
                best_score = score 

        # ms.images(batch["images"], best_mask, denorm=1)        
        if best_mask is not None:
            ann = mask2ann(best_mask, p["category_id"],
                            image_id=batch["name"][0],
                            height=batch["images"].shape[2],
                            width=batch["images"].shape[3], 
                            maskVoid=maskVoid, score=best_score)

            annList += [ann]

    return {"annList":annList}


def mask2ann(binmask, category_id, image_id, 
             height, width, maskVoid=None, score=None, point=None):
    binmask = binmask.squeeze().astype("uint8")

    if maskVoid is not None:
        binmask = binmask * maskVoid

    segmentation = maskUtils.encode(np.asfortranarray(ms.t2n(binmask)).astype("uint8")) 
    segmentation["counts"] = segmentation["counts"].decode("utf-8")
    # print(segmentation)
    ann = {"segmentation":segmentation,
                  "iscrowd":0,
                  "area":int(maskUtils.area(segmentation)),
                 "image_id":image_id,
                 "category_id":int(category_id),
                 "height":height,
                 "width":width,
                 "score":score,
                 "point":point}

    return ann

    # for i, prop in enumerate(propDict['propDict']):
    #     if len(prop["annList"]) == 0:
    #         continue
    #     blobs[prop["annList"][0]["mask"] !=0] = i+1

        
    #     categoryDict[i+1] = prop["category_id"]

    #     if maskVoid is not None:
    #         binmask = prop["annList"][0]["mask"] * (ms.t2n(maskVoid).squeeze())
    #     else:
    #         binmask = prop["annList"][0]["mask"]

    #     seg = maskUtils.encode(np.asfortranarray(ms.t2n(binmask)).astype("uint8")) 
    #     seg["counts"] = seg["counts"].decode("utf-8")
    #     score = prop["annList"][0]["score"]

    #     annList += [{"segmentation":seg,
    #           "iscrowd":0,
    #           "area":int(maskUtils.area(seg)),
    #          "image_id":batch["name"][0],
    #          "category_id":int(prop['category_id']),
    #          "height":h,
    #          "width":w,
    #          "score":score}]

    #############

    # for p in blob_dict["pointList"]:
        
    #     category_id = p["category_id"]
    #     best_score = 0
    #     best_mask = None

    #     gt_object_found = False
    #     cls_mask = (batch["maskClasses"] == category_id).long().squeeze()

    #     for k in range(n_objects):
    #         gt_mask = (batch["maskObjects"] == (k+1)).long().squeeze()
            
    #         if (gt_mask[p["y"],p["x"]].item() != 0 and 
    #             cls_mask[p["y"],p["x"]].item()==1):
    #             gt_object_found = True

    #             break

    #     if gt_object_found == False:
    #         continue

    #         # label_class = (pred_mask*batch["maskClasses"]).max().item()

    #     gt_mask = ms.t2n(gt_mask)
    #     #########################################

    #     best_score = 0
    #     best_mask = None

    #     for k in range(len(sharp_proposals)):
    #         proposal_ann = sharp_proposals[k]
    #         if proposal_ann["score"] < 0.5:
    #             continue

    #         proposal_mask =  proposal_ann["mask"]

    #         #########
    #         # proposal_mask = resize(proposal_mask, (h, w), order=0)
    #         score = sf.dice(gt_mask, proposal_mask)

    #         #########

    #         if score > best_score:
    #             best_mask = proposal_mask
    #             best_score = score 

    #     # ms.images(batch["images"], best_mask, denorm=1)
    #     if best_mask is not None:
    #         ann = bu.mask2ann(best_mask, p["category_id"],
    #                         image_id=batch["name"][0],
    #                         height=batch["images"].shape[2],
    #                         width=batch["images"].shape[3], 
    #                         maskVoid=maskVoid, score=best_score)
    #         annList += [ann]

def naive(pred_mask, gt_mask):
    return (pred_mask*gt_mask).mean()

def dice(pred_mask, gt_mask, smooth=1.):
    iflat = pred_mask.ravel()
    tflat = gt_mask.ravel()
    intersection = (iflat * tflat).sum()

    score = ((2. * intersection) /
            (iflat.sum() + tflat.sum() + smooth))
    return score


def cosine_similarity(pred_mask, true_mask):
    scale = np.linalg.norm(pred_mask) * np.linalg.norm(true_mask)
    return pred_mask.ravel().dot(true_mask.ravel()) / scale


from skimage.transform import resize
@torch.no_grad()
def pointList2propDict(pointList, batch, single_point=False, thresh=0.5, 
                        proposal_type="sharp"):

    proposals = batch2proposals(batch, proposal_type=proposal_type)

    propDict = []
    shape = pointList[0]["shape"]
    foreground = np.zeros(shape, int)

    if single_point:
        points = pointList2mask(pointList)["mask"]

    idDict= {}
    annDict = {}
    aggDict = {}
    for i, p in enumerate(pointList):
        annDict[i] = []
        idDict[i] = []

    n_points = len(annDict)
    for k in range(len(proposals)):
        proposal_ann = proposals[k]

        if not (proposal_ann["score"] > thresh):
            continue

        proposal_mask =  proposal_ann["mask"]

        for i, p in enumerate(pointList):
            if proposal_mask[p["y"], p["x"]]==0:
                continue
            
            if single_point and ((points!=0) * proposal_mask).sum() > 1:
               continue

            # score = proposal_ann["score"]
            if i not in aggDict: 
                aggDict[i] = proposal_mask.astype(float) * proposal_ann["score"]
            else:
                aggDict[i] += proposal_mask.astype(float) * proposal_ann["score"]

            annDict[i] += [proposal_ann]
            idDict[i] += [k]
    
    for i in range(n_points):
        point_annList = annDict[i]
        point_idList = idDict[i]
        p = pointList[i]

        mask = annList2mask(point_annList)["mask"]
        if mask is not None:
            if mask.shape != foreground.squeeze().shape:
                mask = resize(mask, foreground.shape)
            foreground = foreground + mask

        #foreground[foreground<2]=0
        propDict += [{"annList":point_annList,"point":p, "idList":point_idList, 
                      "category_id":int(p["category_id"])}]
        #########  

    return {"propDict":propDict, "aggDict":aggDict, 
            "foreground":foreground, "background":(foreground==0).astype(int)}


def annList2maskList(annList, box=False, color=False):
    n_anns = len(annList)
    if n_anns == 0:
        return {"mask":None}

    ann = annList[0]
    try:
        h, w = ann["mask"].shape
    except:
        h, w = ann["height"], ann["width"]
    maskList = np.zeros((h, w, n_anns), int)
    categoryList = np.zeros(n_anns, int)
    for i in range(n_anns):
        ann = annList[i]

        if "mask" in ann:
            ann_mask = ann["mask"]
        else:
            ann_mask = maskUtils.decode(ann["segmentation"])

        assert ann_mask.max() <= 1
        maskList[:,:,i] = ann_mask

        categoryList[i] = ann["category_id"]
    # mask[mask==1] = ann["category_id"]
    return {"maskList":maskList, "categoryList":categoryList}

def annList2mask(annList, box=False, color=False):
    n_anns = len(annList)
    if n_anns == 0:
        return {"mask":None}

    ann = annList[0]
    try:
        h, w = ann["mask"].shape
    except:
        h, w = ann["height"], ann["width"]
    mask = np.zeros((h, w), int)

    for i in range(n_anns):
        ann = annList[i]

        if "mask" in ann:
            ann_mask = ann["mask"]
        else:
            ann_mask = maskUtils.decode(ann["segmentation"])

        assert ann_mask.max() <= 1

        if color:
            mask[ann_mask!=0] = i + 1
        else:
            mask += ann_mask

    for i in range(n_anns):
        if box:
            box_mask = ann2bbox(annList[i])["mask"]
            mask[box_mask!=0] = 255

    # mask[mask==1] = ann["category_id"]
    return {"mask":mask}

def ann2mask(ann):
    if "mask" in ann:
        mask = ann["mask"]
    else:
        mask =  maskUtils.decode(ann["segmentation"])
    # mask[mask==1] = ann["category_id"]
    return {"mask":mask}



def ann2bbox(ann):
    bbox = maskUtils.toBbox(ann["segmentation"])
    r, c = ann["segmentation"]["size"]
    
    x, y, w, h = bbox

    mask = bbox2mask(r, c, x, y, w, h)
    ye = min(r-1, y+h)
    xe = min(c-1, x+w) 

    return {"mask":mask, "shape":(x, y, xe, ye)}


def bbox2mask(r,c, x,y,w,h):
    x,y,w,h = map(int, (x,y,w,h))    
    mask = np.zeros((r, c), int) 

    ye = min(r-1, y+h)
    xe = min(c-1, x+w) 

    mask[y:ye, x] = 1
    mask[y:ye, xe] = 1
    mask[y, x:xe] = 1
    mask[ye, x:xe] = 1

    return mask


def batch2proposals(batch, proposal_type):
    if proposal_type == "mcg":
        print("MCG used")
        proposals = MCGProposals(batch)
    else:
        # print("Sharp used")
        proposals = SharpProposals(batch)

    return proposals

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

                ann = mask2ann(proposal_mask, category_id=1,
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