import misc as ms 
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
from pycocotools import mask as maskUtils

def pascal2cocoformat():
    dataset = ms.load_trainval({"dataset_name":"VOC"})
    fname = "/mnt/datasets/public/issam/VOCdevkit/annotations/"
    fname += "instances_val2012.json"

    tmp = ms.load_json("/mnt/datasets/public/issam/"
                       "VOCdevkit/annotations/pascal_val2012.json")
    
    ann_json = {}
    ann_json["categories"] = tmp["categories"]
    ann_json["type"] = "instances"


    # Images
    imageList = []
    annList = []
    id = 1
    for i in range(len(dataset)):
        print("{}/{}".format(i, len(dataset)))
        batch = dataset[i]
        image_id = int(batch["name"])

        height, width = batch["images"].shape[-2:]
        imageList += [{"file_name":batch["name"]+".jpg",
                      "height":height,
                      "width":width,
                      "id":image_id}]

        maskObjects = batch["maskObjects"]
        maskClasses = batch["maskClasses"]
        n_objects = maskObjects[maskObjects!=255].max()
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


    

    ms.save_json(fname, ann_json)

    anns = ms.load_json(fname)
    fname_dummy = fname.replace(".json","_best.json")
    annList = anns["annotations"]
    for a in annList:
        a["score"] = 1

    ms.save_json(fname_dummy, annList)

    # ms.images(batch["images"], 

def evaluateAnnList(annList, visualize=False):
    path_base = "/mnt/datasets/public/issam/VOCdevkit/annotations/"
    fname = "{}/instances_val2012.json".format(path_base)
    
    cocoGt = COCO(fname)
    ms.save_json("tmp.json", annList)
    cocoDt = cocoGt.loadRes("tmp.json")

    cocoEval = COCOeval(cocoGt, cocoDt, "segm")

    # cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.params.imgIds  = list(set([v["image_id"] for v in cocoDt.anns.values()]))

    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print("# images:", len(cocoEval.params.imgIds))

    if visualize:
        for image_id in cocoDt.getImgIds():
            visGT(cocoGt, cocoDt, image_id)

    return cocoEval, cocoDt

def visGT(cocoGt, cocoDt, batch):
    image_id = int(batch["name"][0])
    annList = cocoGt.imgToAnns[image_id]
    mask = annList2mask(annList)
    ms.images(batch["images"], mask, denorm=1)

    dt_mask = annList2mask(cocoDt.imgToAnns[image_id])
    

    cocoEval = COCOeval(cocoGt, cocoDt, "segm")

    # cocoEval = COCOeval(cocoGt, cocoDt, annType)
    #cocoEval.params.imgIds  = list(set([v["image_id"] for v in cocoDt.anns.values()]))
   
    cocoEval.params.imgIds = [image_id]
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    print(image_id)
    print("Dt:", len(cocoDt.imgToAnns[image_id]), "Gt:", len(annList))
    ms.images(batch["images"], dt_mask, denorm=1, win="pred")

# def valList(cocoGt, cocoDt, val_set, ind):
#     image_id_list = []
#     for i in ind:
#         batch = ms.get_batch(val_set, [i])
#         image_id = int(batch["name"][0])
#         annList = cocoGt.imgToAnns[image_id]
#         mask = annList2mask(annList)

#         dt_mask = annList2mask(cocoDt.imgToAnns[image_id])
#         ms.images(batch["images"], mask, denorm=1, win=str(i))
#         ms.images(batch["images"], dt_mask, denorm=1, win=str(i)+"_pred")

#         image_id_list += [image_id]

#     cocoEval = COCOeval(cocoGt, cocoDt, "segm")

#     # cocoEval = COCOeval(cocoGt, cocoDt, annType)
#     #cocoEval.params.imgIds  = list(set([v["image_id"] for v in cocoDt.anns.values()]))
   
#     cocoEval.params.imgIds = [image_id_list]
#     cocoEval.evaluate()
#     cocoEval.accumulate()
#     cocoEval.summarize()




def annList2mask(annList):
    def ann2mask(ann):
        mask =  maskUtils.decode(ann["segmentation"])
        mask[mask==1] = ann["category_id"]

        return mask
    
    mask = None
    for ann in annList:
        if mask is None:
            mask = ann2mask(ann)
        else:
            mask += ann2mask(ann)

    return mask



pascal_categories = {}


def get_mask_image(model, gm):
    image_path = "/mnt/datasets/public/issam/VOCdevkit/VOC2007/JPEGImages/"
    seg = gm["segmentation"]
    img_id = gm["image_id"]

    mask = maskUtils.decode(seg)

    img = ms.imread(image_path + "{:0>6}.jpg".format(img_id))
    ms.images(img, mask)


    # model = get_style_model(name="udnie")
    # style1 = ms.t2n(style_content(model, img))[:,:, :mask.shape[0], :mask.shape[1]].squeeze()
    model = get_style_model(name="rain_princess")
    style2 = ms.t2n(style_content(model, img))[:,:, :mask.shape[0], :mask.shape[1]].squeeze()

    ms.images(style2*mask[None] + ms.l2f(img)*(1-mask[None]), win="324324")
    return img, mask


def save_all_proposals(where="/mnt/datasets/public/issam/VOCdevkit/"\
                "proposals/sharpmask/pascal_proposals/", 
                path="pascal_val2007"):
    if 1:
        import glob

        loc = "/mnt/datasets/public/issam/VOCdevkit/proposals/"\
                             "sharpmask/{}/jsons".format(path)

        proposals_dict = {}
       
        jsonList = glob.glob(loc + "/*.json")

        for json in jsonList:
            proposals = ms.load_json(json)
            n = len(proposals)
            for i in range(n):
                print(str(i) + "/" + str(n) + " proposals")

                image_id = proposals[i]["image_id"]
                
                if image_id in proposals_dict:
                    proposals_dict[image_id] += [proposals[i]] 
                else:
                    proposals_dict[image_id] = [proposals[i]] 


        n = len(proposals_dict)
        for j, image_id in enumerate(proposals_dict):
            print(str(j) + "/" + str(n))
            ms.save_json(where + "{}.json".format(str(image_id)), proposals_dict[image_id])










######### Pascal 


# TODO: Some folder settings were changed compared to the original
# repository -- need to change the realtive paths for pascal voc here
# More specifically, the folder that is created after untarring the pascal
# is named VOCdevkit now instead of VOC2012

import skimage.io as io
import numpy as np
import os
import glob
from scipy.io import loadmat
import cv2
from scipy.ndimage.filters import gaussian_filter 
from scipy.misc import imsave, imread
import tqdm

def save_segmentationInstances(pascal_root):
    """Return look-up table with number and correspondng class names
    for PASCAL VOC segmentation dataset. Two special classes are: 0 -
    background and 255 - ambigious region. All others are numerated from
    1 to 20.
    
    Returns
    -------
    classes_lut : dict
        look-up table with number and correspondng class names
    """
    import glob

    imgList = glob.glob(pascal_root + "/SegmentationClass/*")
    imgNames = [ut.extract_fname(img) for img in imgList]
    for name in tqdm.tqdm(imgNames):
        segClass = imread(pascal_root + "/SegmentationClass/" + name)
        segObj = imread(pascal_root + "/SegmentationObject/" + name)
        segClassLabel = ut.rgb2label(segClass, n_classes=256)
        segObjLabel = ut.rgb2label(segObj, n_classes=256)

        categories = {}
        for c in np.unique(segObjLabel):
            if (c == 0) or (c == 255):
                continue

            ind = segObjLabel == c
            vals = segClassLabel[ind]
            v1 = vals.min()
            v2 = vals.max()

            assert v1 == v2

            categories[int(c)] = int(v1)
            
        imsave(pascal_root + "/SegmentationInstances/%s" % name,
               segObjLabel)

        ut.save_json(pascal_root + "/SegmentationInstances/%s" % name.replace(".png",".json"),
               {"Categories":categories})
 

def pascal_segmentation_lut():
    """Return look-up table with number and correspondng class names
    for PASCAL VOC segmentation dataset. Two special classes are: 0 -
    background and 255 - ambigious region. All others are numerated from
    1 to 20.
    
    Returns
    -------
    classes_lut : dict
        look-up table with number and correspondng class names
    """

    class_names = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                   'dog', 'horse', 'motorbike', 'person', 'potted-plant',
                   'sheep', 'sofa', 'train', 'tv/monitor', 'ambigious']
    
    enumerated_array = enumerate(class_names[:-1])
    
    classes_lut = list(enumerated_array)
    
    # Add a special class representing ambigious regions
    # which has index 255.
    classes_lut.append((255, class_names[-1]))
    
    classes_lut = dict(classes_lut)

    return classes_lut


def get_pascal_segmentation_images_lists_txts(pascal_root):
    """Return full paths to files in PASCAL VOC with train and val image name lists.
    This function returns full paths to files which contain names of images
    and respective annotations for the segmentation in PASCAL VOC.
    
    Parameters
    ----------
    pascal_root : string
        Full path to the root of PASCAL VOC dataset.
    
    Returns
    -------
    full_filenames_txts : [string, string, string]
        Array that contains paths for train/val/trainval txts with images names.
    """
    
    segmentation_images_lists_relative_folder = 'VOC2012/ImageSets/Segmentation'
    
    segmentation_images_lists_folder = os.path.join(pascal_root,
                                                    segmentation_images_lists_relative_folder)
    
    pascal_train_list_filename = os.path.join(segmentation_images_lists_folder,
                                              'train.txt')

    pascal_validation_list_filename = os.path.join(segmentation_images_lists_folder,
                                                   'val.txt')
    
    pascal_trainval_list_filname = os.path.join(segmentation_images_lists_folder,
                                                'trainval.txt')
    
    return [
            pascal_train_list_filename,
            pascal_validation_list_filename,
            pascal_trainval_list_filname
           ]


def readlines_with_strip(filename):
    """Reads lines from specified file with whitespaced removed on both sides.
    The function reads each line in the specified file and applies string.strip()
    function to each line which results in removing all whitespaces on both ends
    of each string. Also removes the newline symbol which is usually present
    after the lines wre read using readlines() function.
    
    Parameters
    ----------
    filename : string
        Full path to the root of PASCAL VOC dataset.
    
    Returns
    -------
    clean_lines : array of strings
        Strings that were read from the file and cleaned up.
    """
    
    # Get raw filnames from the file
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Clean filenames from whitespaces and newline symbols
    clean_lines = map(lambda x: x.strip(), lines)
    
    return clean_lines


def readlines_with_strip_array_version(filenames_array):
    """The function that is similar to readlines_with_strip() but for array of filenames.
    Takes array of filenames as an input and applies readlines_with_strip() to each element.
    
    Parameters
    ----------
    array of filenams : array of strings
        Array of strings. Each specifies a path to a file.
    
    Returns
    -------
    clean_lines : array of (array of strings)
        Strings that were read from the file and cleaned up.
    """
    
    multiple_files_clean_lines = map(readlines_with_strip, filenames_array)
    
    return multiple_files_clean_lines


def add_full_path_and_extention_to_filenames(filenames_array, full_path, extention):
    """Concatenates full path to the left of the image and file extention to the right.
    The function accepts array of filenames without fullpath and extention like 'cat'
    and adds specified full path and extetion to each of the filenames in the array like
    'full/path/to/somewhere/cat.jpg.
    Parameters
    ----------
    filenames_array : array of strings
        Array of strings representing filenames
    full_path : string
        Full path string to be added on the left to each filename
    extention : string
        Extention string to be added on the right to each filename
    Returns
    -------
    full_filenames : array of strings
        updated array with filenames
    """
    full_filenames = map(lambda x: os.path.join(full_path, x) + '.' + extention, filenames_array)
    
    return full_filenames


def add_full_path_and_extention_to_filenames_array_version(filenames_array_array, full_path, extention):
    """Array version of the add_full_path_and_extention_to_filenames() function.
    Applies add_full_path_and_extention_to_filenames() to each element of array.
    Parameters
    ----------
    filenames_array_array : array of array of strings
        Array of strings representing filenames
    full_path : string
        Full path string to be added on the left to each filename
    extention : string
        Extention string to be added on the right to each filename
    Returns
    -------
    full_filenames : array of array of strings
        updated array of array with filenames
    """
    result = map(lambda x: add_full_path_and_extention_to_filenames(x, full_path, extention),
                 filenames_array_array)
    
    return result


def get_pascal_segmentation_image_annotation_filenames_pairs(pascal_root):
    """Return (image, annotation) filenames pairs from PASCAL VOC segmentation dataset.
    Returns three dimensional array where first dimension represents the type
    of the dataset: train, val or trainval in the respective order. Second
    dimension represents the a pair of images in that belongs to a particular
    dataset. And third one is responsible for the first or second element in the
    dataset.
    Parameters
    ----------
    pascal_root : string
        Path to the PASCAL VOC dataset root that is usually named 'VOC2012'
        after being extracted from tar file.
    Returns
    -------
    image_annotation_filename_pairs : 
        Array with filename pairs.
    """
    
    pascal_relative_images_folder = 'JPEGImages'
    pascal_relative_class_annotations_folder = 'SegmentationClass'
    
    images_extention = 'jpg'
    annotations_extention = 'png'
    
    pascal_images_folder = os.path.join(pascal_root, pascal_relative_images_folder)
    pascal_class_annotations_folder = os.path.join(pascal_root, pascal_relative_class_annotations_folder)
    
    pascal_images_lists_txts = get_pascal_segmentation_images_lists_txts(pascal_root)
    
    pascal_image_names = readlines_with_strip_array_version(pascal_images_lists_txts)
    
    images_full_names = add_full_path_and_extention_to_filenames_array_version(pascal_image_names,
                                                                               pascal_images_folder,
                                                                               images_extention)
    
    annotations_full_names = add_full_path_and_extention_to_filenames_array_version(pascal_image_names,
                                                                                    pascal_class_annotations_folder,
                                                                                    annotations_extention)
    
    # Combine so that we have [(images full filenames, annotation full names), .. ]
    # where each element in the array represent train, val, trainval sets.
    # Overall, we have 3 elements in the array.
    temp = zip(images_full_names, annotations_full_names)
    
    # Now we should combine the elements of images full filenames annotation full names
    # so that we have pairs of respective image plus annotation
    # [[(pair_1), (pair_1), ..], [(pair_1), (pair_2), ..] ..]
    # Overall, we have 3 elements -- representing train/val/trainval datasets
    image_annotation_filename_pairs = map(lambda x: zip(*x), temp)
    
    return image_annotation_filename_pairs


def convert_pascal_berkeley_augmented_mat_annotations_to_png(pascal_berkeley_augmented_root):
    """ Creates a new folder in the root folder of the dataset with annotations stored in .png.
    The function accepts a full path to the root of Berkeley augmented Pascal VOC segmentation
    dataset and converts annotations that are stored in .mat files to .png files. It creates
    a new folder dataset/cls_png where all the converted files will be located. If this
    directory already exists the function does nothing. The Berkley augmented dataset
    can be downloaded from here:
    http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz
    
    Parameters
    ----------
    pascal_berkeley_augmented_root : string
        Full path to the root of augmented Berkley PASCAL VOC dataset.
    
    """
    
    import scipy.io
    
    def read_class_annotation_array_from_berkeley_mat(mat_filename, key='GTcls'):
    
        #  Mat to png conversion for http://www.cs.berkeley.edu/~bharath2/codes/SBD/download.html
        # 'GTcls' key is for class segmentation
        # 'GTinst' key is for instance segmentation
        #  Credit: https://github.com/martinkersner/train-DeepLab/blob/master/utils.py
    
        mat = scipy.io.loadmat(mat_filename, mat_dtype=True, squeeze_me=True, struct_as_record=False)
        return mat[key].Segmentation
    
    
    mat_file_extension_string = '.mat'
    png_file_extension_string = '.png'
    relative_path_to_annotation_mat_files = 'dataset/cls'
    relative_path_to_annotation_png_files = 'dataset/cls_png'

    mat_file_extension_string_length = len(mat_file_extension_string)


    annotation_mat_files_fullpath = os.path.join(pascal_berkeley_augmented_root,
                                                 relative_path_to_annotation_mat_files)

    annotation_png_save_fullpath = os.path.join(pascal_berkeley_augmented_root,
                                                relative_path_to_annotation_png_files)

    # Create the folder where all the converted png files will be placed
    # If the folder already exists, do nothing
    if not os.path.exists(annotation_png_save_fullpath):

        os.makedirs(annotation_png_save_fullpath)
    else:

        return


    mat_files_names = os.listdir(annotation_mat_files_fullpath)

    for current_mat_file_name in mat_files_names:

        current_file_name_without_extention = current_mat_file_name[:-mat_file_extension_string_length]

        current_mat_file_full_path = os.path.join(annotation_mat_files_fullpath,
                                                  current_mat_file_name)

        current_png_file_full_path_to_be_saved = os.path.join(annotation_png_save_fullpath,
                                                              current_file_name_without_extention)
        
        current_png_file_full_path_to_be_saved += png_file_extension_string

        annotation_array = read_class_annotation_array_from_berkeley_mat(current_mat_file_full_path)
        
        # TODO: hide 'low-contrast' image warning during saving.
        io.imsave(current_png_file_full_path_to_be_saved, annotation_array)


def get_pascal_berkeley_augmented_segmentation_images_lists_txts(pascal_berkeley_root):
    """Return full paths to files in PASCAL Berkley augmented VOC with train and val image name lists.
    This function returns full paths to files which contain names of images
    and respective annotations for the segmentation in PASCAL VOC.
    
    Parameters
    ----------
    pascal_berkeley_root : string
        Full path to the root of PASCAL VOC Berkley augmented dataset.
    
    Returns
    -------
    full_filenames_txts : [string, string]
        Array that contains paths for train/val txts with images names.
    """
    
    segmentation_images_lists_relative_folder = 'dataset'
    
    segmentation_images_lists_folder = os.path.join(pascal_berkeley_root,
                                                    segmentation_images_lists_relative_folder)
    
    
    # TODO: add function that will joing both train.txt and val.txt into trainval.txt
    pascal_train_list_filename = os.path.join(segmentation_images_lists_folder,
                                              'train.txt')

    pascal_validation_list_filename = os.path.join(segmentation_images_lists_folder,
                                                   'val.txt')
    
    return [
            pascal_train_list_filename,
            pascal_validation_list_filename
           ]


def get_pascal_berkeley_augmented_segmentation_image_annotation_filenames_pairs(pascal_berkeley_root):
    """Return (image, annotation) filenames pairs from PASCAL Berkeley VOC segmentation dataset.
    Returns three dimensional array where first dimension represents the type
    of the dataset: train, val in the respective order. Second
    dimension represents the a pair of images in that belongs to a particular
    dataset. And third one is responsible for the first or second element in the
    dataset.
    Parameters
    ----------
    pascal_berkeley_root : string
        Path to the PASCAL Berkeley VOC dataset root that is usually named 'benchmark_RELEASE'
        after being extracted from tar file.
    Returns
    -------
    image_annotation_filename_pairs : 
        Array with filename pairs.
    """

    pascal_relative_images_folder = 'dataset/img'
    pascal_relative_class_annotations_folder = 'dataset/cls_png'

    images_extention = 'jpg'
    annotations_extention = 'png'

    pascal_images_folder = os.path.join(pascal_berkeley_root, pascal_relative_images_folder)
    pascal_class_annotations_folder = os.path.join(pascal_berkeley_root, pascal_relative_class_annotations_folder)

    pascal_images_lists_txts = get_pascal_berkeley_augmented_segmentation_images_lists_txts(pascal_berkeley_root)

    pascal_image_names = readlines_with_strip_array_version(pascal_images_lists_txts)

    images_full_names = add_full_path_and_extention_to_filenames_array_version(pascal_image_names,
                                                                               pascal_images_folder,
                                                                               images_extention)

    annotations_full_names = add_full_path_and_extention_to_filenames_array_version(pascal_image_names,
                                                                                    pascal_class_annotations_folder,
                                                                                    annotations_extention)

    # Combine so that we have [(images full filenames, annotation full names), .. ]
    # where each element in the array represent train, val, trainval sets.
    # Overall, we have 3 elements in the array.
    temp = zip(images_full_names, annotations_full_names)

    # Now we should combine the elements of images full filenames annotation full names
    # so that we have pairs of respective image plus annotation
    # [[(pair_1), (pair_1), ..], [(pair_1), (pair_2), ..] ..]
    # Overall, we have 3 elements -- representing train/val/trainval datasets
    image_annotation_filename_pairs = map(lambda x: zip(*x), temp)
    
    return image_annotation_filename_pairs


def get_pascal_berkeley_augmented_selected_image_annotation_filenames_pairs(pascal_berkeley_root, selected_names):
    """Returns (image, annotation) filenames pairs from PASCAL Berkeley VOC segmentation dataset for selected names.
    The function accepts the selected file names from PASCAL Berkeley VOC segmentation dataset
    and returns image, annotation pairs with fullpath and extention for those names.
    Parameters
    ----------
    pascal_berkeley_root : string
        Path to the PASCAL Berkeley VOC dataset root that is usually named 'benchmark_RELEASE'
        after being extracted from tar file.
    selected_names : array of strings
        Selected filenames from PASCAL VOC Berkeley that can be read from txt files that
        come with dataset.
    Returns
    -------
    image_annotation_pairs : 
        Array with filename pairs with fullnames.
    """
    pascal_relative_images_folder = 'dataset/img'
    pascal_relative_class_annotations_folder = 'dataset/cls_png'

    images_extention = 'jpg'
    annotations_extention = 'png'
    
    pascal_images_folder = os.path.join(pascal_berkeley_root, pascal_relative_images_folder)
    pascal_class_annotations_folder = os.path.join(pascal_berkeley_root, pascal_relative_class_annotations_folder)
    
    images_full_names = add_full_path_and_extention_to_filenames(selected_names,
                                                                 pascal_images_folder,
                                                                 images_extention)
    
    annotations_full_names = add_full_path_and_extention_to_filenames(selected_names,
                                                                      pascal_class_annotations_folder,
                                                                      annotations_extention)
    
    image_annotation_pairs = zip(images_full_names, 
                                 annotations_full_names)
    
    return image_annotation_pairs


def get_pascal_selected_image_annotation_filenames_pairs(pascal_root, selected_names):
    """Returns (image, annotation) filenames pairs from PASCAL VOC segmentation dataset for selected names.
    The function accepts the selected file names from PASCAL VOC segmentation dataset
    and returns image, annotation pairs with fullpath and extention for those names.
    Parameters
    ----------
    pascal_root : string
        Path to the PASCAL VOC dataset root that is usually named 'VOC2012'
        after being extracted from tar file.
    selected_names : array of strings
        Selected filenames from PASCAL VOC that can be read from txt files that
        come with dataset.
    Returns
    -------
    image_annotation_pairs : 
        Array with filename pairs with fullnames.
    """
    pascal_relative_images_folder = 'VOC2012/JPEGImages'
    pascal_relative_class_annotations_folder = 'VOC2012/SegmentationClass'
    
    images_extention = 'jpg'
    annotations_extention = 'png'
    
    pascal_images_folder = os.path.join(pascal_root, pascal_relative_images_folder)
    pascal_class_annotations_folder = os.path.join(pascal_root, pascal_relative_class_annotations_folder)
    
    images_full_names = add_full_path_and_extention_to_filenames(selected_names,
                                                                 pascal_images_folder,
                                                                 images_extention)
    
    annotations_full_names = add_full_path_and_extention_to_filenames(selected_names,
                                                                      pascal_class_annotations_folder,
                                                                      annotations_extention)
    
    image_annotation_pairs = zip(images_full_names, 
                                 annotations_full_names)
    
    return image_annotation_pairs

def get_augmented_filenames(pascal_root, pascal_berkeley_root, mode=2):
    pascal_txts = get_pascal_segmentation_images_lists_txts(pascal_root=pascal_root)
    berkeley_txts = get_pascal_berkeley_augmented_segmentation_images_lists_txts(pascal_berkeley_root=pascal_berkeley_root)

    pascal_name_lists = readlines_with_strip_array_version(pascal_txts)
    berkeley_name_lists = readlines_with_strip_array_version(berkeley_txts)

    pascal_train_name_set, pascal_val_name_set, _ = map(lambda x: set(x), pascal_name_lists)
    berkeley_train_name_set, berkeley_val_name_set = map(lambda x: set(x), berkeley_name_lists)

    
    all_berkeley = berkeley_train_name_set | berkeley_val_name_set
    all_pascal = pascal_train_name_set | pascal_val_name_set

    everything = all_berkeley | all_pascal

    # Extract the validation subset based on selected mode
    if mode == 1:

        # 1449 validation images, 10582 training images
        validation = pascal_val_name_set

    if mode == 2:

        # 904 validatioin images, 11127 training images
        validation = pascal_val_name_set - berkeley_train_name_set

    if mode == 3:

        # 346 validation images, 11685 training images
        validation = pascal_val_name_set - all_berkeley

    # The rest of the dataset is for training
    train = everything - validation

    # Get the part that can be extracted from berkeley
    train_from_berkeley = train & all_berkeley

    # The rest of the data will be loaded from pascal
    train_from_pascal = train - train_from_berkeley

    train_imgNames = list(train_from_pascal) + list(train_from_berkeley)
    val_imgNames = list(validation)

    

    ## Permutate
    # np.random.seed(3)

    
    train_imgNames = np.sort(train_imgNames)
    # train_imgNames = np.random.permutation(train_imgNames)
    assert train_imgNames.size == np.unique(train_imgNames).size
    train_imgNames = train_imgNames.tolist()

    return {"train_imgNames": train_imgNames, "val_imgNames": val_imgNames}


def get_augmented_list(pascal_root, pascal_berkeley_root, mode=2):
    """Returns image/annotation filenames pairs train/val splits from combined Pascal VOC.
    Returns two arrays with train and validation split respectively that has
    image full filename/ annotation full filename pairs in each of the that were derived
    from PASCAL and PASCAL Berkeley Augmented dataset. The Berkley augmented dataset
    can be downloaded from here:
    http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz
    Consider running convert_pascal_berkeley_augmented_mat_annotations_to_png() after extraction.
    
    The PASCAL VOC dataset can be downloaded from here:
    http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    Consider specifying root full names for both of them as arguments for this function
    after extracting them.
    The function has three type of train/val splits(credit matconvnet-fcn):
    
        Let BT, BV, PT, PV, and PX be the Berkeley training and validation
        sets and PASCAL segmentation challenge training, validation, and
        test sets. Let T, V, X the final trainig, validation, and test
        sets.

        Mode 1::
              V = PV (same validation set as PASCAL)

        Mode 2:: (default))
              V = PV \ BT (PASCAL val set that is not a Berkeley training
              image)

        Mode 3::
              V = PV \ (BV + BT)

        In all cases:

              S = PT + PV + BT + BV
              X = PX  (the test set is uncahgend)
              T = (S \ V) \ X (the rest is training material)
    Parameters
    ----------
    pascal_root : string
        Path to the PASCAL VOC dataset root that is usually named 'VOC2012'
        after being extracted from tar file.
    pascal_berkeley_root : string
        Path to the PASCAL Berkeley VOC dataset root that is usually named 'benchmark_RELEASE'
        after being extracted from tar file.
    mode: int
        The type of train/val data split. Read the function main description for more info.
    Returns
    -------
    image_annotation_pairs : [[(string, string), .. , (string, string)][(string, string), .., (string, string)]]
        Array with filename pairs with fullnames.
    """
    pascal_txts = get_pascal_segmentation_images_lists_txts(pascal_root=pascal_root)
    berkeley_txts = get_pascal_berkeley_augmented_segmentation_images_lists_txts(pascal_berkeley_root=pascal_berkeley_root)

    pascal_name_lists = readlines_with_strip_array_version(pascal_txts)
    berkeley_name_lists = readlines_with_strip_array_version(berkeley_txts)

    pascal_train_name_set, pascal_val_name_set, _ = map(lambda x: set(x), pascal_name_lists)
    berkeley_train_name_set, berkeley_val_name_set = map(lambda x: set(x), berkeley_name_lists)

    all_berkeley = berkeley_train_name_set | berkeley_val_name_set
    all_pascal = pascal_train_name_set | pascal_val_name_set

    everything = all_berkeley | all_pascal

    # Extract the validation subset based on selected mode
    if mode == 1:

        # 1449 validation images, 10582 training images
        validation = pascal_val_name_set

    if mode == 2:

        # 904 validatioin images, 11127 training images
        validation = pascal_val_name_set - berkeley_train_name_set

    if mode == 3:

        # 346 validation images, 11685 training images
        validation = pascal_val_name_set - all_berkeley

    # The rest of the dataset is for training
    train = everything - validation

    # Get the part that can be extracted from berkeley
    train_from_berkeley = train & all_berkeley

    # The rest of the data will be loaded from pascal
    train_from_pascal = train - train_from_berkeley

    train_from_berkeley_image_annotation_pairs = \
    get_pascal_berkeley_augmented_selected_image_annotation_filenames_pairs(pascal_berkeley_root,
                                                                            list(train_from_berkeley))

    train_from_pascal_image_annotation_pairs = \
    get_pascal_selected_image_annotation_filenames_pairs(pascal_root,
                                                         list(train_from_pascal))

    overall_train_image_annotation_filename_pairs = (
    list(train_from_berkeley_image_annotation_pairs) + 
    list(train_from_pascal_image_annotation_pairs))

    overall_val_image_annotation_filename_pairs = (
        list(get_pascal_selected_image_annotation_filenames_pairs(pascal_root,
                                                         validation)))



    return overall_train_image_annotation_filename_pairs, overall_val_image_annotation_filename_pairs