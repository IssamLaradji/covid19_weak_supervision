
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
from pycocotools import mask as maskUtils
annType = ['segm','bbox','keypoints']
annType = annType[0]      #specify type here
prefix = 'person_keypoints' if annType=='keypoints' else 'instances'

dataDir='/mnt/datasets/public/issam/COCO2014/'
dataType='val2014'
# annFile = '%s/annotations/%s_%s.json'%(dataDir,prefix,dataType)
annFile = "/mnt/datasets/public/issam/VOCdevkit/annotations/pascal_val2012.json"
import ipdb; ipdb.set_trace()  # breakpoint 332e4eec //

cocoGt = COCO(annFile)

imgIds=sorted(cocoGt.getImgIds())
imgIds=imgIds[0:100]
imgId = imgIds[np.random.randint(100)]
# 
# resFile='%s/results/%s_%s_fake%s100_results.json'
# resFile = resFile%(dataDir, prefix, dataType, annType)
# cocoDt=cocoGt.loadRes("fake_seg_results.json")
import ipdb; ipdb.set_trace()  # breakpoint 69a585bb //

cocoEval = COCOeval(cocoGt, cocoDt, annType)
cocoEval.params.imgIds  = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()