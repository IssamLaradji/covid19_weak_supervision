import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import pickle
import cv2
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import sys
import torch.nn.functional as F
import os
from datasets import plants
import os.path as osp
import datetime
import random
import ann_utils as au
import timeit, tqdm
import pandas as pd 
from pydoc import locate
start = timeit.default_timer()
import datetime as dt
import time
from losses import losses
from metrics import metrics
from skimage.segmentation import find_boundaries
from torch.utils.data.sampler import SubsetRandomSampler
import misc as ms 
import PIL.Image
from addons import vis, val
from addons import transforms
from core import response_map as rm
from scipy.stats import describe
from skimage.segmentation import felzenszwalb, slic, quickshift

# from core import dataset2cocoformat as d2c
from pycocotools import mask as maskUtils
from core import blobs_utils as bu
def test_prm(model, batch, i=1, j=0):
  # image_size = 448
  # image pre-processor
  mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


  transformer = transforms.transforms.Compose([
            
             transforms.transforms.ToTensor(),
             transforms.transforms.Normalize(*mean_std)])             

  if 1:
    # model.inference()
    raw_img = PIL.Image.open('packages/PRM/demo/data/sample%d.jpg'%i).convert('RGB')
    img = transformer(raw_img).unsqueeze(0).cuda().requires_grad_()

    visual_cues  = rm.peak_response(model.backbone, img, peak_threshold=1)
    # visual_cues  = model(img, peak_threshold=1)
    confidence, class_response_maps, class_peak_responses, peak_response_maps = visual_cues
    ms.images(img, denorm=1, win="2343"); ms.images(ms.gray2cmap(peak_response_maps[j]))

from scipy.ndimage.filters import gaussian_filter 
import numpy as np

from scipy import fftpack

def kernel(h, w, sigma=1.5):
    from scipy import stats

    sx, sy = h, w
    X, Y = np.ogrid[0:sx, 0:sy]
    psf = stats.norm.pdf(np.sqrt((X - sx/2)**2 + (Y - sy/2)**2), 0, sigma)    
    psf /= psf.sum()

    return psf

def convolve(star, psf):
    star_fft = fftpack.fftshift(fftpack.fftn(star))
    psf_fft = fftpack.fftshift(fftpack.fftn(psf))
    return fftpack.fftshift(fftpack.ifftn(fftpack.ifftshift(star_fft*psf_fft)))

def deconvolve(star, psf):
    star_fft = fftpack.fftshift(fftpack.fftn(star))
    psf_fft = fftpack.fftshift(fftpack.fftn(psf))
    return fftpack.fftshift(fftpack.ifftn(fftpack.ifftshift(star_fft/psf_fft)))

def debug(main_dict):
  #ud.debug_sheep(main_dict)

  loss_dict = main_dict["loss_dict"]
  metric_dict = main_dict["metric_dict"]

  metric_name = main_dict["metric_name"]
  metric_class = main_dict["metric_dict"][metric_name]
  loss_name = main_dict["loss_name"]
  batch_size = main_dict["batch_size"]

  ms.print_welcome(main_dict)
  train_set, val_set = ms.load_trainval(main_dict)



  #test_set = ms.load_test(main_dict)
  #  train_set, val_set = ms.load_trainval(main_dict)
  #batch=ms.get_batch(test_set, indices=[509]) 
  # batch=ms.get_batch(val_set, indices=[0, 4, 9]) 
  


  # b2 = um.get_batch(val_set, indices=[4]) 
  # ms.fitBatch(model, batch, loss_name="image_loss", opt=opt, epochs=100)
  # batch_train=ms.get_batch(val_set, indices=[15]) 
  # batch=ms.get_batch(val_set, indices=[15]) 

  # tr_batch=ms.get_batch(val_set, indices=[2]) 
  #batch=ms.get_batch(val_set, indices=[1,2,3,12,13,14,16,17,67,68,70])
  # batch=ms.get_batch(val_set,indices=[300])
  
  # ms.images(batch["images"], batch["points"],denorm=1,enlarge=1)
  # for i in range(len(val_set)):
  #   batch=ms.get_batch(val_set,indices=[i])
  #   sharp_proposals = prp.Sharp_class(batch)


  # sharp_proposals = prp.Sharp_class(batch)
  # pointList = bu.mask2pointList(batch["points"])["pointList"]
  # propDict = bu.pointList2propDict(pointList, sharp_proposals, thresh=0.5)
  # for i in range(len(train_set)):
  #   print(i)

  #   sharp_proposals = prp.Sharp_class(ms.get_batch(train_set,indices=[i]))
  # d2c.pascal2cocoformat(main_dict)
  # model, opt, _ = ms.init_model_and_opt(main_dict)
  # 
  # history = ms.load_history(main_dict)
  # print(pd.DataFrame(history["val"]))
  # print(pd.DataFrame(history["train"])[loss_name])
  model, opt, _ = ms.init_model_and_opt(main_dict)
  import ipdb; ipdb.set_trace()  # breakpoint b87b640d //
  
  batch = ms.get_batch(val_set, indices=[1]) 
  ms.visBlobs(model, batch, predict_method="BestDice")
  import ipdb; ipdb.set_trace()  # breakpoint a18a7b92 //
  plants.save_test_to_h5(main_dict)
  model = ms.load_best_model(main_dict)
  
  if 1:
    import train
    train.validation_phase_mIoU(ms.load_history(main_dict), main_dict, model, val_set, 
                                "BestDice", 0)
  
  test_set = ms.load_test(main_dict)
  batch = ms.get_batch(test_set, indices=[4]) 
  # model, opt, _ = ms.init_model_and_opt(main_dict)
  batch = ms.get_batch(val_set, indices=[4]) 
 

  ms.images(batch["images"], model.predict( batch , predict_method="BestDice", use_trans=1,
                                            sim_func=au.compute_dice)["blobs"], denorm=1)

  val_dict, pred_annList = au.validate(model, val_set, 
            predict_method="BestDice", 
            n_val=None, return_annList=True)
  
  model = ms.load_lcfcn(train_set, mode="lcfcn")
  val_dict, pred_annList = au.validate(model, val_set, 
            predict_method="BestDice", 
            n_val=None, return_annList=True)
  model = ms.load_best_model(main_dict)
  ms.fitBatch(model, batch, loss_function=loss_dict[loss_name], 
                opt=opt, epochs=10000, visualize=True)
  import ipdb; ipdb.set_trace()  # breakpoint 4e08c360 //
  
  if os.path.exists(main_dict["path_history"]):
    history = ms.load_history(main_dict)
    print("# Trained Images:", len(history["trained_batch_names"]), "/", len(train_set))
    print("# Epoch:", history["epoch"])
    # print(pd.DataFrame(history["val"]))
    # val_names = [ms.extract_fname(fname).replace(".jpg", "") for fname in val_set.img_names]
    
    # assert np.in1d(history["trained_batch_names"], val_names).sum() == 0


  import ipdb; ipdb.set_trace()  # breakpoint ef2ce16b //
  # print(pd.DataFrame(history["val"]))
  # model, opt, _ = ms.init_model_and_opt(main_dict)
  model = ms.load_best_model(main_dict)
  ms.visBlobs(model, batch, predict_method="BestDice")

  ms.images(batch["images"], au.annList2mask(model.predict(batch, 
      predict_method="loc")["annList"])["mask"], enlarge=1, denorm=1)

  ms.fitBatch(model, batch, loss_function=loss_dict[loss_name], 
                opt=opt, epochs=10000, visualize=True)
  
  ms.images(model, ms.get_batch(val_set, indices=[0]))
  model = ms.load_best_model(main_dict)
  model.extract_proposalMasks(ms.get_batch(train_set, indices=[1]))
  mask = model.visualize(ms.get_batch(val_set, indices=[1]) )
  img = ms.f2l(ms.t2n((ms.denormalize(batch["images"])))).squeeze()
  segments_slic = slic(img, n_segments=250, compactness=10, sigma=1)

  results = model.predict(batch, "ewr")
 
  ms.fitBatch(model, batch, loss_function=loss_dict[loss_name], 
                opt=opt, epochs=10000, visualize=True)

  model = ms.load_best_model(main_dict)
  model.visualize( ms.get_batch(val_set, indices=[0]))
  ms.visBlobs(model, ms.get_batch(val_set, indices=[0]) , with_void=True)
 
  ms.images(ms.gray2cmap(model(batch["images"].cuda())["mask"].squeeze()))
  h, w = batch["images"].shape[-2:]
  ms.images(ms.gray2cmap(deconvolve(ms.t2n(model(batch["images"].cuda())["cam"]), kernel(46,65, sigma=1.5))))
  model = ms.load_latest_model(main_dict)

  opt = ms.create_opt(model, main_dict)
  val_dict, pred_annList = au.validate(model, val_set, 
            predict_method="BestDice", 
            n_val=None, return_annList=True)

  
  ms.visBlobs(model, batch)
  ms.visPretty(model, batch, alpha=0.0)
  if ms.model_exists(main_dict) and main_dict["reset"] != "reset":
    model = ms.load_latest_model(main_dict)
    opt = ms.create_opt(model, main_dict)
    history = ms.load_history(main_dict)
    import ipdb; ipdb.set_trace()  # breakpoint 46fc0d2c //

    batch=ms.get_batch(val_set,indices=[2])
    model.visualize(batch, cam_index=1)
    model.embedding_head.score_8s.bias
    dice_scores = val.valPascal(model, val_set, 
                                  predict_method="BestDice", 
                                  n_val=[11])
    # vis.visBlobs(model,ms.get_batch(val_set,indices=[14]))
    # dice_scores = val.valPascal(model, val_set, 
    #                                 predict_method="BestDice", 
    #                                 n_val=[11])
    
    import ipdb; ipdb.set_trace()  # breakpoint 54f5496d //
    dice_scores = val.valPascal(model, val_set, 
                                    predict_method="BestDice", 
                                    n_val=[80,81])
    vis.visBlobList(model, val_set,[1,2,3])
    dice_scores = val.valPascal(model, val_set, 
                                          predict_method="BestDice", 
                                          n_val=len(val_set))

    obj_scores = val.valPascal(model, val_set, 
                                          predict_method="BestObjectness", 
                                          n_val=len(val_set))

    vis.visBlobs(model, batch)
    import ipdb; ipdb.set_trace()  # breakpoint cbf2e6d1 //
    ms.fitBatch(model, batch, loss_function=loss_dict[loss_name], 
                opt=opt, epochs=10000, visualize=True)
    dice_scores = val.valPascal(model, val_set, 
                                    predict_method="BestDice", 
                                    n_val=[630, 631, 632])

    obj_scores = val.valPascal(model, val_set, 
                              predict_method="BestObjectness", 
                              n_val=100)

    dice_scores = val.valPascal(model, val_set, 
                              predict_method="BestDice", 
                              n_val=len(val_set))
    # val.valPascal(model, val_set, 
    #                     predict_method="BestObjectness", 
    #                     n_val=[10])

    model.predict(batch, predict_method="BestDice")
    import ipdb; ipdb.set_trace()  # breakpoint 797d17b4 //
    
    obj_scores = val.valPascal(model, val_set, 
                              predict_method="BestObjectness", 
                              n_val=30)
    vis.visBlobs(model, ms.get_batch(val_set,indices=[14]), 
        predict_method="BestDice")
    model.predict(batch, predict_method="blobs")
    import ipdb; ipdb.set_trace()  # breakpoint f4598264 //

    model.visaulize(batch)
    

    

    val.valPascal(model, val_set, 
                        predict_method="BestObjectness", 
                        n_val=[10])
    vis.visBlobs(model, batch, 
              predict_method="BestObjectness")
    import ipdb; ipdb.set_trace()  # breakpoint f691d432 //
    ms.fit(model, ms.get_dataloader(val_set, batch_size=1, sampler_class=None),
      opt=opt, loss_function=main_dict["loss_dict"][loss_name])  
    vis.visBlobs(model, ms.get_batch(val_set,indices=[1]), 
              predict_method="UpperBound")

    history = ms.load_history(main_dict)
    # model = ms.load_best_model(main_dict)
    
    #print("Loaded best model...")
    
  else:
    model, opt, _ = ms.init_model_and_opt(main_dict)
  import ipdb; ipdb.set_trace()  # breakpoint e26f9978 //
  
  obj_scores = val.valPascal(model, val_set, 
                              predict_method="BestObjectness", 
                              n_val=[2])
  # ms.images(batch["images"], model.predict(batch, "blobs"), denorm=1)
  import ipdb; ipdb.set_trace()  # breakpoint 08a2a8af //
  vis.visBlobs(model, batch)
  vis.visBlobs(model,ms.get_batch(val_set,indices=[14]))
  ms.fitBatch(model, batch, loss_function=loss_dict[loss_name], 
                opt=opt, epochs=10000, visualize=True)
  ms.fitBatch(model, batch, loss_function=loss_dict[loss_name], 
              opt=opt, epochs=10, visualize=True)

  #test_prm(model, batch)
  # test_prm(model, batch, i=1, j=0)
  # import ipdb; ipdb.set_trace()  # breakpoint a860544a //
  # img2 = batch["images"].cuda().requires_grad_()
  # cues=rm.peak_response(model.backbone, img, peak_threshold=1)
  # batch = ms.get_batch(train_set,indices=[0])
  # vis.visBatch(ms.get_batch(train_set,indices=[72]))
  #vis.visBlobs(model, batch)
  #ms.images(batch["images"], batch["points"], denorm=1, enlarge=1)
  # vis.visSplit(model, batch)
  #model.set_proposal(None); vis.visBlobs(model, batch)
  # vis.visBlobs(model, batch)
  #vis.visBlobList(model, val_set, [0, 1,2,3])
  # for i in range(len(train_set)): print(i);x=train_set[i]
  # vis.visBlobs(model, batch)
  
  '''
  mask = np.zeros(batch["images"].shape)[:,0]
  ms.images(batch["images"], mask, denorm=1)
  for i in range(400):
    mask +=  (i+1)*(rescale(sharp_proposals[i]["mask"],0.5)>0).astype(int)
  annList = vis.visAnnList(model, val_set, [34], cocoGt,  
                 predict_proposal="BestObjectness") 
  ''' 
  n_images = 10
  batch = ms.get_batch(val_set,indices=[9])
  import ipdb; ipdb.set_trace()  # breakpoint 8d385ace //
  batch = ms.get_batch(val_set,indices=[50])
  
  ms.fitBatch(model, batch, loss_function=loss_dict[loss_name], 
              opt=opt, epochs=10)
  vis.visBlobs(model, ms.get_batch(val_set,indices=[3]), predict_method="BestDice")
  vis.visBlobs(model, batch)
  import ipdb; ipdb.set_trace()  # breakpoint 99558393 //
  val.valPascal(model, val_set, 
                        predict_method="BestObjectness", 
                        n_val=10)
  val.valPascal(model, val_set, 
                        predict_method="BestDice", 
                        n_val=10)
  val.valPascal(model, val_set, 
                        predict_method="BestDice_no", 
                        n_val=[10])
  batch = ms.get_batch(val_set,indices=[10])
  model.predict(batch, predict_method="BestDice")
  model.predict(batch, predict_method="BestDice_no")
  vis.visBlobs(model, batch)
  vis.visBlobs(model, batch, predict_method=main_dict["predict_name"], cocoGt=val_set.cocoGt)

  val.valPascal(model, val_set, 
                  predict_method="BestObjectness", 
                  n_val=15)
  val.valPascal(model, val_set, 
                  predict_method="BoxSegment", 
                  n_val=15)

  val.valPascal(model, val_set, 
                  predict_method=main_dict["predict_name"], 
                  n_val=15)
  
  vis.visBlobs(model, batch)
  ms.fitBatch(model, batch, 
              loss_function=loss_dict[main_dict["loss_name"]], 
              opt=opt, epochs=5)
  vis.visBlobs(model, batch)
  ms.images(bu.batch2propDict(ms.get_batch(val_set,indices=[1]))["foreground"])
  batch = ms.get_batch(val_set,indices=[19]);ms.images(batch["images"],bu.batch2propDict(batch)["foreground"],denorm=1)
  ms.fitBatch(model, batch, 
              loss_function=loss_dict[main_dict["loss_name"]], 
              opt=opt, epochs=100)
  vis.visBlobs(model, ms.get_batch(val_set,indices=[1]), 
              predict_method="GlanceBestBox")
  val.valPascal(model, val_set, 
                  predict_method="GlanceBestBox", 
                  n_val=15)

  val.valPascal(model, val_set, 
                  predict_method="BestDice", 
                  n_val=15)


  import ipdb; ipdb.set_trace()  # breakpoint 01f8e3fa //

  val.valPascal(model, val_set, 
                  predict_method=main_dict["predict_name"], 
                  n_val=5)

  import ipdb; ipdb.set_trace()  # breakpoint 78d3f03a //
  vis.visBlobs(model, ms.get_batch(val_set,indices=[1]))
  vis.visBlobs(model, ms.get_batch(val_set,indices=[1]), 
              predict_method="BestObjectness")
  vis.visBlobs(model, ms.get_batch(val_set,indices=[1]), 
              predict_method="UpperBound")
  ms.fitBatch(model, batch, loss_function=loss_dict[main_dict["loss_name"]], 
              opt=opt, epochs=100)
  # ms.fitBatch(model, batch, loss_function=loss_dict[loss_name], 
                # opt=opt, epochs=1)
  # ms.fitData(model, val_set,opt=opt, loss_function=loss_dict[loss_name])
  import ipdb; ipdb.set_trace()  # breakpoint 51e4d47d //

  val.valPascal(model, val_set, 
                  predict_method="BestObjectness", 
                  n_val=n_images)
  val.valPascal(model, val_set, 
                  predict_method="UpperBound", 
                  n_val=len(val_set))
  # vis.visBlobs(model, ms.get_batch(val_set,indices=[1]))
  vis.visBlobs(model, ms.get_batch(val_set,indices=[1]))
  vis.visBlobs(model, ms.get_batch(val_set,indices=[1]), 
              predict_method="BestObjectness")

  n_images = len(val_set)
  for e in range(5):
    for i in range(n_images):
        i_rand = np.random.randint(n_images) 
        i_rand = i
        # print
        print(i_rand)
        batch = ms.get_batch(train_set,indices=[i_rand])
        ms.fitBatch(model, batch, loss_function=loss_dict[loss_name], 
                    opt=opt, epochs=1)

  #cocoGt = ms.load_voc2012_val()
  cocoGt = ms.load_cp_val()
  ms.fitBatch(model, batch, loss_function=loss_dict[main_dict["loss_name"]], 
              opt=opt, epochs=100)

  # vis.visAnns(model, batch, cocoGt, predict_proposal="BestBoundary")
  import ipdb; ipdb.set_trace()  # breakpoint 6f37a744 //
  if 1:
    n_images = 30
    resList = []
    for k in range(5):
      for i in range(n_images):
        print(i)
        batch = ms.get_batch(val_set,indices=[i])
        ms.fitBatch(model, batch, loss_function=loss_dict[loss_name], 
                    opt=opt, epochs=2)

      resList +=[val.valPascal(model, val_set, 
                  predict_proposal="excitementInside", 
                  n_val=n_images)]
# excitementInside
  ms.fitBatch(model, batch, loss_function=loss_dict[main_dict["loss_name"]], 
              opt=opt, epochs=100)
  
  import ipdb; ipdb.set_trace()  # breakpoint 14451165 //

  ms.eval_cocoDt(main_dict, predict_proposal="UB_Sharp_withoutVoid")
  import ipdb; ipdb.set_trace()  # breakpoint f3f0fda5 //
  
  vis.visAnns(model, batch, cocoGt, predict_proposal="BestObjectness")
  annList = vis.visAnnList(model, val_set, [1,2], cocoGt, 
    predict_proposal="BestObjectness")


  annList = ms.load_annList(main_dict, predict_proposal="BestObjectness")
  ms.eval_cocoDt(main_dict, predict_proposal="UB_Sharp_withoutVoid")
  # score = np.array([s["score"] for s in annList])
  batch = ms.get_batch(val_set,indices=[2])
  ms.fitBatch(model, batch, loss_function=loss_dict[main_dict["loss_name"]], 
              opt=opt, epochs=100)

  vis.visBlobs(model, batch)

  ms.fitBatch(model, batch, loss_function=loss_dict["water_loss"], 
              opt=opt, epochs=100)
  ms.fitBatch(model, batch, loss_function=loss_dict["point_loss"], 
              opt=opt, epochs=100)
  vis.visSplit(model, batch, 0,"water")

  


  '''
  val.valPascal(model, val_set, 
                    predict_proposal="excitementInside", 
                    n_val=30)

  '''
  # model.save(batch, path="/mnt/home/issam/Summaries/tmp.png")
  # batch = ms.get_batch(train_set,indices=[52])
  # torch.save(model.state_dict(), "/mnt/home/issam/Saves/model_split.pth")
  vis.save_images(model, val_set, 
                  #indices=np.random.randint(0, len(val_set), 200),
                  indices=np.arange(5,200),
                  path="/mnt/home/issam/Summaries/{}_val/".format(main_dict["dataset_name"]))

  vis.visBlobs(model, batch)
  ms.fitBatch(model, batch, loss_function=loss_dict[loss_name], 
              opt=opt, epochs=10)

  ms.valBatch(model, batch, metric_dict[metric_name])
  ms.validate(model, val_set, metric_class=metric_class)
  # ms.visBlobs(model, tr_batch)
  # model.predict(tr_batch,"counts")

  for i in range(292, 784):
    batch = ms.get_batch(val_set, indices=[i])
    try:
      score = ms.valBatch(model,  batch, 
            metric_dict[metric_name]) 
    except:
      print(i, batch['name']) 
  import ipdb; ipdb.set_trace()  # breakpoint effaca86 //
  
  ms.visBlobs(model, batch)
  if 1:
    resList = []
    for k in range(5):
      for i in range(10):
        print(i)
        batch = ms.get_batch(val_set,indices=[i])
        ms.fitBatch(model, batch, loss_function=loss_dict[loss_name], 
                    opt=opt, epochs=1)

      resList +=[val.valPascal(model, val_set, 
                  predict_proposal="BestObjectness", 
                  n_val=10)]

  val.valPascal(model, val_set, 
              predict_proposal="BestBoundary", 
              n_val=30)
  
  val.valPascal(model, val_set, 
              predict_proposal="BestObjectness", 
              n_val=list(range(len(val_set))))

  #model.predict_proposals(batch)
  batch = ms.get_batch(val_set,indices=[35])
  ms.images(batch["original"], model.predict_proposals(batch, which=0))
  
  ms.images(ms.get_batch(train_set, [300])["original"],
            train_set.get_proposal(300, indices=[0,1]))

  # from spn import object_localization
  #cm = model.class_activation_map(batch["images"].cuda())
  # model.display(ms.get_batch(train_set,indices=[3]))
  # ms.
  ms.images(255*np.abs( model.predict(ms.get_batch(train_set,indices=[3]), "saliency")))
  sal = model.predict(ms.get_batch(train_set,indices=[3]), "saliency")
  ms.images(np.abs(sal)*255)
  import ipdb; ipdb.set_trace()  # breakpoint c7ca398d //

  for i in range(1):
    ms.fit(model, ms.get_dataloader(train_set, batch_size=1, sampler_class=None), 
        loss_function=main_dict["loss_dict"][loss_name], metric_class=main_dict["metric_dict"][metric_name],
                  opt=opt, val_batch=False)
  ms.fitQuick(model, train_set, loss_name=loss_name, metric_name=metric_name,opt=opt)
  ms.fitBatch(model, batch, loss_function=loss_dict[loss_name], opt=opt, epochs=100)
  ms.valBatch(model, batch, metric_dict[metric_name])
  ms.visBlobs(model, ms.get_batch(train_set, indices=[3]) )
  ms.visBlobs(model, batch)
  #model = ms.load_best_model(main_dict)
  #metrics.compute_ap(model, batch)
  #val.val_cm(main_dict)
  batch = ms.visBlobsQ(model, val_set, 8)
  
  import ipdb; ipdb.set_trace()  # breakpoint 5cd16f8f //
  ul.visSp_prob(model, batch)
  
  3
  ms.images(batch["images"], aa, denorm=1)

  ms.visBlobs(model, batch)
  ul.vis_nei(model,batch,topk=1000, thresh=0.8,bg=True)
  ul.vis_nei(model,batch,topk=1000, bg=False)
  ms.fitQuick(model, train_set, batch_size=batch_size,loss_name=loss_name, metric_name=metric_name)
  val.validate(model, val_set, metric_name=main_dict["metric_name"], batch_size=main_dict["val_batchsize"])
  ms.fitQuick(model, train_set, batch_size=batch_size,loss_name=loss_name, metric_name=metric_name)
  ms.fitBatch(model, batch, loss_name=loss_name, opt=opt, epochs=100)
  val.valBatch(model, batch_train, metric_name=metric_name)
  ms.fitBatch(model, batch, loss_function=losses.expand_loss, opt=opt, epochs=100)
  ms.visBlobs(model, batch)
  ms.visWater(model,batch)
  ms.validate(model, val_set, metric_class=metric_class)
  import ipdb; ipdb.set_trace()  # breakpoint ddad840d //
  model, opt, _ = ms.init_model_and_opt(main_dict)
  ms.fitBatch(model, batch, loss_name="water_loss_B", opt=opt, epochs=100)

  ms.fitQuick(model, train_set, loss_name=loss_name, metric_name=metric_name)
  # ms.images(batch["images"], batch["labels"], denorm=1)
  # ms.init.LOSS_DICT["water_loss"](model, batch)
  import ipdb; ipdb.set_trace()  # breakpoint f304b83a //
  ms.images(batch["images"], model.predict(batch, "labels"), denorm=1)
  val.valBatch(model, batch, metric_name=main_dict["metric_name"])
  ms.visBlobs(model, batch)
  import ipdb; ipdb.set_trace()  # breakpoint 074c3921 //

  ms.fitBatch(model, batch, loss_name=main_dict["loss_name"], opt=opt, epochs=100)
  for e in range(10):
    if e == 0:
      scoreList = []
    scoreList += [ms.fitIndices(model, train_set, loss_name=main_dict["loss_name"], batch_size=batch_size,
      metric_name=metric_name, opt=opt, epoch=e, num_workers=1, 
      ind=np.random.randint(0, len(train_set), 32))]
  ms.fitData(model, train_set, opt=opt, epochs=10)
  um.reload(sp);water=sp.watersplit(model, batch).astype(int);ms.images(batch["images"], water, denorm=1)
  ms.visBlobs(model, batch)
  ms.images(batch["images"], ul.split_crf(model, batch),denorm=1)
  losses.dense_crf(model, batch, alpha=61, beta=31, gamma=1)
  
  ms.visBlobs(model, batch)

  model.blob_mode = "superpixels"
  #----------------------

  # Vis Blobs
  ms.visBlobs(model, batch)
  ms.images(batch["images"],model.predict(batch, "labels"), denorm=1)

  # Vis Blobs
  #ms.visBlobs(model, batch)
  ms.images(batch["images"], sp.watersplit_test(model, batch).astype(int), denorm=1)

  #=sp.watersplit(model, batch).astype(int);

  # Vis CRF
  ms.images(batch["images"], ul.dense_crf(model, batch, alpha=5,gamma=5,beta=5,smooth=False), denorm=1)
  ms.images(batch["images"], ul.dense_crf(model, batch), denorm=1)
  # Eval
  val.valBatch(model, batch, metric_name=main_dict["metric_name"])

  import ipdb; ipdb.set_trace()  # breakpoint e9cd4eb0 //
  model = ms.load_best_model(main_dict)

  val.valBatch(model, batch, metric_name=main_dict["metric_name"])
  ms.fitBatch(model, batch, loss_name=main_dict["loss_name"], opt=opt)
  ms.visBlobs(model, batch)
  import ipdb; ipdb.set_trace()  # breakpoint 2167961a //
  batch=ms.get_batch(train_set, indices=[5]) 
  ms.fitBatch(model, batch, loss_name=main_dict["loss_name"], opt=opt)
  ms.images(batch["images"], model.predict(batch, "probs"), denorm=1)

  ms.visBlobs(model, batch)
  val.validate(model, val_set, metric_name=main_dict["metric_name"])
  val.validate(model, val_set, metric_name="SBD")



def plot_density():
  import pylab as plt
  import numpy as np

  # Sample data
  side = np.linspace(-2,2,15)
  X,Y = np.meshgrid(side,side)
  Z = np.exp(-((X-1)**2+Y**2))
  fig = plt.figure()
  # Plot the density map using nearest-neighbor interpolation
  plt.pcolormesh(X,Y,Z)
  ms.visplot(fig)