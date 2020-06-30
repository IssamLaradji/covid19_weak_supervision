import matplotlib
matplotlib.use('Agg')
from addons.pycocotools.coco import COCO
import torch
import pandas as pd
import argparse
import numpy as np
from itertools import product
import experiments
import os 
from glob import glob
# from core import dataset2cocoformat
import misc as ms
import borgy
import test 
from collections import defaultdict
import train
import debug
import configs
# import train_lcfcn
# from core import data_utils
import ann_utils as au
import validate
from addons import vis


def main():
  parser = argparse.ArgumentParser()

  parser.add_argument('-e','--exp') 
  parser.add_argument('-b','--borgy', default=0, type=int)
  parser.add_argument('-br','--borgy_running', default=0, type=int)
  parser.add_argument('-m','--mode', default="summary")
  parser.add_argument('-r','--reset', default="None")
  parser.add_argument('-s','--status', type=int, default=0)
  parser.add_argument('-k','--kill', type=int, default=0)
  parser.add_argument('-g','--gpu', type=int)
  parser.add_argument('-c','--configList', nargs="+",
                      default=None)
  parser.add_argument('-l','--lossList', nargs="+",
                      default=None)
  parser.add_argument('-d','--datasetList', nargs="+",
                      default=None)
  parser.add_argument('-metric','--metricList', nargs="+",
                      default=None)
  parser.add_argument('-model','--modelList', nargs="+",
                      default=None)
  parser.add_argument('-p','--predictList', nargs="+",
                      default=None)

  args = parser.parse_args()

  if args.borgy or args.kill:
    global_prompt = input("Do all? \n(y/n)\n") 

  # SEE IF CUDA IS AVAILABLE
  assert torch.cuda.is_available()
  print("CUDA: %s" % torch.version.cuda)
  print("Pytroch: %s" % torch.__version__)


  mode = args.mode 
  exp_name = args.exp

  exp_dict = experiments.get_experiment_dict(args, exp_name)
  
  pp_main = None
  results = {}

  # Get Main Class
  project_name = os.path.realpath(__file__).split("/")[-2]
  MC = ms.MainClass(path_models="models",
                    path_datasets="datasets", 
                    path_metrics="metrics/metrics.py",
                    path_losses="losses/losses.py",
                    path_samplers="addons/samplers.py",
                    path_transforms="addons/transforms.py",
                    path_saves="/mnt/projects/counting/Saves/main/", 
                    project=project_name)

  
  key_set = set()
  for model_name, config_name, metric_name, dataset_name, loss_name in product(exp_dict["modelList"],
                                                         exp_dict["configList"], 
                                                         exp_dict["metricList"], 
                                                         exp_dict["datasetList"], 
                                                         exp_dict["lossList"]):     

      # if model_name in ["LC_RESFCN"]:
      #   loss_name = "water_loss"

      config = configs.get_config_dict(config_name)
      
      
      key = ("{} - {} - {}".format(model_name, config_name, loss_name), 
             "{}_({})".format(dataset_name, metric_name))

      if key in key_set:
        continue

      key_set.add(key)

      main_dict = MC.get_main_dict(mode, dataset_name, model_name,
                                   config_name, config, 
                                   args.reset, exp_dict["epochs"],
                                   metric_name, loss_name)
      main_dict["predictList"] = exp_dict["predictList"]
     
      if mode == "paths":
        print("\n{}_({})".format(dataset_name, model_name))
        print( main_dict["path_best_model"])
        # print( main_dict["exp_name"])

      predictList_str = ' '.join(exp_dict["predictList"])

      if args.status:        
        results[key] = borgy.borgy_status(mode, config_name, 
                            metric_name, model_name,
                            dataset_name, loss_name, args.reset,
                            predictList_str)

        continue

      if args.kill:   
        results[key] = borgy.borgy_kill(mode, config_name, 
                            metric_name, model_name,
                            dataset_name, loss_name, args.reset,
                            predictList_str)
        continue

      if args.borgy:
        results[key] = borgy.borgy_submit(project_name, global_prompt,
                            mode, config_name, 
                            metric_name, model_name,
                            dataset_name, loss_name, args.reset,
                            predictList_str)

        continue

      if mode == "debug":
        debug.debug(main_dict)

      if mode == "validate":
        validate.validate(main_dict)
      if mode == "save_gam_points":
        train_set, _ = au.load_trainval(main_dict)
        model = ms.load_best_model(main_dict)
        for i in range(len(train_set)):
          print(i, "/", len(train_set))
          batch = ms.get_batch(train_set, [i])
          fname = train_set.path + "/gam_{}.pkl".format(batch["index"].item()) 
          points = model.get_points(batch)
          ms.save_pkl(fname, points)
        import ipdb; ipdb.set_trace()  # breakpoint ee49ab9f //
        

      if mode == "save_prm_points":
        train_set, _ = au.load_trainval(main_dict)
        model = ms.load_best_model(main_dict)
        for i in range(len(train_set)):
          print(i, "/", len(train_set))
          batch = ms.get_batch(train_set, [i])
          
          fname = "{}/prm{}.pkl".format(batch["path"][0], batch["name"][0])           
          points = model.get_points(batch)
          ms.save_pkl(fname, points)
        import ipdb; ipdb.set_trace()  # breakpoint 679ce152 //

        # train_set, _ = au.load_trainval(main_dict)
        # model = ms.load_best_model(main_dict)
        # for i in range(len(train_set)):
        #   print(i, "/", len(train_set))
        #   batch = ms.get_batch(train_set, [i])
        #   fname = train_set.path + "/gam_{}.pkl".format(batch["index"].item()) 
        #   points = model.get_points(batch)
        #   ms.save_pkl(fname, points)

      # if mode == "pascal_annList":
      #   data_utils.pascal2lcfcn_points(main_dict)
      if mode == "upperboundmasks":
        import ipdb; ipdb.set_trace()  # breakpoint 02fac8ce //
        
        results = au.test_upperboundmasks(main_dict, reset=args.reset)
        print(pd.DataFrame(results))

      if mode == "model":

        results = au.test_model(main_dict, reset=args.reset)
        print(pd.DataFrame(results))

      if mode == "upperbound":
        results = au.test_upperbound(main_dict, reset=args.reset)

        print(pd.DataFrame(results))

      if mode == "MUCov":
        gtAnnDict = au.load_gtAnnDict(main_dict, reset=args.reset)

        # model = ms.load_best_model(main_dict)
        fname = main_dict["path_save"]+"/pred_annList.pkl"
        if not os.path.exists(fname):
          _, val_set = au.load_trainval(main_dict)
          model = ms.load_best_model(main_dict)
          pred_annList = au.dataset2annList(model, val_set, 
                            predict_method="BestDice", 
                            n_val=None)
          ms.save_pkl(fname, pred_annList)

        else:
          pred_annList = ms.load_pkl(fname)
        import ipdb; ipdb.set_trace()  # breakpoint 527a7f36 //
        pred_annList = au.load_predAnnList(main_dict, predict_method="BestObjectness")
        # 0.31 best objectness pred_annList = 
        # 0.3482122335421256
        # au.get_MUCov(gtAnnDict, pred_annList)
        au.get_SBD(gtAnnDict, pred_annList)

       

      if mode == "dic_sbd":
        import ipdb; ipdb.set_trace()  # breakpoint 4af08a17 //

      if mode == "point_mask":
        from datasets import base_dataset

        import ipdb; ipdb.set_trace()  # breakpoint 7fd55e0c //
        _, val_set = ms.load_trainval(main_dict)
        batch = ms.get_batch(val_set, [1])
        model = ms.load_best_model(main_dict)
        pred_dict = model.LCFCN.predict(batch)
        # ms.pretty_vis(batch["images"], base_dataset.batch2annList(batch))
        ms.images(ms.pretty_vis(batch["images"], model.LCFCN.predict(batch, predict_method="original")["annList"]), win="blobs")
        ms.images(ms.pretty_vis(batch["images"], base_dataset.batch2annList(batch)), win="erww")
        ms.images(batch["images"], batch["points"], denorm=1, enlarge=1, win="e21e")
        import ipdb; ipdb.set_trace()  # breakpoint ab9240f0 //

      if mode == "lcfcn_output":
        import ipdb; ipdb.set_trace()  # breakpoint 7fd55e0c //
        
        gtAnnDict = au.load_gtAnnDict(main_dict, reset=args.reset)

      if mode == "load_gtAnnDict":
        _, val_set = au.load_trainval(main_dict)
        gtAnnDict = au.load_gtAnnDict(val_set)

        # gtAnnClass = COCO(gtAnnDict)
        # au.assert_gtAnnDict(main_dict, reset=None)
        # _,val_set = au.load_trainval(main_dict)
        # annList_path = val_set.annList_path

        # fname_dummy = annList_path.replace(".json","_best.json")
        # predAnnDict = ms.load_json(fname_dummy)
        import ipdb; ipdb.set_trace()  # breakpoint 100bfe1b //
        pred_annList = ms.load_pkl(main_dict["path_best_annList"])
        # model = ms.load_best_model(main_dict)
        _, val_set = au.load_trainval(main_dict)
        batch = ms.get_batch(val_set, [1])

        import ipdb; ipdb.set_trace()  # breakpoint 2310bb33 //
        model = ms.load_best_model(main_dict)
        pred_dict = model.predict(batch, "BestDice", "mcg")
        ms.images(batch["images"], au.annList2mask(pred_dict["annList"])["mask"], denorm=1)
        # pointList2UpperBoundMCG
        pred_annList = au.load_predAnnList(main_dict, predict_method="BestDice", 
                                          proposal_type="mcg",
                                            reset="reset")
        # annList = au.pointList2UpperBoundMCG(batch["lcfcn_pointList"], batch)["annList"]
        ms.images(batch["images"], au.annList2mask(annList)["mask"], denorm=1)
        pred_annList = au.load_BestMCG(main_dict, reset="reset")
        # pred_annList = au.dataset2annList(model, val_set, 
        #                   predict_method="BestDice", 
        #                   n_val=None)
        au.get_perSizeResults(gtAnnDict, pred_annList)

      if mode == "vis":
        _, val_set = au.load_trainval(main_dict)
        batch = ms.get_batch(val_set, [3])

        import ipdb; ipdb.set_trace()  # breakpoint 05e6ef16 //
        
        vis.visBaselines(batch)

        model = ms.load_best_model(main_dict)
        vis.visBlobs(model, batch)


      if mode == "qual":
        model = ms.load_best_model(main_dict)
        _, val_set = au.load_trainval(main_dict)
        path = "/mnt/home/issam/Summaries/{}_{}".format(dataset_name, model_name)
        try:
          ms.remove_dir(path)
        except:
          pass
        n_images = len(val_set)
        base = "{}_{}".format(dataset_name, model_name)
        for i in range(50):
          print(i,"/10", "- ", base)
          index = np.random.randint(0, n_images)
          batch = ms.get_batch(val_set, [index])
          if len(batch["lcfcn_pointList"]) == 0:
            continue
          image = vis.visBlobs(model, batch, return_image=True)

          # image_baselines = vis.visBaselines(batch, return_image=True)
          # imgAll = np.concatenate([image, image_baselines], axis=1)

          fname = path + "/{}_{}.png".format(i, base)
          ms.create_dirs(fname)
          ms.imsave(fname, image)


      if mode == "test_baselines":
        import ipdb; ipdb.set_trace()  # breakpoint b51c5b1f //
        results = au.test_baselines(main_dict, reset=args.reset)
        print(pd.DataFrame(results))

      if mode == "test_best":
        au.test_best(main_dict)

      if mode == "qualitative":
        au.qualitative(main_dict)

      if mode == "figure1":
        from PIL import Image
        from addons import transforms
        model = ms.load_best_model(main_dict)
        _, val_set = au.load_trainval(main_dict)
        # proposals_path = "/mnt/datasets/public/issam/Cityscapes/demoVideo/leftImg8bit/demoVideo/ProposalsSharp/"
        # vidList = glob("/mnt/datasets/public/issam/Cityscapes/demoVideo/leftImg8bit/demoVideo/stuttgart_01/*")
        # vidList.sort()

          # pretty_image = ms.visPretty(model, batch = ms.get_batch(val_set, [i]), with_void=1, win="with_void")
        batch = ms.get_batch(val_set, [68])
        bestdice = ms.visPretty(model, batch = batch, with_void=0, win="no_void")
        blobs = ms.visPretty(model, batch = batch,predict_method="blobs", with_void=0, win="no_void")

        ms.images(bestdice, win="BestDice")
        ms.images(blobs, win="Blobs")
        ms.images(batch["images"], denorm=1, win="Image")
        ms.images(batch["images"], batch["points"], enlarge=1, denorm=1, win="Points")
        import ipdb; ipdb.set_trace()  # breakpoint cf4bb3d3 //

      if mode == "video2":
        from PIL import Image
        from addons import transforms
        model = ms.load_best_model(main_dict)
        _, val_set = au.load_trainval(main_dict)
        # proposals_path = "/mnt/datasets/public/issam/Cityscapes/demoVideo/leftImg8bit/demoVideo/ProposalsSharp/"
        # vidList = glob("/mnt/datasets/public/issam/Cityscapes/demoVideo/leftImg8bit/demoVideo/stuttgart_01/*")
        # vidList.sort()
        index = 0
        for i in range(len(val_set)):
          

          # pretty_image = ms.visPretty(model, batch = ms.get_batch(val_set, [i]), with_void=1, win="with_void")
          batch = ms.get_batch(val_set, [i])
          pretty_image = ms.visPretty(model, batch = batch, with_void=0, win="no_void")
          # pred_dict = model.predict(batch, predict_method="BestDice")
          path_summary = main_dict["path_summary"]
          ms.create_dirs(path_summary+"/tmp")
          ms.imsave(path_summary+"vid_mask_{}.png".format(index), ms.get_image(batch["images"],batch["points"], enlarge=1,denorm=1))
          index+=1
          ms.imsave(path_summary+"vid_mask_{}.png".format(index), pretty_image)
          index += 1
          # ms.imsave(path_summary+"vid1_full_{}.png".format(i), ms.get_image(img, pred_dict["blobs"], denorm=1))
          print(i, "/", len(val_set))


      if mode == "video":
        from PIL import Image
        from addons import transforms
        model = ms.load_best_model(main_dict)
        # _, val_set = au.load_trainval(main_dict)
        proposals_path = "/mnt/datasets/public/issam/Cityscapes/demoVideo/leftImg8bit/demoVideo/ProposalsSharp/"
        vidList = glob("/mnt/datasets/public/issam/Cityscapes/demoVideo/leftImg8bit/demoVideo/stuttgart_01/*")
        vidList.sort()
        for i, img_path in enumerate(vidList):
          image = Image.open(img_path).convert('RGB')
          image = image.resize((1200, 600),Image.BILINEAR)
          img, _ = transforms.Tr_WTP_NoFlip()([image,image])

          
          pred_dict = model.predict({"images":img[None], "split":["test"],
            "resized":torch.FloatTensor([1]), 
                               "name":[ms.extract_fname(img_path)],
                               "proposals_path":[proposals_path]}, predict_method="BestDice")
          path_summary = main_dict["path_summary"]
          ms.create_dirs(path_summary+"/tmp")
          ms.imsave(path_summary+"vid1_mask_{}.png".format(i), ms.get_image(pred_dict["blobs"]))
          ms.imsave(path_summary+"vid1_full_{}.png".format(i), ms.get_image(img, pred_dict["blobs"], denorm=1))
          print(i, "/", len(vidList))

      if mode == "5_eval_BestDice":
        gtAnnDict = au.load_gtAnnDict(main_dict)
        gtAnnClass = COCO(gtAnnDict)
        results = au.assert_gtAnnDict(main_dict, reset=None)


      if mode == "cp_annList":
        ms.dataset2cocoformat(dataset_name="CityScapes")


      if mode == "pascal2lcfcn_points":
        data_utils.pascal2lcfcn_points(main_dict)

      if mode == "cp2lcfcn_points":
        data_utils.cp2lcfcn_points(main_dict)

      if mode == "train":

        train.main(main_dict)
        import ipdb; ipdb.set_trace()  # breakpoint a5d091b9 //


      if mode == "train_only":

        train.main(main_dict, train_only=True)
        import ipdb; ipdb.set_trace()  # breakpoint a5d091b9 //


  

      if mode == "sharpmask2psfcn":
        for split in ["train", "val"]:
          root = "/mnt/datasets/public/issam/COCO2014/ProposalsSharp/"
          path = "{}/sharpmask/{}/jsons/".format(root, split)

          jsons = glob(path+"*.json")
          propDict = {}
          for k,json in enumerate(jsons):
            print("{}/{}".format(k, len(jsons)))
            props = ms.load_json(json)
            for p in props:
              if p["image_id"] not in propDict:
                propDict[p["image_id"]] = []
              propDict[p["image_id"]] += [p]

          for k in propDict.keys():
            fname = "{}/{}.json".format(root, k)
            ms.save_json(fname, propDict[k])
          




      if mode == "cp2coco":
        import ipdb; ipdb.set_trace()  # breakpoint f2eb9e70 //
        dataset2cocoformat.cityscapes2cocoformat(main_dict)
        # train.main(main_dict)
        import ipdb; ipdb.set_trace()  # breakpoint a5d091b9 //


      if mode == "train_lcfcn":
        train_lcfcn.main(main_dict)
        import ipdb; ipdb.set_trace()  # breakpoint a5d091b9 //


      if mode == "summary":
        
        try:
            history = ms.load_history(main_dict)

            # if predictList_str == "MAE":          
            #   results[key] = "{}/{}: {:.2f}".format(history["best_model"]["epoch"], 
            #                                                           history["epoch"], 
            #                                                           history["best_model"][metric_name])

            # else:
            val_dict = history["val"][-1]
            val_dict = history["best_model"]
            iou25 = val_dict["0.25"]
            iou5 = val_dict["0.5"]
            iou75 = val_dict["0.75"]
            results[key] = "{}/{}: {:.1f} - {:.1f} - {:.1f}".format(val_dict["epoch"], 
                                                                    history["epoch"], 
                                                                    iou25*100, 
                                                                    iou5*100, 
                                                                    iou75*100)
            # if history["val"][-1]["epoch"] != history["epoch"]:
            #   results[key] += " | Val {}".format(history["epoch"])
            try:
              results[key] += " | {}/{}".format(len(history["trained_batch_names"]), 
                                                          history["train"][-1]["n_samples"])
            except:
              pass
        except:
            pass
      if mode == "vals":

        history = ms.load_history(main_dict)
        
        for i in range(1, len(main_dict["predictList"])+1):
          if len(history['val']) == 0:
            res ="NaN"
            continue
          else:
            res = history["val"][-i]
          
          map50 = res["map50"]
          map75 =  res["map75"]

          # if map75 < 1e-3:
          #   continue
          
          string = "{} - {} - map50: {:.2f} - map75: {:.2f}".format(res["epoch"], res["predict_name"], map50, map75)
          
          key_tmp = list(key).copy()
          key_tmp[1] += " {} - {}".format(metric_name, res["predict_name"])
          results[tuple(key_tmp)] = string

        # print("map75", pd.DataFrame(history["val"])["map75"].max())
        # df = pd.DataFrame(history["vals"][:20])["water_loss_B"]
        # print(df)
  try:
    print(ms.dict2frame(results))
  except:
    print("Results not printed...")
      # TEST CASES



        
        # path = exp_dict["summary_path"]
        # pp_main.fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        # figName = "%s/png_plots/SRC_%s.png" % (path, exp_name)
        # ms.create_dirs(figName)
        # pp_main.fig.savefig(figName)

        # pp_main.fig.tight_layout()
        # pp_main.fig.suptitle("")

        # figName = "%s/pdf_plots/SRC_%s.pdf" % (path, exp_name)
        # ms.create_dirs(figName)
        # pp_main.fig.savefig(figName, dpi = 600)

        # print("saved {}".format(figName))
          

if __name__ == "__main__":
    main()