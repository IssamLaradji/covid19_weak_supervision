import os 

def get_experiments(args, exp_name):
    exp_dict = get_experiment_dict(args, exp_name)
  
    pp_main = None
    results = {}

    # Get Main Class
    project_name = os.path.realpath(__file__).split("/")[-2]
    MC = MainClass(path_models="models",
                    path_datasets="datasets", 
                    path_metrics="metrics/metrics.py",
                    path_losses="losses/losses.py",
                    path_samplers="addons/samplers.py",
                    path_transforms="addons/transforms.py",
                    path_saves="/mnt/projects/counting/Saves/main/", 
                    project=project_name)

    expList = []
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

        expList += [main_dict]

    return expList
           
def get_experiment_dict(args, exp_name):
        exp_dict = {"modelList":None,
                    "configList": None,
                    "lossList": None,
                    "datasetList": None,
                    "metricList": None,

                    "epochs":1000}

        if exp_name == "pascal":
            exp_dict = {"modelList":["MaskRCNN"],
                        "configList": ["smallPascal"],
                        "lossList": ["MaskRCNNLoss_small"],
                        "datasetList": ["PascalSmall"],
                        "metricList": ["MAE"],
                        "predictList": ["BestDice"],
                        "epochs":1000}

        if exp_name == "maskrcnn_prm":
            exp_dict = {"modelList":["MaskRCNN"],
                        "configList": ["noFlip"],
                        "lossList": ["MaskRCNNLoss_prm"],
                        "datasetList": ["Pascal2012"],
                        "metricList": ["MAE"],
                        "predictList": ["BestDice"],
                        "epochs":1000}


        if exp_name == "prm":
            exp_dict = {"modelList":["PRM"],
                        "configList": ["noFlip"],
                        "lossList": ["PRMLoss"],
                        "datasetList": ["Pascal2012"],
                        "metricList": ["MAE"],
                        "predictList": ["MAE"],
                        "epochs":1000}


        if exp_name == "gam2":
            exp_dict = {"modelList":["GAM_DISC"],
                        "configList": ["basic"],
                        "lossList": ["GAMDiscOnly"],
                        "datasetList": ["Trancos"],
                        "metricList": ["MAE"],
                        "predictList": ["MAE"],
                        "epochs":1000}

        if exp_name == "lcfcn_gam":
            exp_dict = {"modelList":["Res50FCN"],
                        "configList": ["noFlip2"],
                        "lossList": ["WeaklyLCFCN_Loss"],
                        "datasetList": ["Trancos"],
                        "metricList": ["MAE"],
                        "predictList": ["MAE"],
                        "epochs":1000}

        if exp_name == "gam":
            exp_dict = {"modelList":["GAM"],
                        "configList": ["basic"],
                        "lossList": ["GAMLoss", "GAMLoss2"],
                        "datasetList": ["Trancos"],
                        "metricList": ["MAE"],
                        "predictList": ["MAE"],
                        "epochs":1000}

        if exp_name == "maskrcnn":
            exp_dict = {"modelList":["MaskRCNN"],
                        "configList": ["noFlip"],
                        "lossList": ["MaskRCNNLoss"],
                        "datasetList": ["Pascal2012"],
                        "metricList": ["MAE"],
                        "predictList": ["BestDice"],
                        "epochs":1000}


        if exp_name == "aff":
            exp_dict = {"modelList":["AFF"],
                        "configList": ["noFlip"],
                        "lossList": ["AFFLoss"],
                        "datasetList": ["Pascal2012"],
                        "metricList": ["MAE"],
                        "predictList": ["MAE"],
                        "epochs":1000}

        if exp_name == "cam_como":
            exp_dict = {"modelList":["CAM"],
                        "configList": ["noFlip"],
                        "lossList": ["CAMLoss"],
                        "datasetList": ["ComoLake"],
                        "metricList": ["MAE"],
                        "predictList": ["MAE"],
                        "epochs":1000}

        if exp_name == "cam":
            exp_dict = {"modelList":["CAM"],
                        "configList": ["noFlip"],
                        "lossList": ["CAMLoss"],
                        "datasetList": ["Pascal2012"],
                        "metricList": ["MAE"],
                        "predictList": ["MAE"],
                        "epochs":1000}

        if exp_name == "OneHead_pascal":
            exp_dict = {"modelList":["OneHead"],
                        "configList": ["noFlip"],
                        "lossList": ["OneHeadLoss"],
                        "datasetList": ["Pascal2012"],
                        "metricList": ["MAE"],
                        "predictList": ["BestDice"],
                        "epochs":1000}

        if exp_name == "OneHeadLoc_pascal":
            exp_dict = {"modelList":["OneHeadLoc"],
                        "configList": ["noFlip"],
                        "lossList": ["OneHeadLoss"],
                        "datasetList": ["Pascal2012"],
                        "metricList": ["MAE"],
                        "predictList": ["BestDice"],
                        "epochs":1000}

        if exp_name == "OneHeadRandom_pascal":
            exp_dict = {"modelList":["OneHead"],
                        "configList": ["noFlip"],
                        "lossList": ["OneHeadRandomLoss"],
                        "datasetList": ["Pascal2012"],
                        "metricList": ["MAE"],
                        "predictList": ["BestDice"],
                        "epochs":1000}

        if exp_name == "OneHead_cpAll":
            exp_dict = {"modelList":["OneHeadLoc", "OneHead"],
                        "configList": ["noFlip"],
                        "lossList": ["OneHeadLoss"],
                        "datasetList": ["CityScapesAll"],
                        "metricList": ["MAE"],
                        "predictList": ["BestDice"],
                        "epochs":1000}

        if exp_name == "OneHead_cp":
            exp_dict = {"modelList":["OneHead", "OneHeadLoc"],
                        "configList": ["noFlip"],
                        "lossList": ["OneHeadLoss"],
                        "datasetList": ["CityScapes"],
                        "metricList": ["MAE"],
                        "predictList": ["BestDice"],
                        "epochs":1000}

        if exp_name == "OneHead_kitti":
            exp_dict = {"modelList":["OneHeadLoc"],
                        "configList": ["noFlip"],
                        "lossList": ["OneHeadLoss"],
                        "datasetList": ["Kitti"],
                        "metricList": ["MAE"],
                        "predictList": ["BestDice"],
                        "epochs":1000}

        if exp_name == "OneHead_plants":
            exp_dict = {"modelList":["OneHead","OneHeadLoc"],
                        "configList": ["noFlip"],
                        "lossList": ["OneHeadLoss"],
                        "datasetList": ["Plants"],
                        "metricList": ["MAE"],
                        "predictList": ["BestDice"],
                        "epochs":1000}


        if exp_name == "OneHeadLoc_cp":
            exp_dict = {"modelList":["OneHeadLoc"],
                        "configList": ["noFlip"],
                        "lossList": ["OneHeadLoss"],
                        "datasetList": ["CityScapes"],
                        "metricList": ["MAE"],
                        "predictList": ["BestDice"],
                        "epochs":1000}

        if exp_name == "OneHeadRandom_cp":
            exp_dict = {"modelList":["OneHeadLoc"],
                        "configList": ["noFlip"],
                        "lossList": ["OneHeadRandomLoss"],
                        "datasetList": ["CityScapes"],
                        "metricList": ["MAE"],
                        "predictList": ["BestDice"],
                        "epochs":1000}


        if exp_name == "OneHead_coco2014":
            exp_dict = {"modelList":["OneHead", "OneHeadLoc"],
                        "configList": ["noFlip"],
                        "lossList": ["OneHeadLoss"],
                        "datasetList": ["CocoDetection2014"],
                        "metricList": ["MAE"],
                        "predictList": ["BestDice"],
                        "epochs":1000}

        if exp_name == "TwoHeads_coco2014":
            exp_dict = {"modelList":["TwoHeads"],
                        "configList": ["noFlip"],
                        "lossList": ["TwoHeadLoss_9_1"],
                        "datasetList": ["CocoDetection2014"],
                        "metricList": ["MAE"],
                        "predictList": ["BestDice", "BestObjectness"],
                        "epochs":1000}


        if exp_name == "TwoHeads_kitti":
            exp_dict = {"modelList":["TwoHeads"],
                        "configList": ["noFlip"],
                        "lossList": ["TwoHeadLoss_9_1"],
                        "datasetList": ["Kitti"],
                        "metricList": ["MAE"],
                        "predictList": ["BestDice"],
                        "epochs":1000}

        if exp_name == "TwoHeads_cp":
            exp_dict = {"modelList":["TwoHeads"],
                        "configList": ["noFlip"],
                        "lossList": ["TwoHeadLoss_9_1"],
                        "datasetList": ["CityScapes"],
                        "metricList": ["MAE"],
                        "predictList": ["BestDice"],
                        "epochs":1000}

        if exp_name == "TwoHeads_plants":
            exp_dict = {"modelList":["TwoHeads"],
                        "configList": ["noFlip"],
                        "lossList": ["TwoHeadLoss_9_1"],
                        "datasetList": ["Plants"],
                        "metricList": ["MAE"],
                        "predictList": ["BestDice"],
                        "epochs":1000}


        if exp_name == "TwoHeads_pascal":
            exp_dict = {"modelList":["TwoHeads"],
                        "configList": ["noFlip"],
                        "lossList": ["TwoHeadLoss_9_1"],
                        "datasetList": ["Pascal2012"],
                        "metricList": ["MAE"],
                        "predictList": ["BestDice"],
                        "epochs":1000}


        if exp_name == "TwoHeads_search_pascal":
            configList = []
            for i in range(-5,6): 
                for j in range(-5,6): 
                    configList += ["twoheads_%d_%d" % (i,j)]

            exp_dict = {"modelList":["TwoHeads"],
                        "configList": configList,
                        "lossList": ["TwoHeadLoss"],
                        "datasetList": ["Pascal2012"],
                        "metricList": ["MAE"],
                        "predictList": ["BestDice", "BestObjectness"],
                        "epochs":1000}




        if exp_name == "cp2lcfcn_points":
            exp_dict = {"modelList":["CP_LCFCN"],
                        "configList": ["noFlip"],
                        "lossList": ["box_loss"],
                        "datasetList": ["CityScapes"],
                        "metricList": ["MAE"],
                        "predictList": ["BoxSegment"],
                        "epochs":1000}

        if exp_name == "pascal2lcfcn_points":
            exp_dict = {"modelList":["LC_RESFCN"],
                        "configList": ["noFlip"],
                        "lossList": ["box_loss"],
                        "datasetList": ["Pascal2012"],
                        "metricList": ["MAE"],
                        "predictList": ["BoxSegment"],
                        "epochs":1000}

        if exp_name == "box":
            exp_dict = {"modelList":["Boxer"],
                        "configList": ["noFlip"],
                        "lossList": ["box_loss"],
                        "datasetList": ["Pascal2012"],
                        "metricList": ["MAE"],
                        "predictList": ["BoxSegment"],
                        "epochs":1000}

        if exp_name == "dice":
            exp_dict = {"modelList":["Segmenter"],
                        "configList": ["noFlip"],
                        "lossList": ["region_growing_loss"],
                        "datasetList": ["Pascal2012"],
                        "metricList": ["MAE"],
                        "predictList": ["BestDice"],
                        "epochs":1000}

        if exp_name == "mp":
            exp_dict = {"modelList":["Segmenter"],
                        "configList": ["noFlip"],
                        "lossList": ["matching_proposal",
                        "matching_proposal2",
                        "matching_proposal3",
                        "matching_proposal4"],
                        "datasetList": ["Pascal2012"],
                        "metricList": ["MAE"],
                        "predictList": ["BestDice", "BestObjectness"],
                        "epochs":1000}

        if exp_name == "mp_scratch":
            exp_dict = {"modelList":["Segmenter_Scratch"],
                        "configList": ["noFlip"],
                        "lossList": ["matching_proposal5",
                        "matching_proposal6"],
                        "datasetList": ["Pascal2012"],
                        "metricList": ["MAE"],
                        "predictList": ["BestDice", "BestObjectness"],
                        "epochs":1000}

        if exp_name == "mp_better":
            exp_dict = {"modelList":["Segmenter_Scratch"],
                        "configList": ["noFlip"],
                        "lossList": ["matching_proposal7",
                        "matching_proposal8"],
                        "datasetList": ["Pascal2012"],
                        "metricList": ["MAE"],
                        "predictList": ["BestCounter"],
                        "epochs":1000}


        if exp_name == "cp_lcfcn":
            exp_dict = {"modelList":["Res50FCN"],
                        "configList": ["noFlip"],
                        "lossList": ["lc_loss"],
                        "datasetList": ["CityScapes"],
                        "metricList": ["MAE"],
                        "predictList": ["BestCounter"],
                        "epochs":1000}


        if exp_name == "cp_old":
            exp_dict = {"modelList":["ResEmbedding"],
                        "configList": ["noFlip"],
                        "lossList": ["similarity_loss_old"],
                        "datasetList": ["CityScapes"],
                        "metricList": ["MAE"],
                        "predictList": ["BestCounter"],
                        "epochs":1000}

        if exp_name == "pascal_old":
            exp_dict = {"modelList":["ResEmbedding"],
                        "configList": ["noFlip"],
                        "lossList": ["similarity_loss_old"],
                        "datasetList": ["Pascal2012"],
                        "metricList": ["MAE"],
                        "predictList": ["BestCounter"],
                        "epochs":1000}


        if exp_name == "pascal_similarity":
            exp_dict = {"modelList":["ResEmbedding"],
                        "configList": ["noFlip"],
                        "lossList": ["similarity_loss"],
                        "datasetList": ["Pascal2012"],
                        "metricList": ["MAE"],
                        "predictList": ["BestCounter"],
                        "epochs":1000}

        if exp_name == "pascal_similarity3":
            exp_dict = {"modelList":["ResSimilarity"],
                        "configList": ["noFlip"],
                        "lossList": ["similarity_loss"],
                        "datasetList": ["Pascal2012"],
                        "metricList": ["MAE"],
                        "predictList": ["BestCounter"],
                        "epochs":1000}


        if exp_name == "pascal_similarity2":
            exp_dict = {"modelList":["DoubleResEmbedding"],
                        "configList": ["noFlip"],
                        "lossList": ["similarity_loss"],
                        "datasetList": ["Pascal2012"],
                        "metricList": ["MAE"],
                        "predictList": ["BestCounter"],
                        "epochs":1000}


        if exp_name == "cp_similarity":
            exp_dict = {"modelList":["ResEmbedding"],
                        "configList": ["noFlip"],
                        "lossList": ["similarity_loss"],
                        "datasetList": ["CityScapes"],
                        "metricList": ["MAE"],
                        "predictList": ["BestCounter"],
                        "epochs":1000}

        if exp_name == "cp":
            exp_dict = {"modelList":["Segmenter_Scratch"],
                        "configList": ["noFlip"],
                        "lossList": ["matching_proposal7",
                        "matching_proposal8"],
                        "datasetList": ["CityScapes"],
                        "metricList": ["MAE"],
                        "predictList": ["BestCounter"],
                        "epochs":1000}

        if exp_name == "hybrid":
            exp_dict = {"modelList":["Hybrid"],
                        "configList": ["noFlip"],
                        "lossList": ["matching_proposal_hybrid1",
                        "matching_proposal_hybrid2"],
                        "datasetList": ["Pascal2012"],
                        "metricList": ["MAE"],
                        "predictList": ["BestDice","BestObjectness_hybrid", "BestObjectness"],
                        "epochs":1000}


        if exp_name == "hybrid_scratch":
            exp_dict = {"modelList":["Hybrid_scratch"],
                        "configList": ["noFlip"],
                        "lossList": ["matching_proposal_hybrid3",
                        "matching_proposal_hybrid2"],
                        "datasetList": ["Pascal2012"],
                        "metricList": ["MAE"],
                        "predictList": ["BestDice","BestObjectness_hybrid", "BestObjectness"],
                        "epochs":1000}


        if exp_name == "matching_proposal1":
            exp_dict = {"modelList":["Segmenter"],
                        "configList": ["noFlip"],
                        "lossList": ["matching_proposal"],
                        "datasetList": ["Pascal2012"],
                        "metricList": ["MAE"],
                        "predictList": ["BestDice", "BestObjectness"],
                        "epochs":1000}

        if exp_name == "matching_proposal2":
            exp_dict = {"modelList":["Segmenter"],
                        "configList": ["noFlip"],
                        "lossList": ["matching_proposal2"],
                        "datasetList": ["Pascal2012"],
                        "metricList": ["MAE"],
                        "predictList": ["BestDice", "BestObjectness"],
                        "epochs":1000}
        if exp_name == "matching_proposal3":
            exp_dict = {"modelList":["Segmenter"],
                        "configList": ["noFlip"],
                        "lossList": ["matching_proposal3"],
                        "datasetList": ["Pascal2012"],
                        "metricList": ["MAE"],
                        "predictList": ["BestDice", "BestObjectness"],
                        "epochs":1000}

        if exp_name == "matching_proposal4":
            exp_dict = {"modelList":["Segmenter"],
                        "configList": ["noFlip"],
                        "lossList": ["matching_proposal4"],
                        "datasetList": ["Pascal2012"],
                        "metricList": ["MAE"],
                        "predictList": ["BestDice", "BestObjectness"],
                        "epochs":1000}

        if exp_name == "matching_proposal5":
            exp_dict = {"modelList":["Segmenter_Scratch"],
                        "configList": ["noFlip"],
                        "lossList": ["matching_proposal5"],
                        "datasetList": ["Pascal2012"],
                        "metricList": ["MAE"],
                        "predictList": ["BestDice", "BestObjectness"],
                        "epochs":1000}


        if exp_name == "matching_proposal6":
            exp_dict = {"modelList":["Segmenter_Scratch"],
                        "configList": ["noFlip"],
                        "lossList": ["matching_proposal6"],
                        "datasetList": ["Pascal2012"],
                        "metricList": ["MAE"],
                        "predictList": ["BestDice", "BestObjectness"],
                        "epochs":1000}

        if exp_name == "matching_proposal_hybrid1":
            exp_dict = {"modelList":["Hybrid"],
                        "configList": ["noFlip"],
                        "lossList": ["matching_proposal_hybrid1"],
                        "datasetList": ["Pascal2012"],
                        "metricList": ["MAE"],
                        "predictList": ["BestDice","BestObjectness_hybrid", "BestObjectness"],
                        "epochs":1000}
        if exp_name == "matching_proposal_hybrid2":
            exp_dict = {"modelList":["Hybrid"],
                        "configList": ["noFlip"],
                        "lossList": ["matching_proposal_hybrid2"],
                        "datasetList": ["Pascal2012"],
                        "metricList": ["MAE"],
                        "predictList": ["BestDice","BestObjectness_hybrid", "BestObjectness"],
                        "epochs":1000}

        if exp_name == "matching_proposal_hybrid3":
            exp_dict = {"modelList":["Hybrid_scratch"],
                        "configList": ["noFlip"],
                        "lossList": ["matching_proposal_hybrid3"],
                        "datasetList": ["Pascal2012"],
                        "metricList": ["MAE"],
                        "predictList": ["BestDice","BestObjectness_hybrid", "BestObjectness"],
                        "epochs":1000}

        if exp_name == "similarity":
            exp_dict = {"modelList":["Hybrid_scratch"],
                        "configList": ["noFlip"],
                        "lossList": ["matching_proposal_hybrid3"],
                        "datasetList": ["Pascal2012"],
                        "metricList": ["MAE"],
                        "predictList": ["BestDice","BestObjectness_hybrid", "BestObjectness"],
                        "epochs":1000}

        if exp_name == "matching_proposal_hybrid4":
            exp_dict = {"modelList":["Hybrid_scratch"],
                        "configList": ["noFlip"],
                        "lossList": ["matching_proposal_hybrid2"],
                        "datasetList": ["Pascal2012"],
                        "metricList": ["MAE"],
                        "predictList": ["BestDice","BestObjectness_hybrid", "BestObjectness"],
                        "epochs":1000}

        if exp_name == "glance":
            exp_dict = {"modelList":["Glance_LCFCN"],
                        "configList": ["noFlip"],
                        "lossList": ["glance_loss"],
                        "datasetList": ["Pascal2012"],
                        "metricList": ["MAE"],
                        "predictList":["GlanceBestBox"],
                        "epochs":1000}


        if exp_name == "BestObjectness":
            exp_dict = {"modelList":["LC_RESFCN"],
                        "configList": ["noFlip"],
                        "lossList": ["proposal_pretrained_loss"],
                        "datasetList": ["Pascal2012"],
                        "predictList":["BestObjectness"],
                        "metricList": ["MAE"],
                        "epochs":1000}


        if exp_name == "UpperBound":
            exp_dict = {"modelList":["LC_RESFCN"],
                        "configList": ["noFlip"],
                        "lossList": ["proposal_pretrained_loss"],
                        "datasetList": ["Pascal2012"],
                        "predictList":["UpperBound"],
                        "metricList": ["MAE"],
                        "epochs":1000}

        if exp_name == "plants":
            exp_dict = {"modelList":["PSPNet"],
                        "configList": ["basic"],
                        "lossList": ["recursive_blob_loss",
                                     "water_loss", 
                                     "sp_water_loss"],
                        "datasetList": ["Plants"],
                        "metricList": ["SBD"],
                        "epochs":1000}


        if exp_name == "pascal_yolo":
            exp_dict = {"modelList":["YOLO"],
                        "configList": ["yolo"],
                        "lossList": ["yolo_loss"],
                        "datasetList": ["Pascal2012"],
                        "metricList": ["MAE"],
                        "epochs":1000}

        if exp_name == "recursive":
            exp_dict = {"modelList":["LC_RESFCN1"],
                        "configList": ["wtp"],
                        "lossList": ["fixed_recursive_loss"],
                        "datasetList": ["Pascal2012"],
                        "metricList": ["MAE"],
                        "epochs":1000}

        if exp_name == "recursive2":
            exp_dict = {"modelList":["LC_RESFCN2"],
                        "configList": ["wtp"],
                        "lossList": ["fixed_recursive_loss"],
                        "datasetList": ["Pascal2012"],
                        "metricList": ["MAE"],
                        "epochs":1000}

        if exp_name == "pascal":
            exp_dict = {"modelList":["Res50FCN","PSPNet"],
                        "configList": ["wtp"],
                        "lossList": ["water_loss",
                                "sp_water_loss", "recursive_blob_loss"],
                        "datasetList": ["Pascal2012"],
                        "metricList": ["MAE"],
                        "predictList":["None"],
                        "epochs":1000}


        if exp_name == "pascal_unet":
            exp_dict = {"modelList":["UNet"],
                        "configList": ["unet"],
                        "lossList": ["unet_loss"],
                        "datasetList": ["Pascal2012"],
                        "metricList": ["mIoU"],
                        "epochs":1000}


        if exp_name == "pascal_unetpoint":
            exp_dict = {"modelList":["ResDextrPoint", "UNetPoint"],
                        "configList": ["unet"],
                        "lossList": ["unetpoint_loss"],
                        "datasetList": ["PascalClicksSingle"],
                        "metricList": ["mIoU"],
                        "epochs":1000}

        if exp_name == "pascal_clicks2":
            exp_dict = {"modelList":["ResDextr3"],
                        "configList": ["click"],
                        "lossList": ["seg_balanced_loss"],
                        "datasetList": ["PascalClicksFocus"],
                        "metricList": ["mIoU"],
                        "epochs":1000}


        if exp_name == "unet_clicks":
            exp_dict = {"modelList":["UNet"],
                        "configList": ["click"],
                        "lossList": ["unet_loss"],
                        "datasetList": ["PascalClicksSingle"],
                        "metricList": ["mIoU"],
                        "epochs":1000}


        if exp_name == "pascal_clicks":
            exp_dict = {"modelList":["ResDextr"],
                        "configList": ["click"],
                        "lossList": ["seg_balanced_loss"],
                        "datasetList": ["PascalClicks"],
                        "metricList": ["mIoU"],
                        "epochs":1000}

        if exp_name == "cross_fish":
            exp_dict = {"configList": ["mcnn_1000", "resfcn_1000", "fcn8_1000", "glance_1000"],
                        "lossList": ["water_loss", "water_loss_B"],
                        "datasetList": ["ExceptLafrage", "ExceptComoLake"],
                        "metricList": ["MAE"],
                        "epochs":1000}


        if exp_name == "weakly_count":
            exp_dict = {"configList": ["resfcn_1000", "fcn8_1000", "glance_1000"],
                        "lossList": ["image_loss"],
                        "datasetList": ["ComoLake"],
                        "metricList": ["MAE"],
                        "epochs":1000}

        if exp_name == "weakly_detection":
            exp_dict = {"configList": ["mcnn_1000", "resfcn_1000", "fcn8_1000", "glance_1000"],
                        "lossList": ["water_loss"],
                        "datasetList": ["ComoLake"],
                        "metricList": ["MAE"],
                        "epochs":1000}

        if exp_name == "weakly_instance":
            exp_dict = {"configList": ["mcnn_1000", "resfcn_1000", "fcn8_1000", "glance_1000"],
                        "lossList": ["water_loss"],
                        "datasetList": ["ExceptLafrage", "ExceptComoLake"],
                        "metricList": ["MAE"],
                        "epochs":1000}


        # Override if needed
        exp_dict["configList"] = args.configList or exp_dict["configList"]
        exp_dict["metricList"] = args.metricList or exp_dict["metricList"]
        exp_dict["datasetList"] = args.datasetList or exp_dict["datasetList"]
        exp_dict["lossList"] = args.lossList or exp_dict["lossList"]  
        exp_dict["modelList"] = args.modelList or exp_dict["modelList"]  
        exp_dict["predictList"] = args.predictList or exp_dict["predictList"]  

        return exp_dict


def get_config_dict(config_name):
    base1 = {"val_batchsize" :1,
            "opt_name": "adam",
            "batch_size": 1,
            "epochs": 500,

            "sampler_name": "Random",
            "epoch2val":5,
            "trainTransformer":"hflipNormalize",
            "testTransformer":"normalize",
            
            "opt_options" :{"lr":1e-5, "weight_decay":0.0005},
            "model_options":{},
            "dataset_options":{},
            "verbose":True}

    smallPascal = {"val_batchsize" :1,
            "opt_name": "adam",
            "batch_size": 1,
            "epochs": 500,

            "sampler_name": "Random1000",
            "epoch2val":5,
            "trainTransformer":"Te_WTP",
            "testTransformer":"Te_WTP",
            
            "opt_options" :{"lr":1e-5, "weight_decay":0.0005},
            "model_options":{},
            "dataset_options":{},
            "verbose":True}


    noFlip = {"val_batchsize" :1,
            "opt_name": "adam",
            "batch_size": 1,
            "epochs": 500,

            "sampler_name": "Random1000",
            "epoch2val":5,
            "trainTransformer":"Tr_WTP_NoFlip",
            "testTransformer":"Te_WTP",
            
            "opt_options" :{"lr":1e-5, "weight_decay":0.0005},
            "model_options":{},
            "dataset_options":{},
            "verbose":True}

    debug = {"val_batchsize" :1,
            "opt_name": "adam",
            "batch_size": 1,
            "epochs": 500,

            "sampler_name": "Random10",
            "epoch2val":5,
            "trainTransformer":"Tr_WTP_NoFlip",
            "testTransformer":"Te_WTP",
            
            "opt_options" :{"lr":1e-5, "weight_decay":0.0005},
            "model_options":{},
            "dataset_options":{},
            "verbose":True}

    base2 = base1
    # base2["epoch2val"] = 20
    if config_name == "smallPascal":
        config_dict = smallPascal
        
    if config_name == "basic":
        config_dict = base1
      
    if config_name == "noFlip2":
        config_dict = noFlip
        config_dict["testTransformer"] = "normalize"
      
    if config_name == "noFlip":
        config_dict = noFlip

    if config_name == "debug":
        config_dict = debug

    if "twoheads" in config_name:
        _, s1, s2 = config_name.split("_") 

        config_dict = noFlip
        config_dict["model_options"]={"scale1":eval("1e%s"%s1),
                                      "scale2":eval("1e%s"%s2)}




    if config_name == "wtp":
        config_dict = base1
        config_dict["sampler_name"] ="Random1000"
        config_dict["trainTransformer"]="Tr_WTP"
        config_dict["testTransformer"]="Te_WTP"


        # config_dict["model_options"]={"scale1":1e-3,"scale2":1e-3}
    if config_name == "yolo":
        config_dict = base1
        config_dict["sampler_name"] ="Random1000"
        config_dict["trainTransformer"]="Tr_resize"
        config_dict["testTransformer"]="Te_resize"

    if config_name == "unet":
        config_dict = base1
        config_dict["sampler_name"] ="Random1000"
        config_dict["trainTransformer"]="Tr_WTP"
        config_dict["testTransformer"]="Te_WTP"

    if config_name == "click":
        config_dict = base1
        config_dict["opt_options"] ={"lr":1e-3, "weight_decay":0.0005}
        config_dict["sampler_name"] ="Random1000"
        config_dict["trainTransformer"]="Tr_WTP"
        config_dict["testTransformer"]="Te_WTP"



        
    if config_name == "SPNetWSL":
        config_dict = base1
        config_dict["model_name"] =  "SPNetWSL"

    if config_name == "ResNetWSL":
        config_dict = base1
        config_dict["model_name"] =  "ResNetWSL"

    if config_name == "FCN8_multi":
        config_dict = base1
        config_dict["model_name"] =  "FCN8_multi"
        

    if config_name == "fcn8":
        config_dict = base1
        config_dict["model_name"] =  "FCN8"

    if config_name == "Localizer":
        config_dict = base1
        config_dict["model_name"] =  "Localizer"

    if config_name == "fcn8_1000":
        config_dict = base1
        config_dict["sampler_name"] ="Random1000"
        config_dict["model_name"] =  "FCN8"

    if config_name == "res50fcn" or config_name == "resfcn":
        config_dict = base1
        config_dict["model_name"] =  "Res50FCN"



    if config_name == "resfcn_wtp":
        config_dict = base1
        config_dict["model_name"] =  "Res50FCN"

        config_dict["trainTransformer"]="Tr_WTP"
        config_dict["testTransformer"]="Te_WTP"

    if config_name == "resfcn_1000":
        config_dict = base1
        config_dict["model_name"] =  "Res50FCN"
        config_dict["sampler_name"] ="Random1000"

    if config_name == "mcnn_1000":
        config_dict = base1
        config_dict["model_name"] =  "MCNN"

        config_dict["sampler_name"] ="Random1000"
        config_dict["loss_name"] = "density_loss"

    if config_name == "pspnet":
        config_dict = base1
        config_dict["model_name"] =  "PSPNet"

    if config_name == "density_fcn8":
        config_dict = base1
        config_dict["model_name"] =  "DensityFCN8"
        config_dict["loss_name"] = "density_loss"

    if config_name == "density_res50fcn":
        config_dict = base1
        config_dict["model_name"] =  "DensityRESFCN8"

        config_dict["loss_name"] = "density_loss"


    if config_name == "mcnn":
        config_dict = base1
        config_dict["model_name"] =  "MCNN"

        config_dict["loss_name"] = "density_loss"
        config_dict["opt_options"] = {"lr":1e-5, 
                                      "weight_decay":0.0005}


    if config_name == "glance":
        config_dict = {"model_name": "Glance",
        "loss_name":"least_squares",
        "opt_name": "adam",
        "val_batchsize" :16,
        "batch_size": 16,
        "epochs": 500,

        "sampler_name": "Random",
        "epoch2val":5,
        "trainTransformer":"normalizeResize",
        "testTransformer":"normalizeResize",
        
        "opt_options" :{"lr":1e-5, "weight_decay":0.0005},
        "model_options":{"layers":"500-500"},
        "dataset_options":{},
        "verbose":True}

    if config_name == "glance_1000":
        config_dict = {"model_name": "Glance",
        "loss_name":"least_squares",
        "opt_name": "adam",
        "val_batchsize" :16,
        "batch_size": 16,
        "epochs": 500,

        "sampler_name": "Random1000",
        "epoch2val":5,
        "trainTransformer":"normalizeResize",
        "testTransformer":"normalizeResize",
        
        "opt_options" :{"lr":1e-5, "weight_decay":0.0005},
        "model_options":{"layers":"500-500"},
        "dataset_options":{},
        "verbose":True}

    return config_dict



# Main dict
class MainClass:
    def __init__(self, path_datasets, path_models, path_samplers, path_transforms,
                       path_metrics, path_losses, path_saves, project):
        self.path_datasets = path_datasets
        self.path_saves = path_saves
        self.dataset_dict = get_module_classes(path_datasets)
        self.model_dict = get_module_classes(path_models)

        self.loss_dict = get_functions(path_losses)
        self.metric_dict = get_functions(path_metrics)
        self.sampler_dict = get_functions(path_samplers)
        self.transform_dict = get_functions(path_transforms)
        self.project = project

        

        self.opt_dict = {"adam":optim.Adam, 
                         "adamFast":lambda params, lr, 
                          weight_decay:optim.Adam(params, lr=lr, betas=(0.9999,0.9999999), weight_decay=weight_decay),

                          "sgd":lambda params, lr, 
                           weight_decay:optim.SGD(params, lr=lr, weight_decay=weight_decay,
                                                                momentum=0.9)}
        # DATASETS

        
        
    def get_main_dict(self, mode, dataset_name, model_name, config_name, config, reset, 
                            epochs, metric_name, loss_name, 
                            gpu=None):
        main_dict = config
        main_dict["config_name"] = config_name
        main_dict["model_name"] = model_name
        main_dict["loss_name"] = loss_name
        main_dict["metric_name"] = metric_name
        main_dict["dataset_name"] = dataset_name
        main_dict["epochs"] = epochs
        main_dict["reset"] = reset
        main_dict["project_name"] = self.project
        main_dict["code_path"] = "/mnt/home/issam/Research_Ground/{}".format(self.project)
        # GET GPU
        set_gpu(gpu)

        main_dict["path_datasets"] = self.path_datasets
        main_dict["exp_name"] = ("dataset:{}_model:{}_metric:{}_loss:{}_config:{}".format 
                                (dataset_name, model_name, 
                                 metric_name, loss_name,config_name))


        # SAVE
        main_dict["path_save"] = "{}/{}/".format(self.path_saves, 
                                                 main_dict["exp_name"])

        path_save = main_dict["path_save"]


        main_dict["path_summary"] = main_dict["path_save"].replace("Saves", "Summaries")


        main_dict["metric_dict"] = self.metric_dict
        main_dict["sampler_dict"] = self.sampler_dict
        main_dict["loss_dict"] = self.loss_dict
        main_dict["model_dict"] = self.model_dict
        main_dict["dataset_dict"] = self.dataset_dict
        main_dict["opt_dict"] = self.opt_dict

        main_dict["transform_dict"] = self.transform_dict

        main_dict["path_history"]= path_save + "/history.pkl"
        main_dict["path_train_opt"]= path_save + "/State_Dicts/opt.pth"
        
        main_dict["path_train_model"]= path_save + "/State_Dicts/model.pth"
        main_dict["path_baselines"]= path_save + "/baselines.pkl"
        main_dict["path_best_model"]= path_save + "/State_Dicts/best_model.pth"
        main_dict["path_best_annList"]= path_save + "/State_Dicts/best_annList.pkl"

        assert_exist(main_dict["model_name"], self.model_dict)
        assert_exist(main_dict["loss_name"], self.loss_dict)
        assert_exist(main_dict["metric_name"], self.metric_dict)
        assert_exist(main_dict["dataset_name"], self.dataset_dict)

        return main_dict
