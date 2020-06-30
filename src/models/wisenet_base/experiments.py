

"""
if 1:
    ls = []
    for i in range(-3,4): 
        for j in range(-3,4): 
            ls += ["twoheads_%d_%d" % (i,j)]
"""
def get_experiment_dict(args, exp_name):
        
        exp_dict = {"modelList":None,
                    "configList": None,
                    "lossList": None,
                    "datasetList": None,
                    "metricList": None,

                    "epochs":1000}

        if exp_name == "affinity_pascal":
            exp_dict = {"modelList":["AffinityNet"],
                        "configList": ["noFlip"],
                        "lossList": ["AffinityLoss"],
                        "datasetList": ["PascalOriginal"],
                        "metricList": ["MAE"],
                        "predictList": ["BestDice"],
                        "epochs":1000}


        if exp_name == "prm_original":
            exp_dict = {"modelList":["PRM"],
                        "configList": ["noFlip"],
                        "lossList": ["PRMLoss"],
                        "datasetList": ["PascalOriginal"],
                        "metricList": ["MAE"],
                        "predictList": ["BestDice"],
                        "epochs":1000}

        if exp_name == "lcfcn_bo":
            exp_dict = {"modelList":["LCFCN_BO", "LCFCN_Pyramid","LCFCN_Dilated"],
                        "configList": ["wtp"],
                        "lossList": ["lcfcnLoss"],
                        "datasetList": ["Pascal2012","Plants", "CocoDetection2014", "CityScapes", "Kitti"],
                        "metricList": ["MAE"],
                        "predictList": ["BestDice"],
                        "epochs":100}

        if exp_name == "lcfcn_bo_cp":
            exp_dict = {"modelList":["TwoHeads_mIoU"],
                        "configList": ["noFlip"],
                        "lossList": ["OneHeadRBFLoss"],
                        "datasetList": ["CityScapesObject"],
                        "metricList": ["mIoU"],
                        "predictList": ["BestDice"],
                        "epochs":100}
                       

        if exp_name == "wiseaffinity":
            exp_dict = {"modelList":["WiseAffinity"],
                        "configList": ["noFlip"],
                        "lossList": ["wiseaffinity_loss"],
                        "datasetList": ["PascalOriginal"],
                        "metricList": ["MAE"],
                        "predictList": ["BestDice"],
                        "epochs":100}
                       

        if exp_name == "lcfcn_bo_coco":
            exp_dict = {"modelList":["LCFCN_BO", "LCFCN_Pyramid"],
                        "configList": ["noFlip"],
                        "lossList": ["lcfcnLoss"],
                        "datasetList": ["Plants", "CocoDetection2014", "CityScapes", "Kitti"],
                        "metricList": ["MAE"],
                        "predictList": ["BestDice"],
                        "epochs":100}
                       

        if exp_name == "lcfcn_reg":
            exp_dict = {"modelList":["LCFCN_Regularized"],
                        "configList": ["wtp"],
                        "lossList": ["lcfcnRegularizedLoss"],
                        "datasetList": ["Pascal2012", "PascalOriginal"],
                        "metricList": ["MAE"],
                        "predictList": ["BestDice"],
                        "epochs":100}

        # if exp_name == "lcfcn_bo":
        #     exp_dict = {"modelList":["LCFCN_BO"],
        #                 "configList": ["wtp"],
        #                 "lossList": ["lcfcnLoss"],
        #                 "datasetList": ["PascalOriginal"],
        #                 "metricList": ["MAE"],
        #                 "predictList": ["BestDice"],
        #                 "epochs":200}


        if exp_name == "maskrcnn_gt":
            exp_dict = {"modelList":["MaskRCNN"],
                        "configList": ["noFlip_gt"],
                        "lossList": ["MaskRCNNLoss_gt"],
                        "datasetList": [ 
                                        "Pascal_122"],
                        "metricList": ["MAE"],
                        "predictList": ["BestDice"],
                        "epochs":1000}

        if exp_name == "maskrcnn_sm":
            exp_dict = {"modelList":["MaskRCNN"],
                        "configList": ["noFlip"],
                        "lossList": ["MaskRCNNLoss_sm"],
                        "datasetList": ["CityScapes", "CocoDetection2014","Kitti", "PascalOriginal"],
                        "metricList": ["MAE"],
                        "predictList": ["BestDice"],
                        "epochs":1000}


        if exp_name == "onehead_losses":
            exp_dict = {"modelList":["OneHead"],
                        "configList": ["wtp"],
                        "lossList": [ "OneHeadRBFLoss", "OneHeadRBFLoss_withSim"],
                        "datasetList": ["PascalOriginal", 
                                        "Pascal_1226"],
                        "metricList": ["MAE"],
                        "predictList": ["BestDice"],
                        "epochs":1000}

        if exp_name == "onehead_triplet":
            exp_dict = {"modelList":[ "OneHeadLoc", "OneHead"],
                        "configList": ["noFlip"],
                        "lossList": ["TripletLoss"],
                        "datasetList": ["PascalOriginal"],
                        "metricList": ["MAE"],
                        "predictList": ["BestDice"],
                        "epochs":1000}


        if exp_name == "onehead":
            exp_dict = {"modelList":[ "OneHead_32", "OneHead_256","OneHead_128", "OneHead", "OneHead_Pyramid"],
                        "configList": ["noFlip"],
                        "lossList": ["OneHeadSumLoss"],
                        "datasetList": ["PascalOriginal"],
                        "metricList": ["MAE"],
                        "predictList": ["BestDice"],
                        "epochs":1000}


        if exp_name == "aff":
            exp_dict = {"modelList":["AFFNet"],
                        "configList": ["noFlip"],
                        "lossList": ["AFFLoss"],
                        "datasetList": ["Pascal2012"],
                        "metricList": ["MAE"],
                        "predictList": ["MAE"],
                        "epochs":1000}



        if exp_name == "OneHeadProto":
            exp_dict = {"modelList":["OneHeadProto_original", "OneHeadProto"],
                        "configList": ["noFlip"],
                        "lossList": ["OneHeadLoss_prototypes"],
                        "datasetList": ["Pascal2012", "PascalOriginal"],
                        "metricList": ["MAE"],
                        "predictList": ["BestDice"],
                        "epochs":1000}


        if exp_name == "OneHead_strong":
            exp_dict = {"modelList":["OneHeadStrong"],
                        "configList": ["noFlip"],
                        "lossList": ["OneHeadSumLoss"],
                        "datasetList": ["Pascal2012", "PascalOriginal"],
                        "metricList": ["MAE"],
                        "predictList": ["BestDice"],
                        "epochs":1000}


        if exp_name == "TwoHeads_strong":
            exp_dict = {"modelList":["TwoHeadsStrong","TwoHeadsStrong101"],
                        "configList": ["noFlip"],
                        "lossList": ["TwoHeadLoss_sum", "TwoHeadLoss_9_1"],
                        "datasetList": ["Pascal2012","PascalOriginal"],
                        "metricList": ["MAE"],
                        "predictList": ["BestDice"],
                        "epochs":1000}




        if exp_name == "TwoHeads_pascal":
            exp_dict = {"modelList":[ "TwoHeads"],
                        "configList": ["noFlip"],
                        "lossList": ["TwoHeadLoss_sum", "TwoHeadLoss_9_1"],
                        "datasetList": ["Pascal2012","PascalOriginal"],
                        "metricList": ["MAE"],
                        "predictList": ["BestDice"],
                        "epochs":1000}


        if exp_name == "OneHead_original":
            exp_dict = {"modelList":["OneHead"],
                        "configList": ["noFlip"],
                        "lossList": ["OneHeadSumLoss"],
                        "datasetList": ["Pascal2012", "PascalOriginal"],
                        "metricList": ["MAE"],
                        "predictList": ["BestDice"],
                        "epochs":1000}


        if exp_name == "OneHead_pascal":
            exp_dict = {"modelList":["OneHead"],
                        "configList": ["noFlip"],
                        "lossList": [  "OneHeadLocLoss", 
                                     "OneHeadLoss_tmp"],
                        "datasetList": ["Pascal2012","PascalOriginal"],
                        "metricList": ["MAE"],
                        "predictList": ["BestDice"],
                        "epochs":1000}


        if exp_name == "maskrcnn_small":
            exp_dict = {"modelList":["MaskRCNN"],
                        "configList": ["smallPascal"],
                        "lossList": ["MaskRCNNLoss_small"],
                        "datasetList": ["PascalOriginalSmall", "PascalSmall"],
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
                        "predictList": ["BestDice"],
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


        if exp_name == "maskrcnn_tmp":
            exp_dict = {"modelList":["MaskRCNN"],
                        "configList": ["noFlip"],
                        "lossList": ["MaskRCNNLoss_tmp"],
                        "datasetList": ["Pascal2012"],
                        "metricList": ["MAE"],
                        "predictList": ["BestDice"],
                        "epochs":1000}


        # if exp_name == "maskrcnn":
        #     exp_dict = {"modelList":["MaskRCNN"],
        #                 "configList": ["noFlip"],
        #                 "lossList": ["MaskRCNNLoss"],
        #                 "datasetList": ["PascalOriginal", "Pascal2012"],
        #                 "metricList": ["MAE"],
        #                 "predictList": ["BestDice"],
        #                 "epochs":1000}



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

        # if exp_name == "OneHead_pascal":
        #     exp_dict = {"modelList":["OneHead"],
        #                 "configList": ["noFlip"],
        #                 "lossList": ["OneHeadLoss"],
        #                 "datasetList": ["Pascal2012"],
        #                 "metricList": ["MAE"],
        #                 "predictList": ["BestDice"],
        #                 "epochs":1000}

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



        if exp_name == "TwoHeads_plants_lcfcn":
            exp_dict = {"modelList":[ "TwoHeads_Plants"],
                        "configList": ["noFlip"],
                        "lossList": ["OneHeadSumLoss", "OneHeadRBFLoss"],
                        "datasetList": ["Plants"],
                        "metricList": ["MAE"],
                        "predictList": ["BestDice"],
                        "epochs":1000}

        if exp_name == "TwoHeads_coco_lcfcn":
            exp_dict = {"modelList":["TwoHeads_COCO"],
                        "configList": ["noFlip"],
                        "lossList": ["OneHeadRBFLoss_withSim", "OneHeadRBFLoss_random", "OneHeadSumLoss","OneHeadRBFLoss"],
                        "datasetList": ["CocoDetection2014"],
                        "metricList": ["MAE"],
                        "predictList": ["BestDice"],
                        "epochs":1000}

        if exp_name == "TwoHeads_kitti_lcfcn":
            exp_dict = {"modelList":["TwoHeads_Kitti"],
                        "configList": ["noFlip"],
                        "lossList": ["OneHeadRBFLoss_withSim", "OneHeadRBFLoss"],
                        "datasetList": ["Kitti"],
                        "metricList": ["MAE"],
                        "predictList": ["BestDice"],
                        "epochs":1000}


        if exp_name == "TwoHeads_cp_lcfcn":
            exp_dict = {"modelList":["TwoHeads_CityScapes"],
                        "configList": ["noFlip"],
                        "lossList": ["OneHeadRBFLoss_withSim", "OneHeadRBFLoss_random", "OneHeadSumLoss","OneHeadRBFLoss"],
                        "datasetList": ["CityScapes"],
                        "metricList": ["MAE"],
                        "predictList": ["BestDice"],
                        "epochs":1000}

        if exp_name == "TwoHeads_pascal_lcfcn":
            exp_dict = {"modelList":["TwoHeads_Pascal"],
                        "configList": ["noFlip"],
                        "lossList": ["OneHeadRBFLoss", "OneHeadRBFLoss_withSim_noFP",
                        "OneHeadRBFLoss_multiproposals_noFP",
                                    "OneHeadRBFLoss_multiproposals",
                                     "OneHeadRBFLoss_noFP", "OneHeadRBFLoss_withSim"],
                        "datasetList": ["PascalOriginal"],
                        "metricList": ["MAE"],
                        "predictList": ["BestDice"],
                        "epochs":1000}


        if exp_name == "TwoHeads_prm":
            exp_dict = {"modelList":["TwoHeads_PRM"],
                        "configList": ["noFlip"],
                        "lossList": ["OneHeadRBFLoss"],
                        "datasetList": ["PascalOriginal"],
                        "metricList": ["MAE"],
                        "predictList": ["BestDice"],
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


        if exp_name == "TwoHeads_cp_old":
            exp_dict = {"modelList":["TwoHeads"],
                        "configList": ["noFlip"],
                        "lossList": ["TwoHeadLoss_9_1"],
                        "datasetList": ["CityScapesOld"],
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