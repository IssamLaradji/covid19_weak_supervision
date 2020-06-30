
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


    wtp = {"val_batchsize" :1,
            "opt_name": "adam",
            "batch_size": 1,
            "epochs": 500,

            "sampler_name": "Sequence",
            "epoch2val":5,
            "trainTransformer":"Tr_WTP_NoFlip",
            "testTransformer":"Te_WTP",
            
            "opt_options" :{"lr":1e-5, "weight_decay":0.0005},
            "model_options":{},
            "dataset_options":{},
            "verbose":True}

    debug_noVal = {"val_batchsize" :1,
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
            "verbose":"noVal"}

    base2 = base1
    # base2["epoch2val"] = 20
    if config_name == "smallPascal":
        config_dict = smallPascal

    if config_name == "wtp":
        config_dict = wtp
      
            
    if config_name == "basic":
        config_dict = base1
      
    if config_name == "noFlip2":
        config_dict = noFlip
        config_dict["testTransformer"] = "normalize"
      
    if config_name == "noFlip":
        config_dict = noFlip

    if config_name == "noFlip_gt":
        config_dict = noFlip
        config_dict["trainTransformer"] = "Te_WTP"

    if config_name == "debug_noVal":
        config_dict = debug_noVal

    if config_name == "debug":
        config_dict = debug

    if "twoheads" in config_name:
        _, s1, s2 = config_name.split("_") 

        config_dict = noFlip
        config_dict["model_options"]={"scale1":eval("1e%s"%s1),
                                      "scale2":eval("1e%s"%s2)}




    # if config_name == "wtp":
    #     config_dict = base1
    #     # config_dict["sampler_name"] ="Random1000"
    #     # config_dict["trainTransformer"]="Tr_WTP"
    #     # config_dict["testTransformer"]="Te_WTP"


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

