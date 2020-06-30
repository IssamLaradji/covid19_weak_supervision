import torch

import numpy as np
import timeit
start = timeit.default_timer()
import misc as ms
import ann_utils as au

def main(main_dict, train_only=False):  
  ms.print_welcome(main_dict)

  # EXTRACT VARIABLES
  reset =  main_dict["reset"]
  epochs =  main_dict["epochs"]
  batch_size = main_dict["batch_size"]
  sampler_name = main_dict["sampler_name"]
  verbose = main_dict["verbose"]
  loss_name = main_dict["loss_name"]
  metric_name = main_dict["metric_name"]
  epoch2val = main_dict["epoch2val"]
  val_batchsize = main_dict["val_batchsize"]
  metric_class = main_dict["metric_dict"][metric_name]
  loss_function = main_dict["loss_dict"][loss_name]
  predictList = main_dict["predictList"]

  # Assert everything is available
  ## Sharp proposals
  ## LCFCN points
  ## gt_annDict


  # Dataset  
  train_set, val_set = ms.load_trainval(main_dict)
  train_set[0]

  # Model  
  
  if reset == "reset" or not ms.model_exists(main_dict):
    model, opt, history = ms.init_model_and_opt(main_dict, 
                                                train_set) 
    print("TRAINING FROM SCRATCH EPOCH: %d/%d" % (history["epoch"],
                                                  epochs))
  else:
    model, opt, history = ms.load_latest_model_and_opt(main_dict, 
                                                       train_set) 
    print("RESUMING EPOCH %d/%d" % (history["epoch"], epochs)) 
  
  # Get Dataloader
  trainloader = ms.get_dataloader(dataset=train_set, 
                                  batch_size=batch_size, 
                                  sampler_class=main_dict["sampler_dict"][sampler_name])
  
  # SAVE HISTORY
  history["epoch_size"] = len(trainloader)
  if "trained_batch_names" in history:
    model.trained_batch_names = set(history["trained_batch_names"])

  ms.save_pkl(main_dict["path_history"], history)

  # START TRAINING
  start_epoch = history["epoch"]

  for epoch in range(start_epoch + 1, epochs):
    
    with torch.enable_grad():
      # %%%%%%%%%%% 1. TRAIN PHASE %%%%%%%%%%%%"    
      train_dict = ms.fit(model, trainloader, opt, 
                          loss_function=loss_function,
                          verbose=verbose, 
                          epoch=epoch)

      # Update history
      history["epoch"] = epoch 
      history["trained_batch_names"] = list(model.trained_batch_names)
      history["train"] += [train_dict]

      # Save model, opt and history
      ms.save_latest_model_and_opt(main_dict, model, opt, history)

    # %%%%%%%%%%% 2. VALIDATION PHASE %%%%%%%%%%%%"
    with torch.no_grad():
      for predict_name in predictList: 

        if predict_name == "MAE":
          val_dict = ms.validate(dataset=val_set, 
                          model=model, 
                          verbose=verbose, 
                          metric_class=metric_class, 
                          batch_size=val_batchsize,
                          epoch=epoch)
          val_dict["predict_name"] = predict_name
          val_dict["epoch"] = epoch

          # Update history
          history["val"] += [val_dict]

          # Higher is better

          if (history["best_model"] == {} or 
              history["best_model"][metric_name] >= val_dict[metric_name]):

            history["best_model"] = val_dict
            ms.save_best_model(main_dict, model)
            

        else:
          val_dict, pred_annList = au.validate(model, val_set, 
                      predict_method=predict_name, 
                      n_val=len(val_set), return_annList=True)
        
          val_dict["predict_name"] = predict_name
          val_dict["epoch"] = epoch

          # Update history
          history["val"] += [val_dict]

          # Higher is better
          if (history["best_model"] == {} or 
              history["best_model"]["0.5"] <= val_dict["0.5"]):

            history["best_model"] = val_dict
            ms.save_best_model(main_dict, model)
            
            ms.save_pkl(main_dict["path_best_annList"], pred_annList)


          # annList = ms.load_pkl(main_dict["path_best_annList"])
          # 
          # gtAnnDict = au.load_gtAnnDict(main_dict)
          # au.get_perSizeResults(gtAnnDict, annList)
      ms.save_pkl(main_dict["path_history"], history)


