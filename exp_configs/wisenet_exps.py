from haven import haven_utils as hu
import itertools, copy
EXP_GROUPS = {}

# count_list = []
# for network in ['fcn8_vgg16']:
#         for loss in [ 'att_lcfcn', 'lcfcn', ]:
#                 count_list += [{'name':'semseg_looc',
#                         # 'clip_grad':True,
#                         'n_classes':2,
#                                                 'base':network, 'n_channels':3, 
#                                                 'loss':loss}]
#                 count_list += [{'name':'semseg_looc',
#                         'n_classes':1,
#                                  'base':network, 'n_channels':3, 
#                                  'loss':loss}]
             
EXP_GROUPS["wisenet"] = hu.cartesian_exp_group({'model':{'name':'wisenet', 'loss':"one_head_sum_loss",
                                 'base':'fcn8_vgg16',
                                 'n_channels':3,'n_classes':1},
                 
                        "batch_size": 1,
                        "max_epoch": 500,
                        'dataset_size': [
                 {'train':10, 'val':10, 'test':10},
                #  {'train': 'all', 'val': 'all'},
            ],
                        
                        "optimizer" :"adam",
                        "lr": 1e-5,
                        "dataset": [
                        #  {'name':'pascal', 'transform':'flip', 'supervision':'seam', 'sbd':True},
                        #  {'name':'pascal', 'transform':'basic', 'supervision':'seam', 'sbd':True},
                        #  {'name':'pascal', 'transform':'basic', 'supervision':'full'},
                        {'name':'pascal', 'transform':'basic', 'supervision':'seam'},
                        #  {'name':'pascal', 'transform':'flip', 'supervision':'seam'},
                        
                                ],
                        "predict": "best_dice"})