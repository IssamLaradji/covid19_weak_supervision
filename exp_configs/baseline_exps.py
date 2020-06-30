from haven import haven_utils as hu
import itertools, copy
EXP_GROUPS = {}

EXP_GROUPS["infnet"] = hu.cartesian_exp_group({
        'batch_size': 8,
        'num_channels':1,
        'dataset': [
                {'name':'covid19_v2_mixed', 'n_classes':2},
                {'name':'covid19_v2', 'n_classes':2},
                # {'name':'covid19', 'n_classes':2},
                
                
                    ],
        'dataset_size':[
                
                 {'train':10, 'val':'all'},
                #  {'train':10, 'val':'all'},
                 {'train':15, 'val':'all'},
                 {'train':20, 'val':'all'},
                 {'train':25, 'val':'all'},
                  {'train':30, 'val':'all'},
                   {'train':35, 'val':'all'},
                {'train':'all', 'val':'all'},
                ],
        'max_epoch': [100],
        'optimizer': [ "adam"], 
        'lr': [ 1e-4,],
        'model': [
                #  {'name':'semseg', 'loss':'joint_cross_entropy',
                #                  'base':'fcn8_vgg16',"clip_grad":True,
                #                   'n_channels':3,'n_classes':1},

                # {'name':'semseg', 'loss':'joint_cross_entropy',
                #                  'base':'unet_resnet',
                #                   'n_channels':3,'n_classes':1},
                # {'name':'semseg', 'loss':'joint_cross_entropy',
                #                  'base':'unet_resnet',"clip_grad":True,
                #                   'n_channels':3,'n_classes':1},

                {'name':'semseg', 'loss':'joint_cross_entropy',
                                 'base':'fcn8_vgg16',
                                  'n_channels':3,'n_classes':1},
               
                   {'name':'semseg', 'loss':'joint_only',
                                 'base':'infnet', 'n_channels':3},
            ]
        })
