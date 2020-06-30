from haven import haven_utils as hu
import itertools, copy
EXP_GROUPS = {}

model_list = []


# for loss_weight in [0.001]:
#     model_list += [ 
#         {'name':'semseg', 'loss':'toponet',
#                             'loss_weight': loss_weight,
#                                     'base':'fcn8_vgg16',
#                                     'n_channels':3,'n_classes':1},
#     ]
model_list += [ 
    {'name':'semseg', 'loss':'cons_point_loss',
                                 'base':'fcn8_vgg16',
                                  'n_channels':3,'n_classes':1},
    {'name':'semseg', 'loss':'point_loss',
                                 'base':'fcn8_vgg16',
                                  'n_channels':3,'n_classes':1},
    
                {'name':'semseg', 'loss':'joint_cross_entropy',
                                 'base':'fcn8_vgg16',
                                  'n_channels':3,'n_classes':1},

               
                # {'name':'semseg', 'loss':'att_point_loss',
                #                  'base':'fcn8_vgg16',
                #                   'n_channels':3,'n_classes':1},
            ]
model_count_list = [ 
            {'name':'semseg_counting', 'loss':'const_lcfcn_loss',
                                 'base':'fcn8_vgg16',
                                  'n_channels':3,'n_classes':1},

            {'name':'semseg_counting', 'loss':'point_loss',
                                 'base':'fcn8_vgg16',
                                  'n_channels':3,'n_classes':1},
            {'name':'semseg_counting', 'loss':'lcfcn_loss',
                                 'base':'fcn8_vgg16',
                                  'n_channels':3,'n_classes':1},
            ]

for dataset in ['covid19_v3_mixed', 'covid19_v3_sep',
                'covid19_v2_mixed', 'covid19_v2_sep',
                'covid19_v1']:
    for debug in [False]:
        if debug == True:
            prefix = 'debug_'
            dataset_size = {'train':3, 'val':2, 'test':2}
        else:
            prefix = ''
            dataset_size = {'train':'all', 'val':'all'}
        
        for n_classes in [2, 3]:
            if n_classes == 2:
                suffix = '_c2'
            elif n_classes == 3:
                suffix = '_c3'

            if n_classes == 3 and (dataset == 'covid19_v1' or dataset == 'covid19_v3_mixed'):
                continue
            EXP_GROUPS["%sweakly_%s%s" % (prefix, dataset, suffix)] = hu.cartesian_exp_group({
                    'batch_size': [8],
                    'num_channels':1,
                    'dataset': [
                            {'name':dataset, 'n_classes':n_classes},
                                ],
                    'dataset_size':dataset_size,
                    'max_epoch': [101],
                    'optimizer': [ "adam"],
                    'lr': [1e-4,],
                    'model': model_list
                    })

            EXP_GROUPS["%sweakly_%s%s_count" % (prefix, dataset, suffix)] = hu.cartesian_exp_group({
                                'batch_size': [8],
                                'num_channels':1,
                                'dataset': [
                                        {'name':dataset, 'n_classes':n_classes},
                                            ],
                                'dataset_size':dataset_size,
                                'max_epoch': [101],
                                'optimizer': [ "adam"],
                                'lr': [1e-4,],
                                'model': model_count_list
                                })


EXP_GROUPS["weakly_JCUfish"] = hu.cartesian_exp_group({
            'batch_size': [1],
            'num_channels': 1,
            'dataset': [
                {'name': 'JcuFish', 'n_classes': 2},
                # {'name': 'covid19_v2', 'n_classes': 2},
                # {'name':'covid19', 'n_classes':2},

            ],
            'dataset_size': [
                #  {'train':10, 'val':10, 'test':10},
                {'train': 'all', 'val': 'all'},
            ],
            'max_epoch': [100],
            'optimizer': ["adam"],
            'lr': [1e-4, ],
            'model': [
                    {'name': 'semseg', 'loss': 'joint_cross_entropy',
                 'base': 'fcn8_vgg16',
                 'n_channels': 3, 'n_classes': 1},

                {'name': 'semseg', 'loss': 'cons_point_loss',
                 'base': 'fcn8_vgg16',
                 'n_channels': 3, 'n_classes': 1},
                {'name': 'semseg', 'loss': 'att_point_loss',
                 'base': 'fcn8_vgg16',
                 'n_channels': 3, 'n_classes': 1},
                
                {'name': 'semseg', 'loss': 'point_loss',
                 'base': 'fcn8_vgg16',
                 'n_channels': 3, 'n_classes': 1},

    ]
})