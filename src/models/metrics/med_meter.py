import pandas as pd 
import numpy as np



class MedMeter:
    def __init__(self):
        self.val_list = []

    def val_on_batch(self, model, batch):
        
        if batch['masks'].sum()==0:
            pass
        else:
            image = batch['images']
            gt_mask = np.array(batch['masks'])
            prob_mask = model.predict_on_batch(batch)

            T = 0.1

            pred_mask = prob_mask > T
            
            # FP+TP
            NumRec = (pred_mask == 1).sum()
            # FN+TN
            NumNoRec = (pred_mask == 0).sum()
            # LabelAnd
            LabelAnd = pred_mask & gt_mask
            # TP
            NumAnd = (LabelAnd == 1).sum() 

            # TP + FN
            num_obj = gt_mask.sum()

            # FP + TP
            num_pred = pred_mask.sum()

            FN = num_obj-NumAnd;
            FP = NumRec-NumAnd;
            TN = NumNoRec-FN;

            val_dict = {}

            if NumAnd == 0:
                val_dict['PreFtem'] = 0
                val_dict['RecallFtem'] = 0
                val_dict['FmeasureF'] = 0
                val_dict['Dice'] = 0
                val_dict['SpecifTem'] = 0
            else:
                val_dict['PreFtem'] = NumAnd/NumRec
                val_dict['RecallFtem'] = NumAnd/num_obj
                val_dict['SpecifTem'] = TN/(TN+FP)
                val_dict['Dice'] = 2 * NumAnd/(num_obj+num_pred) 
                val_dict['FmeasureF'] = (( 2.0 * val_dict['PreFtem'] * val_dict['RecallFtem'] ) /
                                            (val_dict['PreFtem'] + val_dict['RecallFtem']))

            val_dict['%s_score' % batch['meta'][0]['split']] = val_dict['Dice']
            # pprint.pprint(val_dict)
            self.val_list += [val_dict]

    def get_avg_score(self):
        return pd.DataFrame(self.val_list).mean().to_dict()