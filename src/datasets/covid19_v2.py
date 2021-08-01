import torch
import os
import h5py
import numpy as np
from haven import haven_utils as hu
from torchvision import transforms
import pydicom, tqdm
from PIL import Image
import PIL

class Covid19V2(torch.utils.data.Dataset):
    def __init__(
        self,
        split,
        datadir,
        exp_dict,
        seperate=True,
    ):
        self.exp_dict = exp_dict
        self.datadir = datadir
        self.split = split
        self.n_classes = exp_dict['dataset']['n_classes']
        self.size = 352


        self.img_path = os.path.join(datadir, 'OpenSourceDCMs')
        self.lung_path = os.path.join(datadir, 'LungMasks')
        self.tgt_path = os.path.join(datadir, 'InfectionMasks')

        self.img_list = []
        scan_list = set()
        for tgt_name in os.listdir(self.tgt_path):
            lung_name = tgt_name
            scan_id, slice_id = tgt_name.split('_')
            scan_list.add(int(scan_id))

            slice_id = str(int(slice_id.replace('z', '').replace('.png', ''))).zfill(4)
            img_name = [f for f in os.listdir(os.path.join(self.img_path, 
                                    'DCM'+scan_id)) if 's%s' % slice_id in f][0]
            img_name = os.path.join('DCM'+scan_id, img_name)

            self.img_list += [{'img': img_name, 
                                   'tgt': tgt_name,
                                   'lung': lung_name,
                                   'scan_id':int(scan_id),
                                   'slice_id':int(slice_id)}]
        scan_list = list(scan_list)
        scan_list.sort()

        if seperate:
            if split == 'train':
                scan_list = scan_list[:5]
            elif split == 'val':
                scan_list = scan_list[5:6]
            elif split == 'test':
                scan_list = scan_list[6:]

            img_list_new = []
            for img_dict in self.img_list:
                if img_dict['scan_id'] in scan_list:
                    img_list_new += [img_dict]
        else:

            img_list_new = []
            for scan in scan_list:
                img_list = [img_dict for img_dict in self.img_list if img_dict['scan_id']==scan]
                if split == 'train':
                    s = 0
                    e = int(0.45*len(img_list))

                elif split == 'val':
                    s = int(0.45*len(img_list))
                    e = int(0.5*len(img_list))
                    

                elif split == 'test':
                    s = int(0.5*len(img_list))
                    e = len(img_list)
                img_list_new += img_list[s:e]
                    
        self.img_list = img_list_new

        self.img_transform = transforms.Compose([

            transforms.CenterCrop((384, 385)),
            # transforms.Resize((self.size, self.size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
        if split == 'train':
            self.gt_transform = transforms.Compose([

            transforms.CenterCrop((384, 385)),
            # transforms.Resize((self.size, self.size), interpolation=PIL.Image.NEAREST),
                # transforms.ToTensor()]
            ])
        else:
            self.gt_transform = transforms.Compose([
                transforms.CenterCrop((384, 385)),
                # transforms.ToTensor()
            ])

        

    def __getitem__(self, i):
        out = self.img_list[i]
        img_name, tgt_name, lung_name = out['img'], out['tgt'], out['lung']

        # read image
        img_dcm = pydicom.dcmread(os.path.join(self.img_path, img_name))
        image = img_dcm.pixel_array.astype('float')

        # read infection mask
        tgt_mask = np.array(Image.open(os.path.join(self.tgt_path, tgt_name)).transpose(Image.FLIP_LEFT_RIGHT).rotate(90))
        
        # read lung mask
        lung_mask = np.array(Image.open(os.path.join(self.lung_path, 
                             lung_name)).transpose(Image.FLIP_LEFT_RIGHT))
        mask = np.zeros(lung_mask.shape)
        # mask[lung_mask== 255] = 1
        # mask[tgt_mask== 127] = 2
        # mask[tgt_mask== 255] = 3
        if self.n_classes == 2:
            mask[tgt_mask!= 0] = 1

        elif self.n_classes == 3:
            mask[tgt_mask== 127] = 1
            mask[tgt_mask== 255] = 2
        
        # assert that these are the only classes
        assert(len(np.setdiff1d(np.unique(tgt_mask),[0,127,255] ))==0)
        assert(len(np.setdiff1d(np.unique(lung_mask),[0,255] ))==0)

        # image, mask = transformers.apply_transform(self.split, image=image, label=mask, 
        #                                transform_name=self.exp_dict['dataset']['transform'], 
        #                                exp_dict=self.exp_dict)
        img_uint8 = ((image/4095)*255).astype('uint8')
        image = self.img_transform(Image.fromarray(img_uint8).convert('RGB'))
        mask = self.gt_transform(Image.fromarray((mask).astype('uint8')))
        mask = torch.LongTensor(np.array(mask))

        if self.n_classes == 2:
            assert (len(np.setdiff1d(np.unique(mask), [0, 1])) == 0)
        if self.n_classes == 3:
            assert (len(np.setdiff1d(np.unique(mask), [0, 1, 2])) == 0)

        from src.modules.lcfcn import lcfcn_loss
        points =  lcfcn_loss.get_points_from_mask(mask.numpy().squeeze(), bg_points=-1)
        # if (points == 255).mean() == 1:
        #     points[:] = 0
        return {'images': image, 
                'masks': mask.long()[None],
                'points':torch.LongTensor(points),
                'meta': {'shape':mask.squeeze().shape, 
                        'index':i,
                        'hash':hu.hash_dict({'id':os.path.join(self.img_path, img_name)}),
                          'name':img_name,
                         'slice_thickness':img_dcm.SliceThickness, 
                         'pixel_spacing':str(img_dcm.PixelSpacing), 
                         'img_name': img_name, 
                         'tgt_name':tgt_name,
                         'image_id': i,
                         'split': self.split}}

    def __len__(self):
        return len(self.img_list)
