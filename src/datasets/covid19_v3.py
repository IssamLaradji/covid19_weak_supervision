import torch
import os
import numpy as np
from haven import haven_utils as hu
from torchvision import transforms
import PIL
try:
    import nibabel as nib
except:
    print('covid19_v3 requires nibabel')

from PIL import Image


class Covid19V3(torch.utils.data.Dataset):
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

        self.img_path = os.path.join(datadir, 'CT')
        self.lung_path = os.path.join(datadir, 'Lung_Mask')
        self.tgt_path = os.path.join(datadir, 'Infection_Mask')

        fname = os.path.join(datadir,'data_tuple.pkl')
        if not os.path.exists(fname):   
            self.img_list = []
            scan_list = set()
            for tgt_name in os.listdir(self.tgt_path):
                lung_name = tgt_name
                img_name = tgt_name
                scan_list.add(img_name)
                slices = nib.load(os.path.join(self.tgt_path, tgt_name)).get_data().shape[-1]
                for idx in range(slices):
                    self.img_list += [{'img': img_name,
                                    'tgt': tgt_name,
                                    'scan_id': img_name,
                                    'slice': idx,
                                    'lung': lung_name}]
            hu.save_pkl(fname, (self.img_list, scan_list))

        self.img_list, scan_list = hu.load_pkl(fname)

        scan_list = list(scan_list)
        scan_list.sort()

        if seperate:
            # TODO: Modify indices for v3 opensource
            if split == 'train':
                scan_list = scan_list[:15]
            elif split == 'val':
                scan_list = scan_list[15:16]
            elif split == 'test':
                scan_list = scan_list[16:]

            img_list_new = []
            for img_dict in self.img_list:
                if img_dict['scan_id'] in scan_list:
                    img_list_new += [img_dict]
        else:
            img_list_new = []
            for scan in scan_list:
                img_list = [img_dict for img_dict in self.img_list if img_dict['scan_id'] == scan]
                if split == 'train':
                    s = 0
                    e = int(0.45 * len(img_list))
                elif split == 'val':
                    s = int(0.45 * len(img_list))
                    e = int(0.5 * len(img_list))
                elif split == 'test':
                    s = int(0.5 * len(img_list))
                    e = len(img_list)
                img_list_new += img_list[s:e]

        self.img_list = img_list_new
        self.img_transform = transforms.Compose([

            transforms.CenterCrop((384, 385)),
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        if split == 'train':
            self.gt_transform = transforms.Compose([
                transforms.CenterCrop((384, 385)),
                transforms.Resize((self.size, self.size), interpolation=PIL.Image.NEAREST),
                # transforms.ToTensor()
                ])
        else:
            self.gt_transform = transforms.Compose([
                transforms.CenterCrop((384, 385)),
                # transforms.ToTensor()
            ])

    def __getitem__(self, i):
        out = self.img_list[i]
        img_name, tgt_name, slice_idx, lung_name = out['img'], out['tgt'], out['slice'], out['lung']

        fname = os.path.join(self.img_path, img_name+'_%d.pkl' % slice_idx)
        if not os.path.exists(fname):
            # read image
            img_nii = nib.load(os.path.join(self.img_path, img_name))
            image = img_nii.get_data()
            img_max = image.max()
            img_min = image.min()
            image = image[:, :, slice_idx]
            image = np.rot90(image, axes=(0, 1))

            # read infection mask
            tgt_mask = nib.load(os.path.join(self.tgt_path, tgt_name)).get_data()
            tgt_mask = tgt_mask[:, :, slice_idx]
            tgt_mask = np.rot90(tgt_mask, axes=(0, 1))
            
        
            # read lung mask
            # lung_mask = nib.load(os.path.join(self.lung_path, lung_name)).get_data()[:, :, slice_idx]
            # lung_mask = np.rot90(lung_mask, axes=(0, 1))

            mask = np.zeros(tgt_mask.shape)
            if self.n_classes == 2:
                mask[tgt_mask!= 0] = 1

            elif self.n_classes == 3:
                mask[tgt_mask== 1] = 1
                mask[tgt_mask== 2] = 2
                mask[tgt_mask== 3] = 0
            assert self.n_classes == 2
            # mask[tgt_mask != 0] = 1
            # assert that these are the only classes
            # print(np.unique(tgt_mask))
            assert (len(np.setdiff1d(np.unique(tgt_mask), [0, 1])) == 0)
            # assert (len(np.setdiff1d(np.unique(lung_mask), [0, 1, 2])) == 0)

            img_scaled = (((image - img_min) / (img_max - img_min)))
            img_scaled = (img_scaled * 255).astype('uint8')
            hu.save_pkl(fname, (img_scaled, mask))

        img_scaled, mask = hu.load_pkl(fname)
        image = self.img_transform(Image.fromarray(img_scaled).convert('RGB'))
        mask = self.gt_transform(Image.fromarray((mask).astype('uint8')))
        mask = torch.LongTensor(np.array(mask))
        
        assert (len(np.setdiff1d(np.unique(mask), [0, 1])) == 0)
        from src.modules.lcfcn import lcfcn_loss
        points = lcfcn_loss.get_points_from_mask(mask.numpy().squeeze(), bg_points=-1)

        return {'images': image,
                'masks': mask.long()[None],
                'points': torch.LongTensor(points),
                'meta': {'shape': mask.squeeze().shape,
                         'index': i,
                        #  'size': self.size,
                         'hash': hu.hash_dict({'id': os.path.join(self.img_path, img_name, str(slice_idx))}),
                         'name': img_name,
                        #  'slice_thickness': img_nii.header['pixdim'][1:4],
                        #  'pixel_spacing': str(img_nii.header['pixdim'][1:4]),
                         'img_name': img_name,
                         'tgt_name': tgt_name,
                         'image_id': slice_idx,
                         'split': self.split}}

    def __len__(self):
        return len(self.img_list)
