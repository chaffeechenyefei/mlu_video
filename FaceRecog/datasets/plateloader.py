import os
import cv2
import torch
import numpy as np
import random

from torch.utils.data import Dataset

pj = os.path.join

class BasicLoader(Dataset):
    def __init__(self, imgs_dir,extstr = '.jpg',expand=1,center_crop=False,rsz = 224,dsz=112,restrict=True):
        super().__init__()
        self.expand = expand
        self.imgs_dir = imgs_dir
        self.names = [ i for i in os.listdir(self.imgs_dir) if i.endswith(extstr) and not i.startswith('.')  ]
        self.imgs = [os.path.join(self.imgs_dir,i) for i in self.names ]
        self.epoch_size = len(self.imgs)

        self.FLG_center_crop = center_crop
        self.FLG_restrict = restrict
        self.rsz = rsz
        self.dsz = dsz

    def __len__(self):
        return self.expand*self.epoch_size

    def _format(self, img_cv2, format_size=112):
        org_h, org_w = img_cv2.shape[0:2]
        rescale_ratio = format_size / max(org_h, org_w)
        h, w = int(org_h * rescale_ratio), int(org_w * rescale_ratio)
        img_rescaled = cv2.resize(img_cv2, (w, h))
        paste_pos = [int((format_size - w)/2), int((format_size - h)/2)]
        img_format = np.zeros((format_size, format_size, 3), dtype=np.uint8)
        img_format[paste_pos[1]:paste_pos[1]+h, paste_pos[0]:paste_pos[0]+w] = img_rescaled
        return img_format

    def _center_crop(self,img_cv2):
        assert img_cv2.shape[0] == img_cv2.shape[1], '_center_crop_size_err_'
        if self.FLG_restrict:
            assert img_cv2.shape[0] == self.rsz, '_input_size_err_'
        else:
            img_cv2 = cv2.resize(img_cv2,(self.rsz,self.rsz))
        margin = int((self.rsz - self.dsz)/2)
        img = img_cv2[margin:-margin,margin:-margin]
        img = self._format(img_cv2=img,format_size=self.dsz)
        return img


    def _normalize(self, img_cv2):
        img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        # mean = [123.675, 116.28, 103.53]
        # std = [58.395, 57.12, 57.375]
        mean = 127.5
        std = 127.5
        img_data = np.asarray(img_cv2, dtype=np.float32)
        img_data = img_data - mean
        img_data = img_data / std
        img_data = img_data.astype(np.float32)
        return img_data

    def __getitem__(self, i):
        idx = i%self.epoch_size
        img_file = self.imgs[idx]

        try:
            img_cv2 = cv2.imread(img_file)
            if self.FLG_center_crop:
                img_format = self._center_crop(img_cv2)
            else:
                img_format = self._format(img_cv2,self.dsz)
            # img_format = self._format(img_cv2)
            img_data = self._normalize(img_format)
            img_data = np.transpose(img_data, axes=[2, 0, 1])
        except:
            print('Error %s'%img_file)
            img_data = np.zeros([3,112,112])
        # img_data = torch.FloatTensor(img_data)
        return {'image': torch.tensor(img_data, dtype=torch.float32),'name':self.names[idx],'imgpath':img_file}


class BasicLoaderV2(Dataset):
    """
    Each sub-folder is used as a class of person
    """
    def __init__(self, imgs_dir,extstr = '.jpg',expand=1,center_crop=False,rsz = 224,dsz=112,restrict=True):
        super().__init__()
        self.expand = expand
        self.imgs_dir = imgs_dir
        self.FLG_center_crop = center_crop
        self.FLG_restrict = restrict
        self.rsz = rsz
        self.dsz = dsz
        self.subfolder = [ i for i in os.listdir(self.imgs_dir) if os.path.isdir(pj(self.imgs_dir,i)) ]
        print('subfolder num:%d'%len(self.subfolder))
        self.imgs = []
        self.names = []
        for subfd in self.subfolder:
            subimgs = [pj(self.imgs_dir,subfd,i) for i in os.listdir(pj(self.imgs_dir,subfd)) if i.endswith(extstr)]
            L = len(subimgs)
            self.imgs.extend(subimgs)
            self.names.extend([subfd]*L )

        self.epoch_size = len(self.imgs)
        print('Epoch size is %d'%self.epoch_size)

    def __len__(self):
        return self.expand*self.epoch_size

    def _format(self, img_cv2, format_size=112):
        org_h, org_w = img_cv2.shape[0:2]
        rescale_ratio = format_size / max(org_h, org_w)
        h, w = int(org_h * rescale_ratio), int(org_w * rescale_ratio)
        img_rescaled = cv2.resize(img_cv2, (w, h))
        paste_pos = [int((format_size - w)/2), int((format_size - h)/2)]
        img_format = np.zeros((format_size, format_size, 3), dtype=np.uint8)
        img_format[paste_pos[1]:paste_pos[1]+h, paste_pos[0]:paste_pos[0]+w] = img_rescaled
        return img_format

    def _center_crop(self,img_cv2):
        assert img_cv2.shape[0] == img_cv2.shape[1], '_center_crop_size_err_'
        if self.FLG_restrict:
            assert img_cv2.shape[0] == self.rsz, '_input_size_err_'
        else:
            img_cv2 = cv2.resize(img_cv2,(self.rsz,self.rsz))
        margin = int((self.rsz - self.dsz)/2)
        img = img_cv2[margin:-margin,margin:-margin]
        img = self._format(img_cv2=img,format_size=self.dsz)
        return img


    def _normalize(self, img_cv2):
        img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        # mean = [123.675, 116.28, 103.53]
        # std = [58.395, 57.12, 57.375]
        mean = 127.5
        std = 127.5
        img_data = np.asarray(img_cv2, dtype=np.float32)
        img_data = img_data - mean
        img_data = img_data / std
        img_data = img_data.astype(np.float32)
        return img_data

    def __getitem__(self, i):
        idx = i%self.epoch_size
        img_file = self.imgs[idx]
        fdname = self.names[idx]

        try:
            img_cv2 = cv2.imread(img_file)
            if self.FLG_center_crop:
                img_format = self._center_crop(img_cv2)
            else:
                img_format = self._format(img_cv2)
            img_data = self._normalize(img_format)
            img_data = np.transpose(img_data, axes=[2, 0, 1])
        except:
            print('Error %s'%img_file)
            img_data = np.zeros([3,112,112])

        # img_data = torch.FloatTensor(img_data)
        return {'image': torch.tensor(img_data, dtype=torch.float32),'fdname':fdname,'name':img_file}


class BasicLoader_ext(Dataset):
    def __init__(self, imgs_dir,extstr = '.jpg',expand=1,center_crop=False,rsz = 224,dsz=112,restrict=True):
        super().__init__()
        self.expand = expand
        self.imgs_dir = imgs_dir
        self.names = [ i for i in os.listdir(self.imgs_dir) if i.endswith(extstr) and not i.startswith('.')  ]
        self.imgs = [os.path.join(self.imgs_dir,i) for i in self.names ]
        self.epoch_size = len(self.imgs)

        self.FLG_center_crop = center_crop
        self.FLG_restrict = restrict
        self.rsz = rsz
        self.dsz = dsz

    def __len__(self):
        return self.expand*self.epoch_size

    def _format(self, img_cv2, format_size=112):
        org_h, org_w = img_cv2.shape[0:2]
        rescale_ratio = format_size / max(org_h, org_w)
        h, w = int(org_h * rescale_ratio), int(org_w * rescale_ratio)
        img_rescaled = cv2.resize(img_cv2, (w, h))
        paste_pos = [int((format_size - w)/2), int((format_size - h)/2)]
        img_format = np.zeros((format_size, format_size, 3), dtype=np.uint8)
        img_format[paste_pos[1]:paste_pos[1]+h, paste_pos[0]:paste_pos[0]+w] = img_rescaled
        return img_format

    def _center_crop(self,img_cv2):
        assert img_cv2.shape[0] == img_cv2.shape[1], '_center_crop_size_err_'
        if self.FLG_restrict:
            assert img_cv2.shape[0] == self.rsz, '_input_size_err_'
        else:
            img_cv2 = cv2.resize(img_cv2,(self.rsz,self.rsz))
        margin = int((self.rsz - self.dsz)/2)
        img = img_cv2[margin:-margin,margin:-margin]
        img = self._format(img_cv2=img,format_size=self.dsz)
        return img


    def _normalize(self, img_cv2):
        img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        # mean = [123.675, 116.28, 103.53]
        # std = [58.395, 57.12, 57.375]
        mean = 127.5
        std = 127.5
        img_data = np.asarray(img_cv2, dtype=np.float32)
        img_data = img_data - mean
        img_data = img_data / std
        img_data = img_data.astype(np.float32)
        return img_data

    def __getitem__(self, i):
        idx = i%self.epoch_size
        img_file = self.imgs[idx]

        try:
            img_cv2 = cv2.imread(img_file)
            if self.FLG_center_crop:
                img_format = self._center_crop(img_cv2)
            else:
                img_format = self._format(img_cv2,self.dsz)

            img_data = self._normalize(img_format)
            img_format = np.float32(img_format)
            img_format -= (104, 117, 123)
            img_format = img_format.transpose(2, 0, 1)
            assert img_format.shape[0] == 3, 'Err: img_format shape'

            img_data = np.transpose(img_data, axes=[2, 0, 1])
        except:
            print('Error %s'%img_file)
            img_data = np.zeros([3,112,112])
            img_format = np.zeros([3,112,112])

        # img_data = torch.FloatTensor(img_data)
        return {'image': torch.tensor(img_data, dtype=torch.float32),
                'det_image':torch.FloatTensor(img_format),
                'name':self.names[idx],'imgpath':img_file}