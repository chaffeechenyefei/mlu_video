import os
import cv2
import torch
import numpy as np
import random

from torch.utils.data import Dataset

pj = os.path.join

class BasicLoader(Dataset):
    def __init__(self, imgs_dir,extstr = '.jpg'):
        super().__init__()
        self.imgs_dir = imgs_dir
        self.names = [i for i in os.listdir(self.imgs_dir) if i.endswith(extstr)]
        self.imgs = [os.path.join(self.imgs_dir,i) for i in os.listdir(self.imgs_dir) if i.endswith(extstr) ]
        self.epoch_size = len(self.imgs)

    def __len__(self):
        return self.epoch_size

    def format_det_img(self,img_cv):
        img = np.float32(img_cv)
        im_height, im_width, _ = img.shape
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)
        return img


    def __getitem__(self, i):
        idx = i%self.epoch_size
        img_file = self.imgs[idx]

        img_cv2 = cv2.imread(img_file)
        img_data = self.format_det_img(img_cv2)
        # img_data = torch.FloatTensor(img_data)
        return {'image': img_data,'name':self.names[idx]}



def face_format(img_cv2, format_size=112):
    org_h, org_w = img_cv2.shape[0:2]
    rescale_ratio = format_size / max(org_h, org_w)
    h, w = int(org_h * rescale_ratio), int(org_w * rescale_ratio)
    img_rescaled = cv2.resize(img_cv2, (w, h))
    paste_pos = [int((format_size - w) / 2), int((format_size - h) / 2)]
    img_format = np.zeros((format_size, format_size, 3), dtype=np.uint8)
    img_format[paste_pos[1]:paste_pos[1] + h, paste_pos[0]:paste_pos[0] + w] = img_rescaled
    return img_format

class BasicLoader_unsized(Dataset):
    def __init__(self, imgs_dir,extstr = '.jpg', sz = 300 ):
        super().__init__()
        self.imgs_dir = imgs_dir
        self.names = [i for i in os.listdir(self.imgs_dir) if i.endswith(extstr)]
        self.imgs = [os.path.join(self.imgs_dir,i) for i in os.listdir(self.imgs_dir) if i.endswith(extstr) ]
        self.epoch_size = len(self.imgs)
        self.sz = sz

    def __len__(self):
        return self.epoch_size

    def format_det_img(self,img_cv):
        img = np.float32(img_cv)
        im_height, im_width, _ = img.shape
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)
        return img


    def __getitem__(self, i):
        idx = i%self.epoch_size
        img_file = self.imgs[idx]

        img_cv2 = cv2.imread(img_file)

        img_cv2 = face_format(img_cv2,format_size=self.sz)

        img_data = self.format_det_img(img_cv2)

        img_cv2 = img_cv2.transpose(2,0,1)
        # img_data = torch.FloatTensor(img_data)
        return {'image': img_data,'name':self.names[idx],'src':img_cv2}