import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
from torch.utils.data import Dataset
import cv2
import numpy as np


class FaceDataset(Dataset):

    def __init__(self,
                 img_dir='/data/data/celebA/Img/img_align_celeba',
                 label_dir='/data/data/celebA/Anno',
                 eval_mode=False,
                 eval_ratio=0.1,
                 format_size=128):
        super(FaceDataset, self).__init__()
        self.img_dir = img_dir
        self.label_fpath = os.path.join(label_dir, 'identity_CelebA.txt')
        self.eval_mode = eval_mode
        self.eval_ratio = eval_ratio
        self.format_size = format_size
        self.selected_img_fn_list, self.label_dict = self._load_data_info()
        self.n_img = len(self.selected_img_fn_list)
        self.max_person_id = max(self.label_dict.values())
        print('n_selected_img: {}'.format(self.n_img))
        print('max_person_id: {}'.format(self.max_person_id))

    def _load_data_info(self):
        img_fn_list = os.listdir(self.img_dir)
        if not self.eval_mode:
            selected_img_fn_list = img_fn_list[0: int(len(img_fn_list) * (1 - self.eval_ratio))]
        else:
            selected_img_fn_list = img_fn_list[int(len(img_fn_list) * (1 - self.eval_ratio)):]
        # load labels
        label_dict = {}
        with open(self.label_fpath, 'r') as f:
            lines = f.readlines()
            # print(len(lines))
            for line in lines:
                img_fn, person_id = line.strip().split(' ')[0:2]
                # print(img_fn, person_id)
                label_dict[img_fn.strip()] = int(person_id) - 1
        return selected_img_fn_list, label_dict

    def _format(self, img_cv2, format_size=128):
        org_h, org_w = img_cv2.shape[0:2]
        rescale_ratio = format_size / max(org_h, org_w)
        h, w = int(org_h * rescale_ratio), int(org_w * rescale_ratio)
        img_rescaled = cv2.resize(img_cv2, (w, h))
        paste_pos = [int((format_size - w)/2), int((format_size - h)/2)]
        img_format = np.zeros((format_size, format_size, 3), dtype=np.uint8)
        img_format[paste_pos[1]:paste_pos[1]+h, paste_pos[0]:paste_pos[0]+w] = img_rescaled
        return img_format


    def _normalize(self, img_cv2):
        img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        mean = [123.675, 116.28, 103.53]
        std = [58.395, 57.12, 57.375]
        img_data = np.asarray(img_cv2, dtype=np.float32)
        img_data = img_data - mean
        img_data = img_data / std
        img_data = img_data.astype(np.float32)
        return img_data

    def __len__(self):
        return self.n_img

    def __getitem__(self, idx):
        img_fn = self.selected_img_fn_list[idx]
        img_fpath = os.path.join(self.img_dir, img_fn)
        img_cv2 = cv2.imread(img_fpath)
        img_cv2 = self._format(img_cv2, format_size=self.format_size)
        img_data = self._normalize(img_cv2)
        img_data = np.transpose(img_data, axes=[2, 0, 1])
        person_id = self.label_dict[img_fn]
        return img_data, person_id


if __name__ == '__main__':

    dataset = FaceDataset()
    for sample in dataset:
        img_data, person_id = sample
        print(img_data.shape)
        print(person_id)
