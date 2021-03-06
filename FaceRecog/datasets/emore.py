import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
from torch.utils.data import Dataset
import cv2
import numpy as np
import pickle
import time
from tqdm import tqdm
import logging


class EmoreDataset(Dataset):

    def __init__(self,
                 data_dir='/data/data/emore',
                 img_dirname='released_data',
                 eval_mode=False,
                 eval_ratio=0.1,
                 shuffle=False):
        super(EmoreDataset, self).__init__()
        self.data_dir = data_dir
        self.img_dir = os.path.join(data_dir, img_dirname)
        self.eval_mode = eval_mode
        self.eval_ratio = eval_ratio
        self.img_fpath_list, self.cls_idx_list, self.idx2dirname = self._load_data_info()
        self.shuffle = shuffle
        if shuffle:
            self.img_fpath_list, self.cls_idx_list = self._shuffle_pairs(self.img_fpath_list, self.cls_idx_list)
            print('shuffled !')
        self.n_img = len(self.img_fpath_list)
        self.n_cls = len(self.idx2dirname)
        print('n_selected_img: {}'.format(self.n_img))
        print('n_cls: {}'.format(self.n_cls))
        self.n_count = 0

    def _load_data_info(self):
        # pickle_fpath = os.path.join('/data/data', '{}_data_load_pickle.pkl'.format(self.data_dir.split('/')[-1]))
        pickle_fpath = os.path.join(self.data_dir, 'preload_emore.pkl')
        if os.path.exists(pickle_fpath):
            print('pickle file for loading emore data already exists, load it.')
            start_time = time.time()
            with open(pickle_fpath, 'rb') as f:
                img_fpath_list, cls_idx_list, idx2dirname = pickle.load(f)[:]
            end_time = time.time()
            time_spend = end_time - start_time
            print('load_success, time spend: ', '%.3f' % time_spend, ', n_samples: ', len(img_fpath_list))
        else:
            print('pickle file for loading emore data not exist, start reading original data...')
            img_fpath_list, cls_idx_list, idx2dirname = self._preload_data_info()
            if len(img_fpath_list) < 1000:
                print('### ERROR: fail to load valid dataset !')
                exit(-1)
            # save load info
            with open(pickle_fpath, 'wb') as f:
                pickle.dump([img_fpath_list, cls_idx_list, idx2dirname], f)
                print('pickle file for loading emore data has been generated.')
        return img_fpath_list, cls_idx_list, idx2dirname

    def _preload_data_info(self):
        ch_dir_list = [ch_dir
                       for ch_dir in os.listdir(self.img_dir)
                       if os.path.isdir(os.path.join(self.img_dir, ch_dir))]
        if not self.eval_mode:
            selected_dir_list = ch_dir_list[0: int(len(ch_dir_list) * (1 - self.eval_ratio))]
        else:
            selected_dir_list = ch_dir_list[int(len(ch_dir_list) * (1 - self.eval_ratio)):]
        # selected_dir_list = selected_dir_list[0:5000]
        print('n_chdir: {}'.format(len(selected_dir_list)))
        # load img_fpath & label
        img_fpath_list = []
        cls_idx_list = []
        idx2dirname = {}
        pbar = tqdm(total=len(selected_dir_list), desc='emore_preload')
        for ch_idx, ch_dir in enumerate(selected_dir_list):
            ch_dir_path = os.path.join(self.img_dir, ch_dir)
            sub_img_fpath_list = [os.path.join(ch_dir_path, img_fn)
                                  for img_fn in os.listdir(ch_dir_path)
                                  if img_fn.endswith('.jpg')]
            img_fpath_list.extend(sub_img_fpath_list)
            cls_idx_list.extend([ch_idx] * len(sub_img_fpath_list))
            idx2dirname.setdefault(ch_idx, ch_dir)
            pbar.update(1)
        pbar.close()
        return img_fpath_list, cls_idx_list, idx2dirname

    def _shuffle_pairs(self, keys, values):
        pairs = [{'key': k, 'value': v} for k, v in zip(keys, values)]
        random.shuffle(pairs)
        keys = [pair['key'] for pair in pairs]
        values = [pair['value'] for pair in pairs]
        return keys, values

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

    def get_n_cls(self):
        return self.n_cls

    def __getitem__(self, idx):
        img_fpath = self.img_fpath_list[idx]
        img_cv2 = cv2.imread(img_fpath)
        img_cv2 = self._format(img_cv2)
        img_data = self._normalize(img_cv2)
        img_data = np.transpose(img_data, axes=[2, 0, 1])
        cls_idx = self.cls_idx_list[idx]
        self.n_count += 1
        if self.n_count >= self.n_img and self.shuffle:
            self.img_fpath_list, self.cls_idx_list = self._shuffle_pairs(self.img_fpath_list, self.cls_idx_list)
            print('shuffled !')
            self.n_count = 0
        return img_data, cls_idx


if __name__ == '__main__':

    dataset = EmoreDataset()
    print(len(dataset))
    print(dataset.n_cls)
    for sample in dataset:
        img_data, cls_idx = sample
        print(img_data.shape)
        print(cls_idx)