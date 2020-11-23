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


class MultiData(Dataset):

    def __init__(self,
                 datasets=[
                     dict(
                         data_type='emore',
                         data_dir='/data/data/emore/released_data',
                         pickle_fpath='/data/data/emore/preload_emore.pkl',
                         n_train=10000,
                     ),
                     dict(
                         data_type='hegui',
                         data_dir='/data/data/hegui_faces_expand20',
                         pickle_fpath='/data/data/hegui_faces/preload_hegui.pkl',
                         n_train=800,
                     ),
                     # dict(
                     #     data_type='casia_facev5',
                     #     data_dir='/data/data/casia_facev5',
                     #     pickle_fpath='/data/data/casia_facev5/preload_casia.pkl'
                     # ),
                 ],
                 eval_mode=False,
                 eval_ratio=0.1,
                 shuffle=False):
        super(MultiData, self).__init__()
        self.datasets = datasets
        self.eval_mode = eval_mode
        self.eval_ratio = eval_ratio


        self.img_fpath_list, self.cls_idx_list, self.idx2dirname = self._load_data_info()
        if shuffle:
            self.img_fpath_list, self.cls_idx_list = self._shuffle_pairs(self.img_fpath_list, self.cls_idx_list)
            print('shuffled !')
        self.n_img = len(self.img_fpath_list)
        self.n_cls = len(self.idx2dirname)
        print('n_selected_img: {}'.format(self.n_img))
        print('n_cls: {}'.format(self.n_cls))

        # hegui data

    def _load_emore(self, emore_dataset):
        data_dir = emore_dataset['data_dir']
        pickle_fpath = emore_dataset['pickle_fpath']
        n_train = emore_dataset['n_train']
        if os.path.exists(pickle_fpath):
            print('pickle file for loading emore data already exists, load it.')
            start_time = time.time()
            with open(pickle_fpath, 'rb') as f:
                img_fpath_list, cls_idx_list, idx2dirname = pickle.load(f)[:]
                n_train = len(idx2dirname)
            end_time = time.time()
            time_spend = end_time - start_time
            print('load_success, time spend: ', '%.3f' % time_spend, ', n_samples: ', len(img_fpath_list))
        else:
            print('pickle file for loading emore data not exist, start reading original data...')
            n_train, img_fpath_list, cls_idx_list = self._preload_data_info(data_dir, n_train)
            if len(img_fpath_list) < 1000:
                print('### ERROR: fail to load valid dataset !')
                exit(-1)
            # save load info
            with open(pickle_fpath, 'wb') as f:
                pickle.dump([img_fpath_list, cls_idx_list, {}], f)
                print('pickle file for loading emore data has been generated.')
        return n_train, img_fpath_list, cls_idx_list

    def _preload_data_info(self, data_dir, n_train):
        ch_dir_list = [ch_dir
                       for ch_dir in os.listdir(data_dir)
                       if os.path.isdir(os.path.join(data_dir, ch_dir))]
        selected_dir_list = ch_dir_list[0:n_train]
        n_load = len(selected_dir_list)
        print('n_chdir_selected: {}'.format(n_load))
        # load img_fpath & label
        img_fpath_list = []
        cls_idx_list = []
        idx2dirname = {}
        pbar = tqdm(total=len(selected_dir_list), desc='emore_preload')
        for ch_idx, ch_dir in enumerate(selected_dir_list):
            ch_dir_path = os.path.join(data_dir, ch_dir)
            sub_img_fpath_list = [os.path.join(ch_dir_path, img_fn)
                                  for img_fn in os.listdir(ch_dir_path)
                                  if img_fn.endswith('.jpg')]
            img_fpath_list.extend(sub_img_fpath_list)
            cls_idx_list.extend([ch_idx] * len(sub_img_fpath_list))
            idx2dirname.setdefault(ch_idx, ch_dir)
            pbar.update(1)
        pbar.close()
        return n_load, img_fpath_list, cls_idx_list

    def _load_hegui_faces(self, hegui_dataset):
        data_dir = hegui_dataset['data_dir']
        # pickle_fpath = hegui_dataset['pickle_fpath']
        n_train = hegui_dataset['n_train']

        register_dir = os.path.join(data_dir, 'register')
        test_dir = os.path.join(data_dir, 'test')
        register_img_fn_list = os.listdir(register_dir)
        register_img_fn_list = register_img_fn_list[0: n_train]
        n_register = len(register_img_fn_list)
        test_img_fn_list = os.listdir(test_dir)
        img_fpath_list = []
        cls_idx_list = []
        for idx, r_img_fn in enumerate(register_img_fn_list):
            img_fpath_list.append(r_img_fn)
            cls_idx_list.append(idx)
            t_img_fn = r_img_fn.replace('0000', '0001')
            if t_img_fn in test_img_fn_list:
                img_fpath_list.append(t_img_fn)
                cls_idx_list.append(idx)
            else:
                print('{} not found.'.format(t_img_fn))
        print('hegui_faces, n_register:{}, n_img:{} '.format(n_register, len(img_fpath_list)))
        return n_register, img_fpath_list, cls_idx_list

    def _shuffle_pairs(self, keys, values):
        pairs = [{'key': k, 'value': v} for k, v in zip(keys, values)]
        random.shuffle(pairs)
        keys = [pair['key'] for pair in pairs]
        values = [pair['value'] for pair in pairs]
        return keys, values

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
        img_cv2 = cv2.resize(img_cv2, (128, 128))
        img_data = self._normalize(img_cv2)
        img_data = np.transpose(img_data, axes=[2, 0, 1])
        cls_idx = self.cls_idx_list[idx]
        return img_data, cls_idx


if __name__ == '__main__':

    dataset = MultiData()
    print(len(dataset))
    print(dataset.n_cls)
    for sample in dataset:
        img_data, cls_idx = sample
        print(img_data.shape)
        print(cls_idx)
