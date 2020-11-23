from __future__ import print_function

from torch.utils.data import Dataset
import os
import numpy as np
from tqdm import tqdm
import pickle
import time
import cv2
import random


class Preprocess(object):

    def __init__(self, format_size=112):
        super(Preprocess, self).__init__()
        self.format_size = format_size

    def _format(self, img_cv2):
        org_h, org_w = img_cv2.shape[0:2]
        rescale_ratio = self.format_size / max(org_h, org_w)
        h, w = int(org_h * rescale_ratio), int(org_w * rescale_ratio)
        img_rescaled = cv2.resize(img_cv2, (w, h))
        paste_pos = [int((self.format_size - w)/2), int((self.format_size - h)/2)]
        img_format = np.zeros((self.format_size, self.format_size, 3), dtype=np.uint8)
        img_format[paste_pos[1]:paste_pos[1]+h, paste_pos[0]:paste_pos[0]+w] = img_rescaled
        return img_format

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

    def __call__(self, img_cv2):
        img_format = self._format(img_cv2)
        img_data = self._normalize(img_format)
        img_data = np.transpose(img_data, axes=[2, 0, 1])
        return img_data




class TripletDataset(Dataset):

    def __init__(self,
                 finetune_dir='/data/FaceDetect/results/hegui_faces',
                 base_dir='/data/data/emore/released_data',
                 n_finetune_selected=0,
                 n_base_selected=20000,
                 finetune_ratio=0.0,
                 n_triplets=1000000):
        super(TripletDataset, self).__init__()
        self.finetune_dir = finetune_dir
        self.base_dir = base_dir
        self.n_finetune_seletected = n_finetune_selected
        self.n_base_selected = n_base_selected
        self.finetune_ratio = finetune_ratio
        self.n_triplets = n_triplets
        self.pickle_fpath = os.path.join(self.base_dir,
                                         'finetune_preload_{}_{}.pkl'.format(
                                             self.base_dir.split('/')[-1],
                                             n_base_selected,
                                         ))

        self.base_idx2fpathes = self._load_data_info()
        self.finetune_idx2fpathes = self._load_hegui_data_info()
        self.merge_idx2fpathes = {**self.base_idx2fpathes, **self.finetune_idx2fpathes}
        self.n_cls = len(self.merge_idx2fpathes)
        print('n_base:{}, n_finetune:{}, n_merge:{}'.format(self.n_base_selected, self.n_finetune_seletected, self.n_cls))
        self.triplet_groups = self._build_triplet_groups(self.merge_idx2fpathes)
        print('n_triplet_groups:{}'.format(len(self.triplet_groups)))
        self.preprocess_func = Preprocess()

    def __len__(self):
        return self.n_triplets

    def get_n_cls(self):
        return self.n_cls

    def _load_data_info(self):
        if os.path.exists(self.pickle_fpath):
            print('pickle file for loading emore data already exists, load it.')
            start_time = time.time()
            with open(self.pickle_fpath, 'rb') as f:
                idx2fpathes = pickle.load(f)
            end_time = time.time()
            time_spend = end_time - start_time
            print('load_success, time spend: ', '%.3f' % time_spend, ', n_identities: ', len(idx2fpathes))
        else:
            print('pickle file for loading emore data not exist, start reading original data...')
            idx2fpathes = self._preload_data(self.n_base_selected)
            if len(idx2fpathes) < 10:
                print('### ERROR: fail to load valid dataset !')
                exit(-1)
            # save load info
            with open(self.pickle_fpath, 'wb') as f:
                pickle.dump(idx2fpathes, f)
                print('pickle file for loading emore data has been generated.')
        return idx2fpathes

    def _preload_data(self, n_identities):
        ch_dir_list = [ch_dir
                       for ch_dir in os.listdir(self.base_dir)
                       if os.path.isdir(os.path.join(self.base_dir, ch_dir))]
        selected_dir_list = ch_dir_list[0: n_identities]
        print('n_chdir: {}'.format(len(selected_dir_list)))
        idx2fpathes = {}
        pbar = tqdm(total=len(selected_dir_list), desc='finetune_emore_preload')
        for ch_idx, ch_dir in enumerate(selected_dir_list):
            ch_dir_path = os.path.join(self.base_dir, ch_dir)
            sub_img_fpath_list = [os.path.join(ch_dir_path, img_fn)
                                  for img_fn in os.listdir(ch_dir_path)
                                  if img_fn.endswith('.jpg')]
            idx2fpathes[ch_idx] = sub_img_fpath_list
            pbar.update(1)
        pbar.close()
        return idx2fpathes

    def _load_hegui_data_info(self):
        register_dir = os.path.join(self.finetune_dir, 'register')
        test_dir = os.path.join(self.finetune_dir, 'test')
        register_img_fn_list = os.listdir(register_dir)[0: self.n_finetune_seletected]
        test_img_fn_list = os.listdir(test_dir)
        idx2fpathes = {}
        for idx, r_img_fn in enumerate(register_img_fn_list):
            t_img_fn = r_img_fn.replace('0000', '0001')
            if t_img_fn not in test_img_fn_list:
                print('{} not found.'.format(t_img_fn))
                exit(1)
            register = os.path.join(register_dir, r_img_fn)
            test = os.path.join(test_dir, t_img_fn)
            idx2fpathes[idx + self.n_base_selected] = [register, test]

        print('n_finetune_identities: {}'.format(len(idx2fpathes)))
        return idx2fpathes

    def _build_triplet_groups(self, idx2fpathes):
        triplet_groups = []
        for _ in range(self.n_triplets):
            if random.uniform(0.0, 1.0) < self.finetune_ratio:
                c1 = np.random.randint(self.n_base_selected, self.n_cls)
            else:
                c1 = np.random.randint(0, self.n_base_selected)
            c2 = np.random.randint(0, self.n_cls)
            while len(idx2fpathes[c1]) < 2:
                c1 = np.random.randint(0, self.n_cls - 1)
            while c1 == c2:
                c2 = np.random.randint(0, self.n_cls - 1)
            if len(idx2fpathes[c1]) == 2:  # hack to speed up process
                s1, s2 = 0, 1
            else:
                s1 = np.random.randint(0, len(idx2fpathes[c1]))
                s2 = np.random.randint(0, len(idx2fpathes[c1]))
                while s1 == s2:
                    s2 = np.random.randint(0, len(idx2fpathes[c1]))
            if len(idx2fpathes[c2]) == 1:
                s3 = 0
            else:
                s3 = np.random.randint(0, len(idx2fpathes[c2]))
            triplet_groups.append(
                [idx2fpathes[c1][s1], idx2fpathes[c1][s2], idx2fpathes[c2][s3], c1, c2])
        return triplet_groups

    def __getitem__(self, idx):
        anchor, positive, negtive, anc_cls_idx, neg_cls_idx = self.triplet_groups[idx]
        anc_img = self.preprocess_func(cv2.imread(anchor))
        pos_img = self.preprocess_func(cv2.imread(positive))
        neg_img = self.preprocess_func(cv2.imread(negtive))
        return anc_img, pos_img, neg_img, anc_cls_idx, neg_cls_idx



if __name__ == '__main__':

    dataset = TripletDataset()
    print(len(dataset))
    for sample in dataset:
        print(sample[3], sample[4])
