import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import os
import pickle
import numpy as np
from tqdm import tqdm
from model_design_v2 import resnet
from eval_utils.inference_api import Inference
from eval_utils import statistic
from datasets.glintasia_v2 import EmoreDataset as Dataset

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'


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
        img_data = np.expand_dims(img_data, axis=0)
        img_t = torch.from_numpy(img_data)
        img_t = img_t.cuda()
        return img_t


class EvalGlintaisa(object):

    def __init__(self,
                 val_dataset):
        super(EvalGlintaisa, self).__init__()
        self.img_dir = val_dataset.img_dir
        self.idx2dirname = val_dataset.idx2dirname
        self.img_pairs = self._load_img_pairs()
        self.preprocess_func = Preprocess()


    def _cosine_distance(self, x1, x2):
        return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

    def _load_img_pairs(self):
        # print('n_chdir: {}'.format(len(self.idx2dirname)))
        img_pairs = []
        for ch_dir in self.idx2dirname.values():
            ch_dir = os.path.join(self.img_dir, ch_dir)
            img_fn_list = os.listdir(ch_dir)
            # todo: random get one as register
            img_pairs.append(dict(
                register=os.path.join(ch_dir, img_fn_list[0]),
                test_list=[os.path.join(ch_dir, img_fn) for img_fn in img_fn_list[1:]]
            ))
        return img_pairs

    def _pair_match_test(self, fpath2feat_dict):
        pair_distance_list = []

        for idx, pair in enumerate(self.img_pairs):
            r_img_fpath = pair['register']
            t_img_fpath_list = pair['test_list']
            for t_img_fpath in t_img_fpath_list:
                r_feature = fpath2feat_dict[r_img_fpath]  # self.infer.execute(img_cv2=cv2.imread(r_img_fpath))
                t_feature = fpath2feat_dict[t_img_fpath]  # self.infer.execute(img_cv2=cv2.imread(t_img_fpath))
                distance = self._cosine_distance(r_feature, t_feature)
                # print(idx, distance, r_img_fpath)
                pair_distance_list.append(distance)
        return pair_distance_list

    def _execute_inference(self, model, img_cv2):
        img_t = self.preprocess_func(img_cv2)
        embedd = model(img_t)
        # embedd = F.normalize(embedd)
        embedd = embedd.data.cpu().numpy()[0]
        return embedd

    def __call__(self, model, thresh=0.4):
        fpath2feat_dict = {}
        pbar = tqdm(total=len(self.img_pairs), desc='eval_glintasia')
        for pair in self.img_pairs:
            img1_cv2 = cv2.imread(pair['register'])
            embedd1 = self._execute_inference(model, img1_cv2)
            fpath2feat_dict[pair['register']] = embedd1
            for test_fpath in pair['test_list']:
                img2_cv2 = cv2.imread(test_fpath)
                embedd2 = self._execute_inference(model, img2_cv2)
                fpath2feat_dict[test_fpath] = embedd2
            pbar.update(1)
        pbar.close()

        pair_distance_list = self._pair_match_test(fpath2feat_dict)
        pair_distances = np.asarray(pair_distance_list, dtype=np.float32)
        acc = np.sum(pair_distances > thresh) / float(pair_distances.shape[0])
        print('### EVAL[glintasia]: {}'.format(acc))
        return acc

def main():

    backbone_type = 'resnet50_irse_mx'
    ckpt_fpath = '/data/output/insight_face_res50irsemx_cosface_emore_dist'
    # ckpt_fpath = '/data/output/insight_face_res50irsemx_cosface_glintasia_dist'
    # ckpt_fpath = '/data/output/insight_face_res50irsemx_cosface_glintasia_full/model_1755.pth'
    # ckpt_fpath = '/data/output/insight_face_res50irsemx_cosface_glintasia_full/model_1804.pth'

    infer = Inference(backbone_type=backbone_type, ckpt_fpath=ckpt_fpath)
    model = infer.model

    val_dataset = Dataset(data_dir='/data/data/glintasia',
                          shuffle=False,
                          eval=True)
    eval = EvalGlintaisa(val_dataset=val_dataset)
    with torch.no_grad():
        eval(model)



if __name__ == '__main__':
    main()
    # search_best_model()