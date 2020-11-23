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

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import time


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




class EvalTriplet(object):

    def __init__(self,
                 data_dir,
                 visual_save_dir=None):
        super(EvalTriplet, self).__init__()
        self.data_dir = data_dir

        self.img_groups = self._load_img_groups()
        # self.img_attack_groups = self._load_img_attack(self.img_pairs)

        self.preprocess_func = Preprocess()
        # self.fpath2feat_dict = {}

        self.visual_save_dir = visual_save_dir
        if visual_save_dir is not None:
            if not os.path.exists(visual_save_dir):
                os.mkdir(visual_save_dir)

    def _cosine_distance(self, x1, x2):
        return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

    def _load_img_groups(self):
        ch_dir_list = [ch_dir
                       for ch_dir in os.listdir(self.data_dir)
                       if os.path.isdir(os.path.join(self.data_dir, ch_dir))]
        img_groups = []
        for ch_dir in ch_dir_list:
            ch_dir = os.path.join(self.data_dir, ch_dir)
            img_fn_list = os.listdir(ch_dir)
            r_img = None
            for img_fn in img_fn_list:
                if img_fn.startswith('0_register'):
                    r_img = img_fn
                    break
            if r_img is None:
                print('### Error! register not found, ch_dir: {}'.format(ch_dir))
                exit(1)
            register = os.path.join(ch_dir, r_img)
            match_list = [
                os.path.join(ch_dir, 'True', img_fn)
                for img_fn in os.listdir(os.path.join(ch_dir, 'True'))
                if os.path.isfile( os.path.join(ch_dir, 'True', img_fn)) and img_fn.endswith('jpg')
            ]
            attack_list = [
                os.path.join(ch_dir, 'False', img_fn)
                for img_fn in os.listdir(os.path.join(ch_dir, 'False'))
                if os.path.isfile(os.path.join(ch_dir, 'False', img_fn)) and img_fn.endswith('jpg')
            ]
            if len(match_list) > 0 or len(attack_list) > 0:
                img_groups.append(dict(
                    register=register,
                    match_list=match_list,
                    attack_list=attack_list
                ))
        return img_groups

    def _execute_inference(self, model, img_cv2):
        img_t = self.preprocess_func(img_cv2)
        embedd = model(img_t)
        # embedd = F.normalize(embedd)
        embedd = embedd.data.cpu().numpy()[0]
        return embedd

    def _visual(self, r_img_cv2, t_img_cv2, score, thresh=0.4, comp_type='unkown'):
        r_img_cv2 = cv2.resize(r_img_cv2, (512, 512))
        t_img_cv2 = cv2.resize(t_img_cv2, (512, 512))
        comp_img = cv2.hconcat(src=[r_img_cv2, t_img_cv2])
        cv2.putText(comp_img,
                    '{} | {:.5f}'.format(comp_type, score),
                    (200, 100),
                    cv2.FONT_HERSHEY_PLAIN,
                    fontScale=6.0,
                    color=(0, 255, 0) if score > thresh else (0, 0, 255),
                    thickness=5
                    )
        return comp_img

    def __call__(self, model):
        match_scores = []
        attack_scores = []
        for group in tqdm(self.img_groups, desc='testing'):
            r_img_cv2 = cv2.imread(group['register'])
            r_feature = self._execute_inference(model, r_img_cv2)
            for test_fpath in group['match_list']:
                t_img_cv2 = cv2.imread(test_fpath)
                t_feature = self._execute_inference(model, t_img_cv2)
                score = self._cosine_distance(r_feature, t_feature)
                match_scores.append(score)
                comp_img = self._visual(r_img_cv2=r_img_cv2,
                                        t_img_cv2=t_img_cv2,
                                        score=score,
                                        comp_type='match')
                cv2.imwrite(
                    os.path.join(self.visual_save_dir, '{}_{}.jpg'.format(int(time.time()*1e3), 'match')),
                    comp_img
                )

            for test_fpath in group['attack_list']:
                t_img_cv2 = cv2.imread(test_fpath)
                t_feature = self._execute_inference(model, t_img_cv2)
                score = self._cosine_distance(r_feature, t_feature)
                attack_scores.append(score)
                comp_img = self._visual(r_img_cv2=r_img_cv2,
                                        t_img_cv2=t_img_cv2,
                                        score=score,
                                        comp_type='attack')
                cv2.imwrite(
                    os.path.join(self.visual_save_dir, '{}_{}.jpg'.format(int(time.time() * 1e3), 'attack')),
                    comp_img
                )

        match_scores = np.asarray(match_scores, dtype=np.float32)
        attack_scores = np.asarray(attack_scores, dtype=np.float32)
        thresh_vals_list = np.arange(start=0.0, stop=1.0, step=0.001)
        n_pair = match_scores.shape[0]
        n_attack = attack_scores.shape[0]
        # print('thresh_vals: {}'.format(thresh_vals_list))
        TPR_list = []
        FAR_list = []
        upper_case = dict(TPR=0, FAR=1)
        lower_case = dict(TPR=0, FAR=0)
        for thresh in thresh_vals_list:
            TPR = np.sum((match_scores > thresh).astype(np.float32)) / n_pair
            FAR = np.sum((attack_scores > thresh).astype(np.float32)) / n_attack
            if FAR > 1e-3 and FAR < upper_case['FAR']:
                upper_case['FAR'] = FAR
                upper_case['TPR'] = TPR
            if FAR < 1e-3 and FAR > lower_case['FAR']:
                lower_case['FAR'] = FAR
                lower_case['TPR'] = TPR
            print('thresh: {:.3f}, TPR: {}, FAR: {}'.format(thresh, TPR, FAR))
            TPR_list.append(TPR)
            FAR_list.append(FAR)
        target_TPR = (lower_case['TPR'] * (upper_case['FAR'] - 1e-3) + upper_case['TPR'] * (1e-3 - lower_case['FAR'])) \
                     / (upper_case['FAR'] - lower_case['FAR'])
        print('### EVAL[ucloud]: {}'.format(target_TPR))
        return target_TPR

    def calc_balanced_acc(self, model):
        match_scores = []
        attack_scores = []
        for group in tqdm(self.img_groups, desc='testing'):
            r_img_cv2 = cv2.imread(group['register'])
            r_feature = self._execute_inference(model, r_img_cv2)
            for test_fpath in group['match_list']:
                t_img_cv2 = cv2.imread(test_fpath)
                t_feature = self._execute_inference(model, t_img_cv2)
                score = self._cosine_distance(r_feature, t_feature)
                match_scores.append(score)
                comp_img = self._visual(r_img_cv2=r_img_cv2,
                                        t_img_cv2=t_img_cv2,
                                        score=score,
                                        comp_type='match')
                cv2.imwrite(
                    os.path.join(self.visual_save_dir, '{}_{}.jpg'.format(int(time.time()*1e3), 'match')),
                    comp_img
                )

            for test_fpath in group['attack_list']:
                t_img_cv2 = cv2.imread(test_fpath)
                t_feature = self._execute_inference(model, t_img_cv2)
                score = self._cosine_distance(r_feature, t_feature)
                attack_scores.append(score)
                comp_img = self._visual(r_img_cv2=r_img_cv2,
                                        t_img_cv2=t_img_cv2,
                                        score=score,
                                        comp_type='attack')
                cv2.imwrite(
                    os.path.join(self.visual_save_dir, '{}_{}.jpg'.format(int(time.time() * 1e3), 'attack')),
                    comp_img
                )

        match_scores = np.asarray(match_scores, dtype=np.float32)
        attack_scores = np.asarray(attack_scores, dtype=np.float32)
        thresh_vals_list = np.arange(start=0.0, stop=1.0, step=0.001)
        n_pair = match_scores.shape[0]
        n_attack = attack_scores.shape[0]
        # print('thresh_vals: {}'.format(thresh_vals_list))
        TPR_list = []
        FAR_list = []
        upper_case = dict(TPR=0, FAR=1)
        lower_case = dict(TPR=0, FAR=0)
        for thresh in thresh_vals_list:
            TPR_acc = np.sum((match_scores > thresh).astype(np.float32)) / n_pair
            FAR_acc = np.sum((attack_scores < thresh).astype(np.float32)) / n_attack
            print('thresh: {:.3f}, TPR_acc: {:.5f}, FAR_acc: {:.5f}'.format(thresh, TPR_acc, FAR_acc))

def main():

    backbone_type = 'resnet50_irse_mx'
    ckpt_fpath = '/data/output/insight_face_res50irsemx_cosface_emore_dist'
    # ckpt_fpath = '/data/output/insight_face_res50irsemx_cosface_glintasia_dist'
    # ckpt_fpath = '/data/output/insight_face_res50irsemx_cosface_glintasia_full/model_1755.pth'
    # ckpt_fpath = '/data/output/insight_face_res50irsemx_cosface_glintasia_full/model_1804.pth'

    infer = Inference(backbone_type=backbone_type, ckpt_fpath=ckpt_fpath)
    model = infer.model
    test_obj = EvalTriplet(data_dir='/data/FaceRecog/tupu_data/zhongsheng_labeled_update0603',
                           visual_save_dir='/data/FaceRecog/results/zhongsheng_labeled_test_results')
    # test_obj(model)
    test_obj.calc_balanced_acc(model)

if __name__ == '__main__':
    main()
    # search_best_model()