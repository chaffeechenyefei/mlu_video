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

#os.environ["CUDA_VISIBLE_DEVICES"] = '3'


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




class EvalTupu(object):
    '''
    source_dir = '/data/FaceDetect/results/hegui_faces'
    register_dir = os.path.join(source_dir, 'register')
    test_dir = os.path.join(source_dir, 'test')
    '''
    def __init__(self,
                 data_dir='/data/FaceRecog/tupu_data/valid_labeled_faces'):
        super(EvalTupu, self).__init__()
        self.data_dir = data_dir

        self.img_pairs = self._load_img_pairs()
        self.img_attack_groups = self._load_img_attack(self.img_pairs)

        self.preprocess_func = Preprocess()
        # self.fpath2feat_dict = {}


    def _cosine_distance(self, x1, x2):
        return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

    def _euclidean_metric(self, x1, x2):
        return np.sqrt(np.sum(np.power(x1 - x2, 2)))

    def _load_img_pairs(self):
        ch_dir_list = [ch_dir
                       for ch_dir in os.listdir(self.data_dir)
                       if os.path.isdir(os.path.join(self.data_dir, ch_dir))]
        selected_dir_list = ch_dir_list
        # print(selected_dir_list)
        # print('n_chdir: {}'.format(len(selected_dir_list)))
        img_pairs = []
        for ch_dir in selected_dir_list:
            ch_dir = os.path.join(self.data_dir, ch_dir)
            img_fn_list = os.listdir(ch_dir)
            img_pairs.append(dict(
                register=os.path.join(ch_dir, img_fn_list[0]),
                test_list=[os.path.join(ch_dir, img_fn) for img_fn in img_fn_list[1:]]
            ))
        return img_pairs

    def _load_img_attack(self, img_pairs):
        img_attack_groups = []
        for img_pair in img_pairs:
            r_img_fpath = img_pair['register']
            attack_list = []
            for attack_pair in img_pairs:
                if attack_pair['register'] == r_img_fpath:
                    continue
                else:
                    attack_list.extend(attack_pair['test_list'])
            img_attack_groups.append(dict(
                register=r_img_fpath,
                attack_list=attack_list,
            ))
        # print('len_attack_dict: {}'.format(len(img_attack_groups)))
        return img_attack_groups

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

    def _attack_test(self, fpath2feat_dict):
        attack_distance_list = []
        for person_id, attack_group in enumerate(self.img_attack_groups):
            register_fpath = attack_group['register']
            attack_fpath_list = attack_group['attack_list']
            r_feature = fpath2feat_dict[register_fpath]
            for test_id, attack_fpath in enumerate(attack_fpath_list):
                attack_feature = fpath2feat_dict[attack_fpath]
                distance = self._cosine_distance(r_feature, attack_feature)
                # print('{}-{}'.format(person_id, test_id), distance)
                attack_distance_list.append(distance)
        return attack_distance_list

    def _execute_inference(self, model, img_cv2):
        img_t = self.preprocess_func(img_cv2)
        embedd = model(img_t)
        # embedd = F.normalize(embedd)
        embedd = embedd.data.cpu().numpy()[0]
        return embedd

    def __call__(self, model):
        fpath2feat_dict = {}
        for pair in self.img_pairs:
            img1_cv2 = cv2.imread(pair['register'])
            embedd1 = self._execute_inference(model, img1_cv2)
            fpath2feat_dict[pair['register']] = embedd1
            for test_fpath in pair['test_list']:
                img2_cv2 = cv2.imread(test_fpath)
                embedd2 = self._execute_inference(model, img2_cv2)
                fpath2feat_dict[test_fpath] = embedd2

        pair_distance_list = self._pair_match_test(fpath2feat_dict)
        pair_distances = np.asarray(pair_distance_list, dtype=np.float32)
        attack_distance_list = self._attack_test(fpath2feat_dict)
        attack_distances = np.asarray(attack_distance_list, dtype=np.float32)
        thresh_vals_list = np.arange(start=0.0, stop=1.0, step=0.001)
        n_pair = pair_distances.shape[0]
        n_attack = attack_distances.shape[0]
        # print('thresh_vals: {}'.format(thresh_vals_list))
        TPR_list = []
        FAR_list = []
        upper_case = dict(TPR=0, FAR=1)
        lower_case = dict(TPR=0, FAR=0)
        for thresh in thresh_vals_list:
            TPR = np.sum((pair_distances > thresh).astype(np.float32)) / n_pair
            FAR = np.sum((attack_distances > thresh).astype(np.float32)) / n_attack
            if FAR > 1e-3 and FAR < upper_case['FAR']:
                upper_case['FAR'] = FAR
                upper_case['TPR'] = TPR
            if FAR < 1e-3 and FAR > lower_case['FAR']:
                lower_case['FAR'] = FAR
                lower_case['TPR'] = TPR
            # print('thresh: {:.3f}, TPR: {}, FAR: {}'.format(thresh, TPR, FAR))
            TPR_list.append(TPR)
            FAR_list.append(FAR)
        target_TPR = (lower_case['TPR'] * (upper_case['FAR'] - 1e-3) + upper_case['TPR'] * (1e-3 - lower_case['FAR'])) \
                     / (upper_case['FAR'] - lower_case['FAR'])
        print('### EVAL[tupu]: {}'.format(target_TPR))
        return target_TPR

def main():

    backbone_type = 'resnet50_irse_mx'
    # ckpt_fpath = '/data/output/insight_face_res50irsemx_cosface_emore_dist'
    # ckpt_fpath = '/data/output/insight_face_res50irsemx_cosface_glintasia_dist'
    ckpt_fpath = '/data/output/insight_face_res50irsemx_cosface_glintasia_full/model_1755.pth'
    # ckpt_fpath = '/data/output/insight_face_res50irsemx_cosface_glintasia_full/model_1804.pth'

    data_dir = '/data/FaceRecog/tupu_data/valid_labeled_faces'
    # data_dir = '/data/FaceRecog/tupu_data/valid_ucloud_eval_faces_update20200528'
    infer = Inference(backbone_type=backbone_type, ckpt_fpath=ckpt_fpath)
    model = infer.model
    eval = EvalTupu(data_dir=data_dir)
    with torch.no_grad():
        eval(model)


def search_best_model():
    backbone_type = 'resnet50_irse_v2'
    # ckpt_dir = '/data/output/insight_face_res50irsev2_emore_dist'
    ckpt_dir = '/data/output/insight_face_res50irsev2_emore_dist/best_model4tupu'
    data_dir = '/data/FaceRecog/tupu_data/valid_labeled_faces'

    test_obj = EvalTupu(data_dir=data_dir)

    ckpt_fpath_list = [
        os.path.join(ckpt_dir, ckpt_fn)
        for ckpt_fn in os.listdir(ckpt_dir)
        if os.path.isfile(os.path.join(ckpt_dir, ckpt_fn)) and ckpt_fn.endswith('.pth')
    ]
    for ckpt_fpath in tqdm(ckpt_fpath_list):
        infer = Inference(backbone_type=backbone_type, ckpt_fpath=ckpt_fpath)
        model = infer.model
        test_obj(model)

if __name__ == '__main__':
    main()
    # search_best_model()