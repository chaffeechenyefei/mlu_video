import sys,os
pj = os.path.join
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
sys.path.append(curPath)
sys.path.append(pj(curPath,'FaceRecog'))

import torch
import cv2
import os
import numpy as np
import time
import argparse
from mlu_inference import mlu_face_rec_inference

class EvalTupu(object):
    '''
    source_dir = '/data/FaceDetect/results/hegui_faces'
    register_dir = os.path.join(source_dir, 'register')
    test_dir = os.path.join(source_dir, 'test')
    '''

    def __init__(self,
                 data_dir='/data/FaceRecog/tupu_data/valid_labeled_faces',use_TTA = False,verbose=False):
        super(EvalTupu, self).__init__()
        self.data_dir = data_dir

        self.img_pairs = self._load_img_pairs()
        self.img_attack_groups = self._load_img_attack(self.img_pairs)
        self.use_TTA = use_TTA
        self.verbose = verbose

        # self.preprocess_func = Preprocess(use_TTA=use_TTA)
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

    @torch.no_grad()
    def _execute_inference(self, model, img_cv2):
        embedd = model.execute(img_cv2)
        return embedd

    def __call__(self, model):
        fpath2feat_dict = {}
        start_time = time.time()
        n_finish = 0

        img_cv2 = []
        print('loading images')
        for pair in self.img_pairs:
            img_cv2.append(cv2.imread(pair['register']))
            img2_cv2 = [cv2.imread(c) for c in pair['test_list']]
            img_cv2.extend(img2_cv2)

        print('extracting features')
        embedd = self._execute_inference(model,img_cv2)

        print('creating dictionary')
        pcnt = 0
        for pair in self.img_pairs:
            fpath2feat_dict[pair['register']] = embedd[pcnt,:].reshape(-1)
            pcnt += 1
            for test_fpath in pair['test_list']:
                fpath2feat_dict[test_fpath] = embedd[pcnt,:].reshape(-1)
                pcnt += 1

        # for pcnt,pair in enumerate(self.img_pairs):
        #     print('%d/%d=%0.3f'%(pcnt,len(self.img_pairs),pcnt/len(self.img_pairs)),end='\r')
        #     img1_cv2 = cv2.imread(pair['register'])
        #     embedd1 = self._execute_inference(model, img1_cv2)
        #     fpath2feat_dict[pair['register']] = embedd1
        #
        #     img2_cv2 = [ cv2.imread(c) for c in pair['test_list']]
        #     embedd2 = self._execute_inference(model,img2_cv2)
        #     for _pcnt,test_fpath in enumerate(pair['test_list']):
        #         fpath2feat_dict[test_fpath] = embedd2[_pcnt,:].reshape(1,512)
        #     # for test_fpath in pair['test_list']:
        #     #     img2_cv2 = cv2.imread(test_fpath)
        #     #     embedd2 = self._execute_inference(model, img2_cv2)
        #     #     fpath2feat_dict[test_fpath] = embedd2
        #     n_finish = n_finish + 1 + len(pair['test_list'])
        end_time = time.time()
        process_speed = (end_time - start_time) / (pcnt + 1e-5)
        print('process_speed: {} sec'.format(process_speed))

        pair_distance_list = self._pair_match_test(fpath2feat_dict)
        pair_distances = np.asarray(pair_distance_list, dtype=np.float32)
        attack_distance_list = self._attack_test(fpath2feat_dict)
        attack_distances = np.asarray(attack_distance_list, dtype=np.float32)
        thresh_vals_list = np.arange(start=0.0, stop=1.0, step=0.001)
        n_pair = pair_distances.shape[0]
        n_attack = attack_distances.shape[0]
        # print('thresh_vals: {}'.format(thresh_vals_list))
        t_FAR = 1e-3
        TPR_list = []
        FAR_list = []
        upper_case = dict(TPR=0, FAR=1, thresh=0.4)
        lower_case = dict(TPR=0, FAR=0, thresh=0.4)
        for thresh in thresh_vals_list:
            TPR = np.sum((pair_distances > thresh).astype(np.float32)) / n_pair
            FAR = np.sum((attack_distances > thresh).astype(np.float32)) / n_attack
            if FAR > t_FAR and FAR < upper_case['FAR']:
                upper_case['FAR'] = FAR
                upper_case['TPR'] = TPR
                upper_case['thresh'] = thresh
            if FAR < t_FAR and FAR > lower_case['FAR']:
                lower_case['FAR'] = FAR
                lower_case['TPR'] = TPR
                lower_case['thresh'] = thresh
            if self.verbose:
                print('thresh: {:.3f}, TPR: {}, FAR: {}'.format(thresh, TPR, FAR))
            TPR_list.append(TPR)
            FAR_list.append(FAR)
        target_TPR = (lower_case['TPR'] * (upper_case['FAR'] - t_FAR) + upper_case['TPR'] * (
                t_FAR - lower_case['FAR'])) \
                     / (upper_case['FAR'] - lower_case['FAR'])
        target_thresh = (lower_case['thresh'] * (upper_case['FAR'] - t_FAR) + upper_case['thresh'] * (
                t_FAR - lower_case['FAR'])) \
                        / (upper_case['FAR'] - lower_case['FAR'])
        print('### EVAL[tupu]: thresh={:.5f}, TPR={:.5f}, FAR={:.5f}'.format(target_thresh, target_TPR, t_FAR))
        #############################################################################################
        t_FAR = 1e-4
        TPR_list = []
        FAR_list = []
        upper_case = dict(TPR=0, FAR=1, thresh=0.4)
        lower_case = dict(TPR=0, FAR=0, thresh=0.4)
        for thresh in thresh_vals_list:
            TPR = np.sum((pair_distances > thresh).astype(np.float32)) / n_pair
            FAR = np.sum((attack_distances > thresh).astype(np.float32)) / n_attack
            if FAR > t_FAR and FAR < upper_case['FAR']:
                upper_case['FAR'] = FAR
                upper_case['TPR'] = TPR
                upper_case['thresh'] = thresh
            if FAR < t_FAR and FAR > lower_case['FAR']:
                lower_case['FAR'] = FAR
                lower_case['TPR'] = TPR
                lower_case['thresh'] = thresh
            # print('thresh: {:.3f}, TPR: {}, FAR: {}'.format(thresh, TPR, FAR))
            TPR_list.append(TPR)
            FAR_list.append(FAR)
        target_TPR_2 = (lower_case['TPR'] * (upper_case['FAR'] - t_FAR) + upper_case['TPR'] * (
                t_FAR - lower_case['FAR'])) \
                       / (upper_case['FAR'] - lower_case['FAR'])
        target_thresh = (lower_case['thresh'] * (upper_case['FAR'] - t_FAR) + upper_case['thresh'] * (
                t_FAR - lower_case['FAR'])) \
                        / (upper_case['FAR'] - lower_case['FAR'])
        print('### EVAL[tupu]: thresh={:.5f}, TPR={:.5f}, FAR={:.5f}'.format(target_thresh, target_TPR_2, t_FAR))

        return target_TPR


def main(data_dir, use_mlu = False ,use_TTA = False, **kwargs):
    backbone_type = 'resnet101_irse_mx'
    if use_mlu:
        ckpt_fpath = 'resnet101_mlu_int8.pth'
    else:
        ckpt_fpath = 'weights/face_rec/r101irse_model_3173.pth'

    if data_dir is None:
        data_dir = '/data/data/evaluation/valid_labeled_faces'

    infer_model = mlu_face_rec_inference(weights=ckpt_fpath,model_name=backbone_type,use_mlu=use_mlu)
    eval = EvalTupu(data_dir=data_dir, use_TTA=use_TTA)
    with torch.no_grad():
        eval(infer_model)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mlu',action='store_true')
    parser.add_argument('--data',default=None)
    args = parser.parse_args()

    main(data_dir=args.data,use_mlu=args.mlu)
    # search_best_model()
