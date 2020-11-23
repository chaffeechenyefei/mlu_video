import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import os
import pickle
import numpy as np
from model_design_v2 import resnet
from eval_utils.inference_api import Inference
from mtcnn_pytorch.mtcnn_inference import MTCNNInference



class Preprocess(object):

    def __init__(self, format_size=112):
        super(Preprocess, self).__init__()
        self.format_size = format_size

    def _format(self, img_cv2):
        org_h, org_w = img_cv2.shape[0:2]
        rescale_ratio = self.format_size / max(org_h, org_w)
        h, w = int(org_h * rescale_ratio), int(org_w * rescale_ratio)
        img_rescaled = cv2.resize(img_cv2, (w, h))
        paste_pos = [int((self.format_size - w) / 2), int((self.format_size - h) / 2)]
        img_format = np.zeros((self.format_size, self.format_size, 3), dtype=np.uint8)
        img_format[paste_pos[1]:paste_pos[1] + h, paste_pos[0]:paste_pos[0] + w] = img_rescaled
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



class CompareFace(object):
    def __init__(self):
        super(CompareFace, self).__init__()
        self.mtcnn_infer = MTCNNInference()
        self.preprocess_func = Preprocess()


    def _cosine_distance(self, x1, x2):
        return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum(np.power(x1 - x2, 2)))

    def _load_img_pair(self, img_dir):
        img_fn_list = [img_fn
                       for img_fn in os.listdir(img_dir)
                       if img_fn.endswith(('.jpg', '.jpeg', '.png', 'PNG'))
                       and os.path.isfile(os.path.join(img_dir, img_fn))]
        register_img_fn = None
        test_img_fn_list = []
        for img_fn in img_fn_list:
            if '0_register' in img_fn:
                register_img_fn = img_fn
            else:
                test_img_fn_list.append(img_fn)
        if register_img_fn is None:
            print('### ERROR: register img not found !')
            exit(-1)
        return dict(
            register=os.path.join(img_dir, register_img_fn),
            test_list=[os.path.join(img_dir, img_fn) for img_fn in test_img_fn_list]
        )

    def _execute_inference(self, model, img_cv2):
        img_t = self.preprocess_func(img_cv2)
        embedd = model(img_t)
        # embedd = F.normalize(embedd)
        embedd = embedd.data.cpu().numpy()[0]
        return embedd

    def _visual(self, r_img_cv2, t_img_cv2, score, thresh=0.4):
        r_img_cv2 = cv2.resize(r_img_cv2, (512, 512))
        t_img_cv2 = cv2.resize(t_img_cv2, (512, 512))
        comp_img = cv2.hconcat(src=[r_img_cv2, t_img_cv2])
        cv2.putText(comp_img,
                    '{:.5f}'.format(score),
                    (300, 100),
                    cv2.FONT_HERSHEY_PLAIN,
                    fontScale=6.0,
                    color=(0, 255, 0) if score > thresh else (0, 0, 255),
                    thickness=5
                    )
        return comp_img

    def __call__(self,
                 model,
                 r_img_fpath,
                 t_img_fpath,
                 save_dir='/data/FaceRecog/results/compare_results'):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        r_img_cv2 = cv2.imread(r_img_fpath)
        r_face_bbox = self.mtcnn_infer(r_img_cv2)
        r_face_img = r_img_cv2[r_face_bbox[1]:r_face_bbox[3], r_face_bbox[0], r_face_bbox[2]]
        r_feature = self._execute_inference(model, r_face_img)

        t_img_cv2 = cv2.imread(t_img_fpath)
        t_face_bbox = self.mtcnn_infer(t_img_cv2)
        t_face_img = t_img_cv2[t_face_bbox[1]:t_face_bbox[3], t_face_bbox[0], t_face_bbox[2]]
        t_feature = self._execute_inference(model, t_face_img)

        score = self._cosine_distance(r_feature, t_feature)
        comp_img = self._visual(r_img_cv2, t_img_cv2, score)
        save_fpath = os.path.join(save_dir,
                                  '{}_{}.jpg'.format(
                                      r_img_fpath.split('/')[-1].replace('.jpg', ''),
                                      t_img_fpath.split('/')[-1].replace('.jpg', ''))
                                  )
        cv2.imwrite(save_fpath, comp_img)





def main():
    backbone_type = 'resnet50_irse_mx'
    ckpt_fpath = '/data/output/insight_face_res50irsemx_cosface_emore_dist/best_model4tupu/1591242318_tupu_0.981.pth'
    infer = Inference(backbone_type=backbone_type, ckpt_fpath=ckpt_fpath)
    model = infer.model
    test_obj = CompareFace()
    r_img_fpath = ''
    t_img_fpath = ''
    test_obj(model, r_img_fpath, t_img_fpath)


if __name__ == '__main__':
    main()

