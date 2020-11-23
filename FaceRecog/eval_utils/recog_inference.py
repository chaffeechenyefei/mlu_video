import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import os
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import pickle

from model_design_v2 import resnet
from eval_utils.inference_api import Inference
from eval_utils import statistic

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




class InferenceVisual(object):

    def __init__(self, model, register_dir, save_dir, thresh=0.4):
        super(InferenceVisual, self).__init__()
        self.model = model
        self.register_dir = register_dir
        self.save_dir = save_dir
        self.thresh = thresh

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        self.preprocess_func = Preprocess()
        self.register_features = self._load_register_features()

    def _load_register_features(self, features_cache='/data/FaceRecog/tupu_data/ucloud_staff_features_update0602'):
        if os.path.exists(features_cache):
            print('load features from pickle file !')
            with open(features_cache, 'rb') as f:
                register_features = pickle.load(f)
        else:
            register_features = {}
            img_dir = self.register_dir
            img_fn_list = [img_fn
                           for img_fn in os.listdir(img_dir)
                           if img_fn.endswith(('.jpg', '.jpeg', '.png', 'PNG'))
                           and os.path.isfile(os.path.join(img_dir, img_fn))]
            for img_fn in tqdm(img_fn_list, desc='load_registers'):
                face_id = img_fn.split('.')[0]
                img_fpath = os.path.join(img_dir, img_fn)
                feature = self._execute_inference(model=self.model, img_cv2=cv2.imread(img_fpath))
                register_features[face_id] = dict(feature=feature, img_fpath=img_fpath)
            with open(features_cache, 'wb') as f:
                pickle.dump(register_features, f)
        return register_features

    def _cosine_distance(self, x1, x2):
        return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

    def _execute_inference(self, model, img_cv2):
        img_t = self.preprocess_func(img_cv2)
        embedd = model(img_t)
        # embedd = F.normalize(embedd)
        embedd = embedd.data.cpu().numpy()[0]
        return embedd

    def _add_ch_text(self, img_cv2, text, font_scale=40, left_top_point=(100, 100), rgb_color=(0, 0, 255)):
        img_pil = Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        fontStyle = ImageFont.truetype(
            "/data/FaceRecog/eval_utils/font/simfang.ttf",
            font_scale,
            encoding="utf-8"
        )
        draw.text(left_top_point,
                  text,
                  rgb_color,
                  font=fontStyle)
        img_cv2 = cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGB2BGR)
        print(img_cv2.shape)
        return img_cv2

    def _visual(self, r_img_cv2, t_img_cv2, face_id, score, thresh=0.4):
        r_img_cv2 = cv2.resize(r_img_cv2, (512, 512))
        t_img_cv2 = cv2.resize(t_img_cv2, (512, 512))
        comp_img = cv2.hconcat(src=[r_img_cv2, t_img_cv2])
        comp_img = self._add_ch_text(img_cv2=comp_img,
                                     text=face_id,
                                     font_scale=40,
                                     left_top_point=(10, 10),
                                     rgb_color=(0, 0, 255))
        # cv2.imwrite('1.jpg', comp_img)
        cv2.putText(comp_img,
                    '{:.5f}'.format(score),
                    (300, 150),
                    cv2.FONT_HERSHEY_PLAIN,
                    fontScale=5.0,
                    color=(0, 255, 0) if score > thresh else (0, 0, 255),
                    thickness=5
                    )
        print(comp_img.shape)
        # cv2.imwrite('2.jpg', comp_img)
        return comp_img

    def __call__(self, img_dir):
        img_fn_list = [img_fn
                       for img_fn in os.listdir(img_dir)
                       if img_fn.endswith(('.jpg', '.jpeg', '.png', 'PNG'))
                       and os.path.isfile(os.path.join(img_dir, img_fn))]
        for img_fn in img_fn_list:
            img_fpath = os.path.join(img_dir, img_fn)
            img_cv2 = cv2.imread(img_fpath)
            h, w = img_cv2.shape[0:2]
            cx, cy = int(w/2.0), int(h/2.0)
            shrink = 0.3
            x0, y0 = int(h*shrink/2.0), int(w*shrink/2.0)
            new_h = int(h*(1-shrink))
            new_w = int(w*(1-shrink))
            img_cv2 = img_cv2[y0:y0+new_h, x0:x0+new_w]

            t_feature = self._execute_inference(model=self.model, img_cv2=img_cv2)
            match_count = 0
            for face_id, face_info in self.register_features.items():
                r_feature = face_info['feature']
                score = self._cosine_distance(t_feature, r_feature)
                if score >= self.thresh:
                    print('Match:[{}], score:[{:.5f}]'.format(face_id, score))
                    r_img_cv2 = cv2.imread(face_info['img_fpath'])
                    compare_img = self._visual(r_img_cv2, img_cv2, face_id, score, thresh=self.thresh)
                    save_fpath = os.path.join(self.save_dir, img_fn.replace('.jpg', '_{}.jpg'.format(match_count)))
                    cv2.imwrite(save_fpath, compare_img)
                    match_count += 1



def main():


    backbone_type = 'resnet101_irse_mx'
    ckpt_fpath = '/data/output/insight_face_res101irsemx_cosface_emore_dist'
    infer = Inference(backbone_type=backbone_type, ckpt_fpath=ckpt_fpath)
    model = infer.model

    register_dir = '/data/FaceRecog/tupu_data/ucloud_staff_faces_update0602'
    save_dir = '/data/FaceRecog/results/image_cache_0602_mtcnn_crop'
    test_obj = InferenceVisual(model=model,
                               register_dir=register_dir,
                               save_dir=save_dir)
    test_img_dir = '/data/FaceRecog/tupu_data/image_cache_0602_mtcnn_crop'
    test_obj(test_img_dir)


if __name__ == '__main__':
    main()
