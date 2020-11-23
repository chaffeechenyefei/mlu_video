import sys,os
pj = os.path.join
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
sys.path.append(curPath)
sys.path.append(pj(curPath,'FaceRecog'))
# sys.path.append(pj(curPath,'Pytorch_Retinaface'))

import torch
import torch.nn as nn
from torchvision.models.quantization.utils import quantize_model
import torch_mlu
import torch_mlu.core.mlu_model as ct
import torch_mlu.core.mlu_quantize as mlu_quantize
import argparse
import cv2
import numpy as np
from sklearn.preprocessing import normalize
from eval_utils.inference_api import Inference

def _format(img_cv2, format_size=112):
    org_h, org_w = img_cv2.shape[0:2]
    rescale_ratio = format_size / max(org_h, org_w)
    h, w = int(org_h * rescale_ratio), int(org_w * rescale_ratio)
    img_rescaled = cv2.resize(img_cv2, (w, h))
    paste_pos = [int((format_size - w) / 2), int((format_size - h) / 2)]
    img_format = np.zeros((format_size, format_size, 3), dtype=np.uint8)
    img_format[paste_pos[1]:paste_pos[1] + h, paste_pos[0]:paste_pos[0] + w] = img_rescaled
    return img_format


def _normalize(img_cv2,mlu=False):
    img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    # mean = [123.675, 116.28, 103.53]
    # std = [58.395, 57.12, 57.375]
    if mlu:
        mean = 0
        std = 1.0
    else:
        mean = 127.5
        std = 127.5
    img_data = np.asarray(img_cv2, dtype=np.float32)
    img_data = img_data - mean
    img_data = img_data / std
    img_data = img_data.astype(np.float32)
    return img_data

def preprocess(img_cv2, mlu=False):
    img_format = _format(img_cv2)
    img_data = _normalize(img_format,mlu=mlu)
    img_data = np.transpose(img_data, axes=[2, 0, 1])
    img_data = np.expand_dims(img_data, axis=0)
    img_t = torch.from_numpy(img_data)
    return img_t


class mlu_face_rec_inference(object):
    def __init__(self ,weights,model_name='resnet101_irse_mx',use_mlu=True):
        super(mlu_face_rec_inference,self).__init__()
        use_device = 'cpu'
        ckpt_fpath = None if use_mlu else weights
        infer = Inference(backbone_type=model_name,
                          ckpt_fpath=ckpt_fpath,
                          device=use_device)
        model = infer.model
        if use_mlu:
            model = mlu_quantize.quantize_dynamic_mlu(model)
            checkpoint = torch.load(weights, map_location='cpu')
            model.load_state_dict(checkpoint, strict=False)
            model.eval().float()
            self.model = model.to(ct.mlu_device())
        else:
            model.eval()
            self.model = model

    @torch.no_grad()
    def execute(self,img_cv2):
        data = preprocess(img_cv2)
        out = self.model(data)
        out = out.detach().numpy().reshape(-1, 512)
        return out


