import sys,os
pj = os.path.join
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
sys.path.append(curPath)
sys.path.append(pj(curPath,'FaceRecog'))
sys.path.append(pj(curPath,'Pytorch_Retinaface'))

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
from FaceRecog.eval_utils.inference_api import Inference
from Pytorch_Retinaface.retina_infer import RetinaFaceDet

ct.set_core_number(4)
ct.set_core_version('MLU270')

"""
face bbox detection
"""

def fetch_cpu_data(x,use_half_input=False, to_numpy=True):
    if use_half_input:
        output = x.cpu().type(torch.FloatTensor)
    else:
        output = x.cpu()
    if to_numpy:
        return output.detach().numpy()
    else:
        return output


def resize(img_cv, dh = 540 , dw = 860 ):
    output_img = np.zeros((dh, dw, 3), np.uint8)
    dratio = dh / dw
    h, w = img_cv.shape[:2]
    ratio = h / w
    if ratio > dratio:
        # w should be filled
        _h = dh
        scale = _h / h
        _w = int(w * scale)
    else:
        # h should be filled
        _w = dw
        scale = _w / w
        _h = int(h * scale)

    scale_img_cv = cv2.resize(img_cv, dsize=(_w, _h))
    #filling the output_img with scale_img_cv where ratio of width and height is same as img_cv.
    output_img[:_h, :_w, :] = scale_img_cv[:, :, :]

    return {
        'output': output_img,
        'ratio': scale,#得到的检测框与原图上人脸框之间的比例
    }

def _normalize_retinaface(img_cv2,dst_size=[480,640],mlu=False):
    """
    :param img_cv2: 
    :param dst_size:[width,height] 
    :param mlu: 
    :return: 
    """
    dw,dh = dst_size
    out_data = resize(img_cv2,dh,dw)
    img_cv2 = out_data['output']
    ratio = out_data['ratio']
    img_data = np.asarray(img_cv2, dtype=np.float32)
    if mlu:
        return img_data #[0,255]
    else:
        mean = (104, 117, 123)
        img_data = img_data - mean
        img_data = img_data.astype(np.float32) #[0,1] normalized
    return img_data,ratio

def preprocess_retinaface(img_cv2, dst_size = [480,640] ,mlu=False):
    img_data, ratio = _normalize_retinaface(img_cv2,dst_size=dst_size,mlu=mlu)
    img_data = np.transpose(img_data, axes=[2, 0, 1])
    img_data = np.expand_dims(img_data, axis=0)
    img_t = torch.from_numpy(img_data)
    return img_t, ratio

class mlu_face_det_inference(object):
    def __init__(self ,weights,model_name='mobile0.25',use_mlu=True,use_jit=False):
        super(mlu_face_det_inference,self).__init__()

        self.use_mlu = use_mlu
        self.use_jit = use_jit
        loading = False if use_mlu else True
        infer = RetinaFaceDet(model_type=model_name,model_path=weights,use_cpu=True,loading=loading)
        model = infer.net
        if use_mlu:
            print('==using mlu quantization model==')
            model = mlu_quantize.quantize_dynamic_mlu(model)
            checkpoint = torch.load(weights, map_location='cpu')
            model.load_state_dict(checkpoint, strict=False)
            model.eval()
            model = model.to(ct.mlu_device())
            if use_jit:
                print('==jit==')
                randinput = torch.rand(1,3,112,112)*255
                randinput = randinput.to(ct.mlu_device())
                traced_model = torch.jit.trace(model, randinput, check_trace=False)
                self.model = traced_model
            else:
                self.model = model
        else:
            print('==using pytorch model==')
            model.eval()
            self.model = model

        self.infer = infer

    @torch.no_grad()
    def execute(self,img_cv2,dst_size=[480,640],threshold=0.8,topk=5000,keep_topk=750,nms_threshold=0.2):
        """
        :param dst_size: [width,height] all image will be scaled into that size for detection, but bbox will be returned in its original scale
        :param img_cv2: img_cv2 = cv2.imread() or [cv2.imread(c) for c in image_list] 
        :return: detss = list of np.array, [ np.array(n,15)]
                    where len(detss) = len(img_cv2) if isinstance(img_cv2,list) else 1
                          n = detected faces in each image
                          15 :[x0,y0,x1,y1,score,landmarkx0,landmarky0,...,]
        """
        if isinstance(img_cv2,list):
            data = [preprocess_retinaface(c, dst_size ,mlu=self.use_mlu) for c in img_cv2]
            ratio = [ c[1] for c in data ]
            data = [c[0] for c in data]
            data = torch.cat(data, dim=0)
        else:
            data = preprocess_retinaface(img_cv2,dst_size=dst_size,mlu=self.use_mlu)
            ratio = [data[1]]
            data = data[0]

        if self.use_mlu:
            data = data.to(ct.mlu_device())

        locs,confs,landmss = self.model(data)

        if self.use_mlu:
            locs = fetch_cpu_data(locs,use_half_input=False,to_numpy=False)
            confs = fetch_cpu_data(confs,use_half_input=False,to_numpy=False)
            landmss = fetch_cpu_data(landmss,use_half_input=False,to_numpy=False)

        net_output = [locs,confs,landmss]
        dets = self.infer.execute_batch_mlu(net_output=net_output, batch_shape=data.shape,
                                            threshold=threshold,topk=topk,keep_topk=keep_topk,
                                            nms_threshold=nms_threshold)
        scale = 1.3 #expanding bbox for better feature extraction

        assert len(dets) == len(ratio), 'Err len(dets) != len(ratio)'

        detss = []
        for n,det in enumerate(dets):
            det = det*ratio[n]
            detss.append(det)
        return detss

"""
face feature extraction
"""
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
    img_data = np.asarray(img_cv2, dtype=np.float32)
    if mlu:
        return img_data #[0,255]
    else:
        mean = 0.5
        std = 0.5
        img_data /= 255
        img_data = img_data - mean
        img_data = img_data / std
        img_data = img_data.astype(np.float32) #[0,1] normalized
    return img_data

def preprocess(img_cv2, mlu=False):
    img_format = _format(img_cv2)
    img_data = _normalize(img_format,mlu=mlu)
    img_data = np.transpose(img_data, axes=[2, 0, 1])
    img_data = np.expand_dims(img_data, axis=0)
    img_t = torch.from_numpy(img_data)
    return img_t


class mlu_face_rec_inference(object):
    def __init__(self ,weights,model_name='resnet101_irse_mx',use_mlu=True,use_jit=False):
        super(mlu_face_rec_inference,self).__init__()

        self.use_mlu = use_mlu
        self.use_jit = use_jit
        use_device = 'cpu'
        ckpt_fpath = None if use_mlu else weights
        infer = Inference(backbone_type=model_name,
                          ckpt_fpath=ckpt_fpath,
                          device=use_device)
        model = infer.model
        if use_mlu:
            print('==using mlu quantization model==')
            model = mlu_quantize.quantize_dynamic_mlu(model)
            checkpoint = torch.load(weights, map_location='cpu')
            model.load_state_dict(checkpoint, strict=False)
            model.eval()
            model = model.to(ct.mlu_device())
            if use_jit:
                print('==jit==')
                randinput = torch.rand(1,3,112,112)*255
                randinput = randinput.to(ct.mlu_device())
                traced_model = torch.jit.trace(model, randinput, check_trace=False)
                self.model = traced_model
            else:
                self.model = model
        else:
            print('==using pytorch model==')
            model.eval()
            self.model = model

    @torch.no_grad()
    def execute(self,img_cv2):
        """
        :param img_cv2: img_cv2 = cv2.imread() or [cv2.imread(c) for c in image_list] 
        :return: unnormalized feature [N,512], N = len(img_cv2) if isinstance(img_cv2,list) else 1
        """
        if isinstance(img_cv2,list):
            data = [preprocess(c, mlu=self.use_mlu) for c in img_cv2]
            data = torch.cat(data, dim=0)
        else:
            data = preprocess(img_cv2,mlu=self.use_mlu)

        if self.use_mlu:
            data = data.to(ct.mlu_device())

        out = self.model(data)
        out = out.cpu().detach().numpy().reshape(-1, 512)
        return out


if __name__ == "__main__":
    print('just a usage example')
    img_cv2 = cv2.imread('sally.jpg')
    cpu_face_det_model = mlu_face_det_inference(weights='weights/face_det/mobilenet0.25_Final.pth',use_mlu=False,use_jit=False)
    detss = cpu_face_det_model.execute(img_cv2)
    print(detss[0])

    exit(0)
    img_cv2 = cv2.imread('test.jpg')
    mlu_face_model = mlu_face_rec_inference(weights='weights/face_rec/resnet101_mlu_int8.pth',use_mlu=True,use_jit=True)
    cpu_face_model = mlu_face_rec_inference(weights='weights/face_rec/r101irse_model_3173.pth',use_mlu=False,use_jit=False)

    mlu_face_feature = mlu_face_model.execute(img_cv2)
    cpu_face_feature = cpu_face_model.execute(img_cv2)

    mlu_face_feature = normalize(mlu_face_feature,axis=1)
    cpu_face_feature = normalize(cpu_face_feature,axis=1)

    print('cosine similarity =', mlu_face_feature@(cpu_face_feature.transpose()))