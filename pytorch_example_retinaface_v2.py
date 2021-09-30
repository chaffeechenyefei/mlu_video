import sys,os
pj = os.path.join
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
sys.path.append(curPath)
# sys.path.append(pj(curPath,'FaceRecog'))
sys.path.append(pj(curPath,'Pytorch_Retinaface'))

import torch
import torch_mlu
import torch_mlu.core.mlu_model as ct
import torch_mlu.core.mlu_quantize as mlu_quantize
import argparse
import cv2
import numpy as np
import random,math
from sklearn.preprocessing import normalize
from cfg import model_dict
# from FaceRecog.eval_utils.inference_api import Inference
from Pytorch_Retinaface.retina_infer import RetinaFaceDetModule
# from Pytorch_Retinaface.retina_infer import RetinaFaceDetModuleNMS
from Pytorch_Retinaface.retina_infer import execute_batch_mlu


def _format(img_cv2, format_size=112):
    org_h, org_w = img_cv2.shape[0:2]
    rescale_ratio = format_size / max(org_h, org_w)
    h, w = int(org_h * rescale_ratio), int(org_w * rescale_ratio)
    img_rescaled = cv2.resize(img_cv2, (w, h))
    paste_pos = [int((format_size - w) / 2), int((format_size - h) / 2)]
    img_format = np.zeros((format_size, format_size, 3), dtype=np.uint8)
    img_format[paste_pos[1]:paste_pos[1] + h, paste_pos[0]:paste_pos[0] + w] = img_rescaled
    return img_format


def resize(img_cv2, dst_size):
    dst_img = np.zeros([dst_size[1],dst_size[0],3], np.uint8)
    h,w = img_cv2.shape[:2]
    dst_w,dst_h = dst_size
    aspect_ratio_w = dst_w/w
    aspect_ratio_h = dst_h/h
    aspect_ratio = min([aspect_ratio_h,aspect_ratio_w])

    _h = min([ int(h*aspect_ratio),dst_h])
    _w = min([ int(w*aspect_ratio),dst_w])
    _tmp_img = cv2.resize(img_cv2,dsize=(_w,_h))
    dst_img[:_h,:_w] = _tmp_img[:,:]

    return dst_img, aspect_ratio


def _normalize_retinaface(img_cv2,mlu=False):
    # img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    # mean = [123.675, 116.28, 103.53]
    # std = [58.395, 57.12, 57.375]
    img_data = np.asarray(img_cv2, dtype=np.float32)
    if mlu:
        return img_data #[0,255]
    else:
        mean = (104, 117, 123)
        img_data = img_data - mean
        img_data = img_data.astype(np.float32) #[0,1] normalized
    return img_data

def preprocess_retinaface(img_cv2, dst_size=[1024,720],mlu=False):
    """
    :param img_cv2: 
    :param img_size: [w,h]
    :param mlu: 
    :return: 
    """
    resized_img, aspect_ratio = resize(img_cv2, dst_size=dst_size)
    resized_img_copy = resized_img.copy()
    img_data = _normalize_retinaface(resized_img ,mlu=mlu)
    img_data = np.transpose(img_data, axes=[2, 0, 1])
    img_data = np.expand_dims(img_data, axis=0)
    img_t = torch.from_numpy(img_data)
    return img_t, resized_img_copy, aspect_ratio

def fetch_cpu_data(x,use_half_input=False, to_numpy=False):
    if use_half_input:
        output = x.cpu().type(torch.FloatTensor)
    else:
        output = x.cpu()
    if to_numpy:
        return output.detach().numpy()
    else:
        return output.detach()


ct.set_core_number(4)
ct.set_core_version("MLU270")

if __name__ == '__main__':
    MLU=True
    datapath = './data'
    imgext = '.jpg'
    IMG_SIZE = [1024,720]#[w,h]
    threshold=0.8
    image_list = [pj(datapath, c) for c in os.listdir(datapath) if c.endswith(imgext)]
    input_img = [cv2.imread(c) for c in image_list]
    data = [preprocess_retinaface(c, dst_size=IMG_SIZE,mlu=MLU) for c in input_img]
    raw_img = [ c[1] for c in data]
    data = [c[0] for c in data]
    print('len of data: %d' % len(data))
    data = torch.cat(data, dim=0)
    print('data shape =', data.shape)
    n,c,h,w = data.shape

    print('==pytorch==')
    use_device = 'cpu'
    loading = False if MLU else True
    print('loading =', loading)
    model_path = 'weights/face_det/mobilenet0.25_Final.pth'
    model_path = os.path.abspath(model_path)
    print(model_path)
    infer = RetinaFaceDetModule(model_path=model_path,H=IMG_SIZE[1],W=IMG_SIZE[0],use_cpu=True,loading=loading)
    # infer = RetinaFaceDetModuleNMS(model_path=model_path, use_cpu=True, loading=loading)
    # infer.set_default_size([IMG_SIZE[1], IMG_SIZE[0], 3])
    print('==end==')

    if MLU:
        print('==mlu layer by layer==')
        data = data.to(ct.mlu_device())
        infer = mlu_quantize.quantize_dynamic_mlu(infer)
        checkpoint = torch.load('./weights/face_det/retinaface_mlu_int8.pth', map_location='cpu')
        infer.load_state_dict(checkpoint, strict=False)
        # model.eval().float()
        infer = infer.to(ct.mlu_device())
        print('==start infer==')
        preds = infer(data)
        print('==end==')
        net_outputs = fetch_cpu_data(preds, use_half_input=False, to_numpy=True)
        net_outputs = torch.FloatTensor(net_outputs)
    else:
        net_outputs = infer(data)

    print('net_outputs.shape =',net_outputs.shape)
    preds = execute_batch_mlu(net_outputs,batch_shape=[n,c,h,w],threshold=0.8, topk=5000, keep_topk=750,nms_threshold=0.2)
    # preds = net_outputs
    print('preds.shape =', len(preds))
    # print('preds.shape =', preds.shape)

    # preds = preds.cpu().data.numpy()
    # for bnt, dets in enumerate(preds):
    #     # dets = (n,d)
    #     img = raw_img[bnt]
    #     for idx, b in enumerate(dets):
    #         # b (d)
    #         if b[4] < threshold:
    #             continue
    #         scale=1.3
    #         cv2.rectangle(img, (int(b[0]), int(b[1])), ( int(b[2]), int(b[3])), (0, 255, 255), thickness=3)
    #
    #     savename = image_list[bnt].split(imgext)[0] + '.png'
    #     savefullname = pj(savename)
    #     cv2.imwrite(savefullname, img)


    for bnt, dets in enumerate(preds):
        img = raw_img[bnt]
        for idx, b in enumerate(dets):
            b = list(map(int, b))
            scale = 1.3
            pwidth = int((b[2] - b[0]) * scale)
            pheight = int((b[3] - b[1]) * scale)
            pcx = int((b[2] + b[0]) / 2)
            pcy = int((b[3] + b[1]) / 2)

            cv2.rectangle(img, (b[0],b[1]), (b[2],b[3]) , (0,255,255) , thickness=3)

        savename = image_list[bnt].split(imgext)[0] + '.png'
        savefullname = pj(savename)
        cv2.imwrite(savefullname, img)


