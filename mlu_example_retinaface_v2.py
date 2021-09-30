import sys,os
pj = os.path.join
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
sys.path.append(curPath)
# sys.path.append(pj(curPath,'FaceRecog'))
sys.path.append(pj(curPath,'Pytorch_Retinaface'))

import torch
import torch.nn as nn
import torchvision
# from torchvision.models.quantization.utils import quantize_model
import torch_mlu
import torch_mlu.core.mlu_model as ct
import torch_mlu.core.mlu_quantize as mlu_quantize
import argparse
import cv2
import numpy as np
import random,math
from Pytorch_Retinaface.retina_infer import RetinaFaceDetModule
# from Pytorch_Retinaface.retina_infer import RetinaFaceDetModuleNMS

"python XXX.py --mlu false  --quantization true"

torch.set_grad_enabled(False)

def str2bool(v):
     return v.lower() in ("yes", "true", "t", "1")

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

def fetch_cpu_data(x,use_half_input=False):
    if use_half_input:
        output = x.cpu().type(torch.FloatTensor)
    else:
        output = x.cpu()
    return output.detach().numpy()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check',action='store_true',help='result will be compared only...')
    parser.add_argument('--data',help='data path to the images used for quant')
    parser.add_argument('--ext',default='.jpg')

    parser.add_argument('--mlu', default=True, type=str2bool,
                        help='Use mlu to train model')
    parser.add_argument('--jit', default=True, type=str2bool,
                        help='Use jit for inference net')
    parser.add_argument("--quantized_mode", dest='quantized_mode', help=
    "the data type, 0-float16 1-int8 2-int16, default 1.",
                        default=1, type=int)
    parser.add_argument("--half_input", dest='half_input', help=
    "the input data type, 0-float32, 1-float16/Half, default 1.",
                        default=1, type=int)
    parser.add_argument('--core_number', default=4, type=int,
                        help='Core number of mfus and offline model with simple compilation.')
    parser.add_argument('--mcore', default='MLU270', type=str,
                        help="Set MLU Architecture")
    parser.add_argument('--quantization', default=False, type=str2bool,
                        help='Whether to quantize, set to True for quantization')
    parser.add_argument('--save_cambricon', default=False, type=str2bool,
                        help='Whether to save cambricon model')
    parser.add_argument("--fake_device", dest='fake_device',
                        help="genoff offline cambricon without mlu device if fake device is true. 1-fake_device, 0-mlu_device",
                        default=1, type=int)
    parser.add_argument("--mname", dest='mname', help="The name for the offline model to be generated",
                        default="resnet101_offline", type=str)
    parser.add_argument("--batch_size", dest="batch_size", help="batch size for one inference.",
                        default=1, type=int)
    args = parser.parse_args()

    # IMG_SIZE=[736,416] #[w,h]
    IMG_SIZE=[736,416]

    if args.check:
        pytorch_loc = np.load('cpu_pred.npy')
        mlu_loc = np.load('mlu_pred.npy')
        # cpu_quant_loc = np.load('cpu_quant_pred.npy')

        B = pytorch_loc.shape[0]
        assert B == mlu_loc.shape[0], 'Err!!!'

        diff1 = pytorch_loc - mlu_loc
        diff1 = math.sqrt((diff1**2).sum()) / B
        print('mean instance difference: %f' % diff1)

        # diff2 = pytorch_result - mlu_jit_result
        # diff2 = math.sqrt((diff2**2).sum()) / B
        # print('mean instance difference: %f' % diff2)
        #
        # diff3 = pytorch_loc - cpu_quant_loc
        # diff3 = math.sqrt((diff3**2).sum()) / B
        # print('mean instance difference: %f' % diff3)

        exit(0)


    ct.set_core_number(args.core_number)
    ct.set_core_version(args.mcore)
    print("batch_size is {}, core number is {}".format(args.batch_size, args.core_number))

    image_list = [ pj(args.data,c) for c in os.listdir(args.data) if c.endswith(args.ext) ]
    K = min([len(image_list),args.batch_size])
    image_list = image_list[:K]
    print('sampled %d data'%len(image_list))
    print(image_list[0])

    input_img = [cv2.imread(c) for c in image_list]
    data = [preprocess_retinaface(c , dst_size=IMG_SIZE , mlu=args.mlu) for c in input_img]
    data = [c[0] for c in data]
    print('len of data: %d'%len(data))
    data = torch.cat(data,dim=0)
    print('data shape =',data.shape)

    if args.mlu:
        if args.half_input:
            data = data.type(torch.HalfTensor)
        data = data.to(ct.mlu_device())

    # model = torchvision.models.resnet50()
    print('==pytorch==')
    use_device = 'cpu'
    loading = True if not args.mlu else False
    print('loading =',loading)
    model_path = 'weights/face_det/mobilenet0.25_Final.pth'
    model_path = os.path.abspath(model_path)
    print(model_path)
    infer = RetinaFaceDetModule(model_path=model_path,H=IMG_SIZE[1],W=IMG_SIZE[0],use_cpu=True,loading=loading)
    # infer = RetinaFaceDetModuleNMS(model_path=model_path, use_cpu=True, loading=loading)
    print('==end==')

    model = infer.eval()

    if args.quantization:
        print('doing quantization on cpu')
        mean = [ 104 / 255, 117 / 255, 123 / 255]
        std = [1/255,1/255,1/255]
        use_avg = False if data.shape[0] == 1 else True
        qconfig = {'iteration':data.shape[0],
                   'use_avg':use_avg, 'data_scale':1.0, 'mean':mean, 'std':std,
                   'per_channel':False, 'firstconv':True}
        model_quantized = mlu_quantize.quantize_dynamic_mlu(model, qconfig, dtype='int8', gen_quant = True)
        #print(model_quantized)
        #print('data:',data)
        print('data.shape=',data.shape)
        preds = model_quantized(data)
        torch.save(model_quantized.state_dict(), "./weights/face_det/retinaface_mlu_int8.pth")
        print("Retinaface int8 quantization end!")
        _preds = fetch_cpu_data(preds,args.half_input)


        print('saving', _preds.shape)
        np.save('cpu_quant_pred.npy', _preds)


    else:
        if not args.mlu:
            print('doing cpu inference')
            with torch.no_grad():
                preds = model(data)
                _preds = fetch_cpu_data(preds, args.half_input)

                np.save('cpu_pred.npy', _preds)
            print("run cpu finish!")
        else:
            print('doing mlu inference')
            # model = quantize_model(model, inplace=True)
            model = mlu_quantize.quantize_dynamic_mlu(model)
            checkpoint = torch.load('./weights/face_det/retinaface_mlu_int8.pth', map_location='cpu')
            model.load_state_dict(checkpoint, strict=False)
            # model.eval().float()
            model = model.to(ct.mlu_device())
            if args.jit:
                print('using jit inference')
                randinput = torch.rand(1,3,IMG_SIZE[1],IMG_SIZE[0])*255
                randinput = randinput.to(ct.mlu_device())
                traced_model = torch.jit.trace(model, randinput, check_trace=False)
                # print(traced_model.graph)
                print('start inference')
                preds = traced_model(data)
                print('end inference')
                _preds = fetch_cpu_data(preds, args.half_input)
                print('saving', _preds.shape)
                np.save('mlu_jit_pred.npy', _preds)
                print("run mlu fusion finish!")
            else:
                print('using layer by layer inference')
                data = data.to(ct.mlu_device())
                preds = model(data)
                print('done')
                _preds = fetch_cpu_data(preds, args.half_input)
                print('saving', _preds.shape )
                np.save('mlu_pred.npy', _preds)
                print("run mlu layer_by_layer finish!")


