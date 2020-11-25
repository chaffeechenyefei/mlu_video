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
from sklearn.preprocessing import normalize
from cfg import model_dict
# from FaceRecog.eval_utils.inference_api import Inference
from Pytorch_Retinaface.retina_infer import RetinaFaceDet

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


def _normalize_retinaface(img_cv2,mlu=False):
    # img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    # mean = [123.675, 116.28, 103.53]
    # std = [58.395, 57.12, 57.375]
    img_cv2 = cv2.resize(img_cv2,dsize=(512,512))
    img_data = np.asarray(img_cv2, dtype=np.float32)
    if mlu:
        return img_data #[0,255]
    else:
        mean = (104, 117, 123)
        img_data = img_data - mean
        img_data = img_data.astype(np.float32) #[0,1] normalized
    return img_data

def preprocess_retinaface(img_cv2, mlu=False):
    img_format = _format(img_cv2)
    img_data = _normalize_retinaface(img_format,mlu=mlu)
    img_data = np.transpose(img_data, axes=[2, 0, 1])
    img_data = np.expand_dims(img_data, axis=0)
    img_t = torch.from_numpy(img_data)
    return img_t

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

    if args.check:
        pytorch_loc = np.load('cpu_loc.npy')
        mlu_loc = np.load('mlu_loc.npy')
        cpu_quant_loc = np.load('cpu_quant_loc.npy')
        # mlu_jit_result = np.load('mlu_out_jit.npy')


        B = pytorch_loc.shape[0]
        assert B == mlu_loc.shape[0], 'Err!!!'

        diff1 = pytorch_loc - mlu_loc
        diff1 = math.sqrt((diff1**2).sum()) / B
        print('mean instance difference: %f' % diff1)

        # diff2 = pytorch_result - mlu_jit_result
        # diff2 = math.sqrt((diff2**2).sum()) / B
        # print('mean instance difference: %f' % diff2)

        diff3 = pytorch_loc - cpu_quant_loc
        diff3 = math.sqrt((diff3**2).sum()) / B
        print('mean instance difference: %f' % diff3)

        exit(0)


    ct.set_core_number(args.core_number)
    ct.set_core_version(args.mcore)
    print("batch_size is {}, core number is {}".format(args.batch_size, args.core_number))

    image_list = [ pj(args.data,c) for c in os.listdir(args.data) if c.endswith(args.ext) ]
    K = min([len(image_list),args.batch_size])
    # image_list = random.sample(image_list,K)
    image_list = image_list[:K]
    print('sampled %d data'%len(image_list))
    print(image_list[0])

    input_img = [cv2.imread(c) for c in image_list]
    data = [preprocess_retinaface(c , mlu=args.mlu) for c in input_img]
    print('len of data: %d'%len(data))
    #print('data:',data)
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
    model_path = 'weights/face_det/mobilenet0.25_Final.pth'
    model_path = os.path.abspath(model_path)
    infer = RetinaFaceDet(model_path=model_path,use_cpu=True,loading=False)
    print('==end==')

    if not args.mlu:
        model = infer.net
        model.eval().float()
    else:
        model = infer._get_model()

    if args.quantization:
        print('doing quantization on cpu')
        mean = [ 104 / 255, 117 / 255, 123 / 255]
        std = [1/255,1/255,1/255]
        use_avg = False if data.shape[0] == 1 else True
        qconfig = {'iteration':data.shape[0], 'use_avg':use_avg, 'data_scale':1.0, 'mean':mean, 'std':std, 'per_channel':False}
        model_quantized = mlu_quantize.quantize_dynamic_mlu(model, qconfig, dtype='int8', gen_quant = True)
        #print(model_quantized)
        #print('data:',data)
        print('data.shape=',data.shape)
        locs, confs, landmss = model_quantized(data)
        torch.save(model_quantized.state_dict(), "./retinaface_mlu_int8.pth")
        print("Retinaface int8 quantization end!")
        _locs = fetch_cpu_data(locs,args.half_input)
        _confs = fetch_cpu_data(confs,args.half_input)
        _landmss = fetch_cpu_data(landmss,args.half_input)

        print('saving', _locs.shape,_confs.shape,_landmss.shape)
        np.save('cpu_quant_loc.npy', _locs)
        np.save('cpu_quant_conf.npy', _confs)
        np.save('cpu_quant_landms.npy', _landmss)

    else:
        if not args.mlu:
            print('doing cpu inference')
            with torch.no_grad():
                locs, confs, landmss = model(data)
                _locs = fetch_cpu_data(locs, args.half_input)
                _confs = fetch_cpu_data(confs, args.half_input)
                _landmss = fetch_cpu_data(landmss, args.half_input)

                np.save('cpu_loc.npy', _locs)
                np.save('cpu_conf.npy', _confs)
                np.save('cpu_landms.npy', _landmss)
            print("run cpu finish!")
        else:
            print('doing mlu inference')
            # model = quantize_model(model, inplace=True)
            model = mlu_quantize.quantize_dynamic_mlu(model)
            checkpoint = torch.load('./retinaface_mlu_int8.pth', map_location='cpu')
            model.load_state_dict(checkpoint, strict=False)
            # model.eval().float()
            model = model.to(ct.mlu_device())
            if args.jit:
                print('using jit inference')
                randinput = torch.rand(1,3,112,112)*255
                randinput = randinput.to(ct.mlu_device())
                traced_model = torch.jit.trace(model, randinput, check_trace=False)
                # print(traced_model.graph)
                print('start inference')
                locs, confs, landmss = traced_model(data)
                print('end inference')
                _locs = fetch_cpu_data(locs, args.half_input)
                _confs = fetch_cpu_data(confs, args.half_input)
                _landmss = fetch_cpu_data(landmss, args.half_input)
                print('saving', _locs.shape, _confs.shape, _landmss.shape)
                np.save('mlu_jit_loc.npy', _locs)
                np.save('mlu_jit_conf.npy', _confs)
                np.save('mlu_jit_landms.npy', _landmss)
                print("run mlu fusion finish!")
            else:
                print('using layer by layer inference')
                locs, confs, landmss = model(data)
                print('done')
                _locs = fetch_cpu_data(locs, args.half_input)
                _confs = fetch_cpu_data(confs, args.half_input)
                _landmss = fetch_cpu_data(landmss, args.half_input)
                print('saving', _locs.shape, _confs.shape, _landmss.shape)
                np.save('mlu_loc.npy', _locs)
                np.save('mlu_conf.npy', _confs)
                np.save('mlu_landms.npy', _landmss)
                print("run mlu layer_by_layer finish!")


