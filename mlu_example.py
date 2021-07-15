import sys,os
pj = os.path.join
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
sys.path.append(curPath)
sys.path.append(pj(curPath,'FaceRecog'))
# sys.path.append(pj(curPath,'Pytorch_Retinaface'))

import torch
import time
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
from FaceRecog.eval_utils.inference_api import Inference
# from Pytorch_Retinaface.retina_infer import RetinaFaceDet

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

def preprocessv2(img_cv2, mlu=False):
    img_format = cv2.resize(img_cv2,(112,112))
    img_data = _normalize(img_format,mlu=mlu)
    img_data = np.transpose(img_data, axes=[2, 0, 1])
    img_data = np.expand_dims(img_data, axis=0)
    img_t = torch.from_numpy(img_data)
    return img_t


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check',action='store_true',help='result will be compared only...')
    parser.add_argument('--model_name', default='resnet101_irse_mx')
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
        pytorch_result = np.load('cpu_out.npy')
        mlu_result = np.load('mlu_out.npy')
        cpu_quant_result = np.load('cpu_quant_out.npy')
        # mlu_jit_result = np.load('mlu_out_jit.npy')

        print('shapes:',pytorch_result.shape,mlu_result.shape)#,mlu_jit_result.shape)
        B = pytorch_result.shape[0]
        assert B == mlu_result.shape[0], 'Err!!!'

        diff1 = pytorch_result - mlu_result
        diff1 = math.sqrt((diff1**2).sum()) / B
        print('mean instance difference: %f' % diff1)

        # diff2 = pytorch_result - mlu_jit_result
        # diff2 = math.sqrt((diff2**2).sum()) / B
        # print('mean instance difference: %f' % diff2)

        diff3 = pytorch_result - cpu_quant_result
        diff3 = math.sqrt((diff3**2).sum()) / B
        print('mean instance difference: %f' % diff3)

        exit(0)


    ct.set_core_number(args.core_number)
    ct.set_core_version(args.mcore)
    print("batch_size is {}, core number is {}".format(args.batch_size, args.core_number))

    image_list = [ pj(args.data,c) for c in os.listdir(args.data) if c.endswith(args.ext) ]
    image_list = [
        '/project/data/tupu/5ea5090844c2683545aa564f_0.jpg'
    ]
    K = min([len(image_list),args.batch_size])
    # image_list = random.sample(image_list,K)
    image_list = image_list[:K]
    print('sampled %d data'%len(image_list))

    input_img = [cv2.imread(c) for c in image_list]
    # data = [preprocess(c , mlu=args.mlu) for c in input_img]
    data = [preprocessv2(c, mlu=args.mlu) for c in input_img]
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
    backbone_type = args.model_name
    model_type = model_dict[args.model_name]['weights']
    model_pth = pj(model_dict[args.model_name]['path'], model_type)
    model_pth = os.path.abspath(model_pth)
    infer = Inference(backbone_type=backbone_type,
                      ckpt_fpath=model_pth,
                      device=use_device)
    print('==end==')

    if not args.mlu:
        model = infer.model
        model.eval().float()
    else:
        model = infer._get_model()

    if args.quantization:
        print('doing quantization on cpu')
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        use_avg = False if data.shape[0] == 1 else True
        qconfig = {'iteration':data.shape[0], 'use_avg':use_avg,
                   'data_scale':1.0, 'mean':mean, 'std':std, 'per_channel':False, 'firstconv':True}
        model_quantized = mlu_quantize.quantize_dynamic_mlu(model, qconfig, dtype='int8', gen_quant = True)
        #print(model_quantized)
        #print('data:',data)
        print('data.shape=',data.shape)
        output = model_quantized(data)
        torch.save(model_quantized.state_dict(), "./weights/face_rec/resnet101_mlu_int8.pth")
        print("Resnet101 int8 quantization end!")
        if args.half_input:
            output = output.cpu().type(torch.FloatTensor)
        else:
            output = output.cpu()
        print('saving', output.shape)
        np.save('cpu_quant_out.npy', output.detach().numpy().reshape(-1, 512))

    else:
        if not args.mlu:
            print('doing cpu inference')
            with torch.no_grad():
                bgt = time.time()
                out = model(data)
                out = out.data.cpu().numpy().reshape(-1,512)
                timing = time.time() - bgt
                print('using {:.3f}s per img'.format(timing/K))
                np.save('cpu_out.npy',out)
            # np.savetxt("cpu_out.txt", out.cpu().detach().numpy().reshape(-1,1), fmt='%.6f')
            print("run cpu finish!")
        else:
            print('doing mlu inference')
            # model = quantize_model(model, inplace=True)
            model = mlu_quantize.quantize_dynamic_mlu(model)
            checkpoint = torch.load('./weights/face_rec/resnet101_mlu_int8.pth', map_location='cpu')
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
                bgt = time.time()
                out = traced_model(data)
                print('end inference')
                timing = time.time() - bgt
                print('using {:.3f}s per img'.format(timing / K))
                if args.half_input:
                    out = out.cpu().type(torch.FloatTensor)
                else:
                    out = out.cpu()
                print('saving', out.shape)
                np.save('mlu_out_jit.npy', out.detach().numpy().reshape(-1, 512))
                # np.savetxt("mlu_out_firstconv_half_4c.txt", out.detach().numpy().reshape(-1,1), fmt='%.6f')
                print("run mlu fusion finish!")
            else:
                print('using layer by layer inference')
                bgt = time.time()
                out = model(data)
                print('done')
                timing = time.time() - bgt
                print('using {:.3f}s per img'.format(timing / K))
                if args.half_input:
                    out = out.cpu().type(torch.FloatTensor)
                else:
                    out = out.cpu()
                print('out:', out.shape)
                np.save('mlu_out.npy', out.detach().numpy().reshape(-1, 512))
                print("run mlu layer_by_layer finish!")


