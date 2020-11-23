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
import torchvision
# from torchvision.models.quantization.utils import quantize_model
import torch_mlu
import torch_mlu.core.mlu_model as ct
import torch_mlu.core.mlu_quantize as mlu_quantize
import argparse
import cv2
import numpy as np
import random
from sklearn.preprocessing import normalize
from cfg import model_dict
from eval_utils.inference_api import Inference

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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

    ct.set_core_number(args.core_number)
    ct.set_core_version(args.mcore)
    print("batch_size is {}, core number is {}".format(args.batch_size, args.core_number))

    image_list = [ pj(args.data,c) for c in os.listdir(args.data) if c.endswith(args.ext) ]
    K = min([len(image_list),128])
    image_list = random.sample(image_list,K)
    print('sampled %d data'%len(image_list))

    input_img = [cv2.imread(c) for c in image_list]
    data = [preprocess(c , mlu=False) for c in input_img]
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
        model = infer._load_model()

    if args.quantization:
        mean = [0, 0, 0]
        std = [1.0, 1.0, 1.0]
        qconfig = {'iteration':data.shape[0], 'use_avg':True, 'data_scale':1.0, 'mean':mean, 'std':std, 'per_channel':False}
        model_quantized = mlu_quantize.quantize_dynamic_mlu(model, qconfig, dtype='int8', gen_quant = True)
        #print(model_quantized)
        #print('data:',data)
        model_quantized(data)
        torch.save(model_quantized.state_dict(), "./resnet101_mlu_int8.pth")
        print("Resnet101 int8 quantization end!")
    else:
        if not args.mlu:
            with torch.no_grad():
                out = model(data)
                out = out.data.cpu().numpy().reshape(-1,512)
                np.save('cpu_out.npy',out)
            # np.savetxt("cpu_out.txt", out.cpu().detach().numpy().reshape(-1,1), fmt='%.6f')
            print("run cpu finish!")
        else:
            # model = quantize_model(model, inplace=True)
            model = mlu_quantize.quantize_dynamic_mlu(model)
            checkpoint = torch.load('./resnet101_mlu_int8.pth', map_location='cpu')
            model.load_state_dict(checkpoint, strict=False)
            # model.eval().float()
            model = model.to(ct.mlu_device())
            if args.jit:
                if args.save_cambricon:
                    ct.save_as_cambricon(args.mname)
                    if args.fake_device:
                        ct.set_device(-1)
                traced_model = torch.jit.trace(model, data, check_trace=False)
                #print(traced_model.graph)
                if not args.save_cambricon:
                    out = traced_model(data)
                    if args.half_input:
                        out = out.cpu().type(torch.FloatTensor)
                    else:
                        out = out.cpu()
                    np.save('mlu_out_jit.txt',out.detach().numpy().reshape(-1,512))
                    # np.savetxt("mlu_out_firstconv_half_4c.txt", out.detach().numpy().reshape(-1,1), fmt='%.6f')
                    print("run mlu fusion finish!")
                else:
                    traced_model(data)
                    ct.save_as_cambricon("")
                    print("Save Resnet101 offline cambricon successfully!")
            else:
                out = model(data)
                print('out:',out.shape)
                np.save('mlu_out.txt', out.detach().numpy().reshape(-1, 512))
                print("run mlu layer_by_layer finish!")


