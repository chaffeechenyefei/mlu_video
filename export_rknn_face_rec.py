import sys,os
pj = os.path.join
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
sys.path.append(curPath)
sys.path.append(pj(curPath,'FaceRecog'))
# sys.path.append(pj(curPath,'Pytorch_Retinaface'))

import numpy as np
import cv2
from rknn.api import RKNN
import torch
# from checkpoint.checkpoint import CheckpointMgr
# from config import config_param
import argparse
# from Pytorch_Retinaface.retina_infer import RetinaFaceDetModule
from FaceRecog.eval_utils.inference_api import Inference

pj = os.path.join

QUANTIZE_ON = False

def export_pytorch_model(param, pth_path, pt_path):
    model_path = os.path.abspath(pth_path)
    print(model_path)

    use_device = 'cpu'
    backbone_type = args.model_name

    net_object = Inference(backbone_type=backbone_type,
                    ckpt_fpath=model_path,
                    device=use_device)
    net = net_object.model
    net.eval()

    trace_model = torch.jit.trace(net, torch.Tensor(1,3,param['h'],param['w']))
    trace_model.save(pt_path)

    data = torch.rand(1,3,param['h'],param['w'])
    output = net(data)
    if isinstance(output, list):
        for o in output:
            print(o.shape)
    else:
        print(output.shape)
    return net

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


def _normalize_(img_cv2,npu=False):
    img_data = np.asarray(img_cv2, dtype=np.float32)
    if npu:
        return img_data #[0,255]
    else:
        mean = 127.5
        std = 127.5
        img_data = (img_data - mean)/std
        img_data = img_data.astype(np.float32) #[0,1] normalized
    return img_data

def preprocess(img_cv2, dst_size=[112,112], npu=False):
    """
    :param img_cv2: 
    :param img_size: [w,h]
    :param mlu: 
    :return: 
    """
    img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    resized_img, aspect_ratio = resize(img_rgb, dst_size=dst_size)
    resized_img_copy = resized_img.copy()
    img_data = _normalize_(resized_img, npu=npu)
    img_data = np.transpose(img_data, axes=[2, 0, 1])
    img_data = np.expand_dims(img_data, axis=0)
    img_t = torch.from_numpy(img_data)
    return img_t, resized_img_copy, aspect_ratio


def show_perfs(perfs):
    perfs = 'perfs: {}\n'.format(perfs)
    print(perfs)


def softmax(x):
    return np.exp(x)/sum(np.exp(x))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',default='resnet50_irse_mx')
    parser.add_argument('--weight', default='r50irsemx_webface260m_model_539_no_serial.pth')
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--fast',action='store_true')#是否采用pre-compile
    parser.add_argument('--img', type = str, default=None)
    args = parser.parse_args()

    dtype = 'int8'

    param = {
        'w': 112,
        'h': 112,
    }

    model_path = 'weights/face_rec/'
    model_pth_full_name = 'weights/face_rec/{}'.format(args.weight)
    print('using model path: {}'.format(model_path))
    args_w = param['w']
    args_h = param['h']
    model_date =  '2022xx'
    pre_compile = True if args.fast else False
    pre_compile_str = 'fast' if pre_compile else 'slow'
    model_pt_full_name = pj(model_path, '{}_{}_{}_{:d}x{:d}_{}.pt'.format(args.model_name,dtype,model_date,
                                                                               args_w,args_h, pre_compile_str))

    if not args.test_only:
        print('--> export_pytorch_model')
        pynet = export_pytorch_model(param, model_pth_full_name, model_pt_full_name)
    # exit(0)


    model_rknn_full_name = model_pt_full_name.replace('.pt','.rknn')
    input_size_list = [[3, args_h, args_w]]

    # Create RKNN object
    print('--> create rknn object')
    rknn = RKNN()

    if not args.test_only:
    # pre-process config
        print('--> Config model')
        mean = [127.5 , 127.5 , 127.5 ]
        std = [127.5 , 127.5 , 127.5 ]
        rknn.config(mean_values=[mean], std_values=[std], reorder_channel='0 1 2',
                    target_platform='rk3399pro',
                    optimization_level=3,
                    quantize_input_node=QUANTIZE_ON,
                    output_optimize=1,

                    )
        print('done')

    # Load Pytorch model
    print('--> Loading model')
    ret = rknn.load_pytorch(model=model_pt_full_name, input_size_list=input_size_list)
    if ret != 0:
        print('** Load Pytorch model failed!')
        exit(ret)
    print('done')

    # Build model
    if not args.test_only:
        print('--> Building model')
        ret = rknn.build(do_quantization=True, dataset='./datasets.txt', pre_compile=pre_compile)
        if ret != 0:
            print('** Build model failed!')
            exit(ret)
        print('done')

        # Export RKNN model
        print('--> Export RKNN model')
        ret = rknn.export_rknn(model_rknn_full_name)
        if ret != 0:
            print('Export {} failed!'.format(model_rknn_full_name))
            exit(ret)
        print('done')

    # fast模式下， cpu上不能进行模拟推理
    if not args.fast:
        test_img_path = args.img
        if test_img_path is None or test_img_path == '':
            test_img_path = './data/test.jpg'
            print('using default image for test = {}'.format(test_img_path))

        else:
            print('using image for test = {}'.format(test_img_path))
        img = cv2.imread(test_img_path)
        pydata ,img,_ = preprocess(img,[args_w, args_h])

        print('--> pytorch infering')
        outputs = pynet(pydata)
        if isinstance(outputs, list):
            for o in outputs:
                print(o.shape)
                print(o.reshape(-1)[:5].data.numpy())
        else:
            print(outputs.shape)


        print('--> Import RKNN model and infering')
        ret = rknn.load_rknn(model_rknn_full_name)

        # Init runtime environment
        print('--> Init runtime environment')
        ret = rknn.init_runtime()
        if ret != 0:
            print('Init runtime environment failed')
            exit(ret)
        print('done')

        # Inference
        print('--> Running model')
        outputs = rknn.inference(inputs=[img])

        if isinstance(outputs, list):
            for o in outputs:
                print(o.shape)
                print(o.reshape(-1)[:5])
        else:
            print(outputs.shape)
        # print(outputs[0][0].shape)
        # print(outputs[0][0])#if model already softmax
        print('done')

    # print('--> Second Running')
    # img = cv2.imread('./data/fire/fire001.jpg')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.resize(img,dsize=(224,224))
    # outputs = rknn.inference(inputs=[img])
    # print(outputs[0][0])
    # # print(softmax(np.array(outputs[0][0])))  # id_of_output id_of_batch


    rknn.release()