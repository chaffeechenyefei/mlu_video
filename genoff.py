from __future__ import division
import sys,os
pj = os.path.join
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
sys.path.append(curPath)


import os
import sys
import logging
import argparse
import torch
import torchvision.models as models
import torch_mlu.core.mlu_model as ct
import torch_mlu.core.mlu_quantize as mlu_quantize

#configure logging path
logging.basicConfig(level=logging.INFO,
                    format='[genoff.py line:%(lineno)d] - %(levelname)s: %(message)s')
logger = logging.getLogger("TestNets")

#support model
support = ['retinaface','resnet50', 'resnext50_32x4d', 'resnext101_32x8d', 'vgg16', 'inception_v3', 'mobilenet',
           'googlenet', 'resnet101', 'mobilenet_v2', 'mobilenet_v3', 'ssd', 'yolov3', 'alexnet',
           'resnet18', 'resnet34', 'resnet152', 'vgg16_bn', 'squeezenet1_0', 'squeezenet1_1',
           'densenet121', 'faceboxes', 'yolov2', 'pnet', 'rnet', 'onet', 'east', 'ssd_mobilenet_v1',
           'efficientnet', 'ssd_mobilenet_v2', 'ssd', 'PreActResNet50', 'PreActResNet101',
           'fasterrcnn_fpn', 'centernet', 'fcn8s', 'segnet', 'vdsr', 'fsrcnn', 'yolov4', 'yolov5', 'resnet101_irse_mx']

abs_path = os.path.dirname(os.path.realpath(__file__))

torch.set_grad_enabled(False)

def get_args():
    parser = argparse.ArgumentParser(description='Generate offline model.')
    parser.add_argument("-model", dest='model', help=
                        "The network name of the offline model needs to be generated",
                        default="", type=str)
    parser.add_argument("-core_number", dest="core_number", help=
                        "Core number of offline model with simple compilation. ",
                        default=1, type=int)
    parser.add_argument("-mname", dest='mname', help=
                        "The name for the offline model to be generated",
                        default="offline", type=str)
    parser.add_argument("-mcore", dest='mcore', help=
                        "Specify the offline model run device type.",
                        default="MLU270", type=str)
    parser.add_argument("-modelzoo", dest='modelzoo', type=str,
                        help="Specify the path to the model weight file.",
                        default=None)
    parser.add_argument("-channel_size", dest="channel_size", help=
                        "channel size for one inference.",
                        default=3, type=int)
    parser.add_argument("-batch_size", dest="batch_size", help="batch size for one inference.",
                        default=1, type=int)
    parser.add_argument("-in_height", dest="in_height", help="input height.",
                        default=224, type=int)
    parser.add_argument("-in_width", dest="in_width", help="input width.",
                        default=224, type=int)
    parser.add_argument("-half_input", dest='half_input', help=
                        "the input data type, 0-float32, 1-float16/Half, default 1.",
                        default=1, type=int)
    parser.add_argument("-fake_device", dest='fake_device', help=
                        "genoff offline cambricon without mlu device if \
                        fake device is true. 1-fake_device, 0-mlu_device",
                        default=1, type=int)
    parser.add_argument("-quantized_mode", dest='quantized_mode', help=
                        "the data type, 1-mlu int8, 2-mlu int16, default 1.",
                        default=1, type=int)
    parser.add_argument("-input_format", dest="input_format", help=
                        "describe input image channel order in C direction, \
                        0-rgba, 1-argb, 2-bgra, 3-abgr",
                        default=0, type=int)
    parser.add_argument("-autotune", dest="autotune", help="autotune mode",
                        default=0, type=int)
    parser.add_argument("-autotune_config_path", dest="autotune_config_path", \
                        help="autotune configuration file path", default="config.ini", type=str)
    parser.add_argument("-autotune_time_limit", dest="autotune_time_limit", \
                        help="time limit for autotune", default=120, type=int)

    return parser.parse_args()

def genoff(model, mname, batch_size, core_number, in_heigth, in_width,
           half_input, input_format, fake_device):
    # set offline flag
    ct.set_core_number(core_number)
    ct.set_core_version(mcore)
    if fake_device:
        ct.set_device(-1)
    ct.save_as_cambricon(mname)
    ct.set_input_format(input_format)

    # construct input tensor
    net = None
    in_h = 224
    in_w = 224

    model = 'resnet101_irse_mx'
    print(model)
    sys.path.append(pj(curPath, 'FaceRecog'))
    from cfg import model_dict
    from FaceRecog.eval_utils.inference_api import Inference
    use_device = 'cpu'
    infer = Inference(backbone_type=model,
                      ckpt_fpath=None,
                      device=use_device)
    net = infer._get_model()
    net = mlu_quantize.quantize_dynamic_mlu(net)
    checkpoint = torch.load('./weights/face_rec/resnet101_mlu_int8.pth', map_location='cpu')
    net.load_state_dict(checkpoint, strict=False)
    net = net.to(ct.mlu_device())
    in_h = 112
    in_w = 112

    # prepare input
    example_mlu = torch.randn(batch_size, args.channel_size, in_h, in_w, dtype=torch.float)
    randn_mlu = torch.randn(1, args.channel_size, in_h, in_w, dtype=torch.float)
    if half_input:
        randn_mlu = randn_mlu.type(torch.HalfTensor)
        example_mlu = example_mlu.type(torch.HalfTensor)


    # set autotune
    if autotune:
        ct.set_autotune(True)
        ct.set_autotune_config_path(autotune_config_path)
        ct.set_autotune_time_limit(autotune_time_limit)

    net_traced = torch.jit.trace(net.to(ct.mlu_device()),
                                 randn_mlu.to(ct.mlu_device()),
                                 check_trace=False)


    # run inference and save cambricon
    net_traced(example_mlu.to(ct.mlu_device()))


def genoff_retinaface(model, mname, batch_size, core_number, in_heigth, in_width,
           half_input, input_format, fake_device):
    # set offline flag
    ct.set_core_number(core_number)
    ct.set_core_version(mcore)
    if fake_device:
        ct.set_device(-1)
    ct.save_as_cambricon(mname)
    ct.set_input_format(input_format)

    # construct input tensor
    net = None

    # IMG_SIZE = [736,416]#[w,h]
    IMG_SIZE = [736, 416]  # [w,h]

    sys.path.append(pj(curPath, 'Pytorch_Retinaface'))
    from Pytorch_Retinaface.retina_infer import RetinaFaceDetModule

    print('==pytorch==')
    loading = False
    print('loading =',loading)
    model_path = 'weights/face_det/mobilenet0.25_Final.pth'
    model_path = os.path.abspath(model_path)
    print(model_path)
    infer = RetinaFaceDetModule(model_path=model_path,H=IMG_SIZE[1],W=IMG_SIZE[0],use_cpu=True,loading=loading)
    # infer.set_default_size([IMG_SIZE[1],IMG_SIZE[0],3])
    net = infer.eval()
    print('==end==')

    net = mlu_quantize.quantize_dynamic_mlu(net)
    checkpoint = torch.load('./weights/face_det/retinaface_mlu_int8.pth', map_location='cpu')
    net.load_state_dict(checkpoint, strict=False)
    net = net.to(ct.mlu_device())


    # prepare input
    example_mlu = torch.randn(batch_size, args.channel_size, IMG_SIZE[1], IMG_SIZE[0], dtype=torch.float)
    randn_mlu = torch.randn(1, args.channel_size, IMG_SIZE[1], IMG_SIZE[0], dtype=torch.float)
    if half_input:
        randn_mlu = randn_mlu.type(torch.HalfTensor)
        example_mlu = example_mlu.type(torch.HalfTensor)


    net_traced = torch.jit.trace(net.to(ct.mlu_device()),
                                 randn_mlu.to(ct.mlu_device()),
                                 check_trace=False)


    # run inference and save cambricon
    net_traced(example_mlu.to(ct.mlu_device()))

if __name__ == "__main__":
    args = get_args()
    model = args.model
    core_number = args.core_number
    mname = args.mname
    modelzoo = args.modelzoo
    mcore = args.mcore
    batch_size = args.batch_size
    in_height = args.in_height
    in_width = args.in_width
    half_input = args.half_input
    input_format = args.input_format
    fake_device = args.fake_device
    autotune = args.autotune
    autotune_config_path = args.autotune_config_path
    autotune_time_limit = args.autotune_time_limit

    #check param
    assert model != "", "Generating the offline model requires" + \
        "specifying the generated network name."
    assert model in support, "The specified model is not currently supported.\n" + \
        "Support model list: " + str(support)
    assert not fake_device or not autotune, "Fake device is not supported for autotune!"

    # env
    if modelzoo != None:
        os.environ['TORCH_HOME'] = modelzoo
        logger.info("TORCH_HOME: " + modelzoo)
    # else:
    #     TORCH_HOME = os.getenv('TORCH_HOME')
    #     if TORCH_HOME == None:
    #         print("Warning: please set environment variable TORCH_HOME such as $PWD/models/pytorch")
    #         exit(1)

    #genoff
    logger.info("Generate offline model: " + model)
    if model == 'retinaface':
        genoff_retinaface(model, mname, batch_size, core_number,
                          in_height, in_width, half_input, input_format, fake_device)
    elif model == 'resnet101_irse_mx':
        genoff(model, mname, batch_size, core_number,
           in_height, in_width, half_input, input_format, fake_device)
    else:
        print('Unknown model')
