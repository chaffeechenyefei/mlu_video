import sys,os
pj = os.path.join
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
sys.path.append(curPath)
sys.path.append(pj(curPath,'FaceRecog'))

import torch
import os, argparse
from cfg import model_dict
from FaceRecog.eval_utils.inference_api import Inference
pj = os.path.join


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',default='resnet101_irse_mx')
    args = parser.parse_args()

    backbone_type = args.model_name
    model_type = model_dict[args.model_name]['weights']
    model_pth = pj(model_dict[args.model_name]['path'], model_type)
    model_pth = os.path.abspath(model_pth)
    infer = Inference(backbone_type=backbone_type,
                      ckpt_fpath=model_pth,
                      device='cpu')
    model = infer.model
    model.eval().float()

    new_model_name = pj(model_dict[args.model_name]['path'], model_dict[args.model_name]['no_serial_weights'])
    torch.save(model.state_dict(), new_model_name,_use_new_zipfile_serialization=False)