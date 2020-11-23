import sys,os
pj = os.path.join
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
sys.path.append(curPath)
sys.path.append(pj(curPath,'FaceRecog'))

import torch,argparse
import numpy as np
import cv2
import pickle
from torch.utils.data import DataLoader
from FaceRecog.datasets.plateloader import BasicLoaderV2,BasicLoader
# from udftools.functions import face_format
from cfg import model_dict
from functools import reduce

pj = os.path.join

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


from eval_utils.inference_api import Inference

def save_obj(obj, name ):
    with open( name, 'wb') as f:
        pickle.dump(obj, f)

def load_obj(name ):
    with open(name , 'rb') as f:
        return pickle.load(f)

def face_normalize(img_cv2):
    img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    mean = 127.5
    std = 127.5
    img_data = np.asarray(img_cv2, dtype=np.float32)
    img_data = img_data - mean
    img_data = img_data / std
    img_data = img_data.astype(np.float32)
    return img_data

def face_format(img_cv2, format_size=112):
    org_h, org_w = img_cv2.shape[0:2]
    rescale_ratio = format_size / max(org_h, org_w)
    h, w = int(org_h * rescale_ratio), int(org_w * rescale_ratio)
    img_rescaled = cv2.resize(img_cv2, (w, h))
    paste_pos = [int((format_size - w) / 2), int((format_size - h) / 2)]
    img_format = np.zeros((format_size, format_size, 3), dtype=np.uint8)
    img_format[paste_pos[1]:paste_pos[1] + h, paste_pos[0]:paste_pos[0] + w] = img_rescaled
    return img_format



def main(args):
    backbone_type = 'resnet50_irse_mx'
    model_type = model_dict[args.model_name]['weights']
    model_pth = pj(model_dict[args.model_name]['path'],model_type)
    model_pth = os.path.abspath(model_pth)

    if torch.cuda.is_available():
        use_device = 'cuda'
    else:
        use_device = 'cpu'

    infer = Inference(backbone_type=backbone_type,
                      ckpt_fpath=model_pth,
                      device=use_device)

    datapath = args.data


    db_feature = {}

    with torch.no_grad():
        img_cv = cv2.imread('/Users/marschen/Ucloud/Data/debug.png')
        # img_face = face_format(img_cv, 112)
        # # cv2.imwrite('test.jpg',img_face)
        # img_face = face_normalize(img_face)
        # img_face = np.transpose(img_face,[2,0,1])
        # print(img_face.shape)
        # img_face = torch.FloatTensor(img_face)
        # img_face = img_face.unsqueeze(dim=0)
        # probe_feat = infer.execute3(img_face).reshape(1,-1)#[B,F]

        probe_feat = infer.execute(img_cv).reshape(1,-1)
        np.save(pj('/Users/marschen/Ucloud/Data','debug.npy'),probe_feat)




if __name__ == '__main__':
    cand_model = [model_dict.keys()]
    cand_model = reduce( lambda x,y: '%s/%s'%(x,y), cand_model)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',default='baseline5932',help=cand_model)
    parser.add_argument('--data',default='')
    args = parser.parse_args()
    main(args=args)