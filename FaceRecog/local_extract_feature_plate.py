import torch
import numpy as np
import cv2
import os
import pickle
from sklearn.preprocessing import normalize

pj = os.path.join

import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)


from .eval_utils.inference_api import Inference

def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)



def main():
    backbone_type = 'resnet50_irse_mx'
    model_type = 'model_2529.pth'
    model_pth = pj('/Users/marschen/Ucloud/Project/FaceRecog/model_output/insight_face_res50irsemx_cosface_emore_dist/',model_type)

    infer = Inference(backbone_type=backbone_type,
                      ckpt_fpath=model_pth,
                      device='cpu')

    # datapath = '/Users/marschen/Ucloud/Data/error_analysis/error_probe/'
    datapath = '/Users/marschen/Ucloud/Data/image_cache1/'
    img_lst = [ c for c in os.listdir(datapath) if c.endswith('.jpg')]

    db_feature = {}

    for img_name in img_lst:
        print('Executing %s...' % img_name)

        probe_path = pj(datapath, img_name)
        probe_img = cv2.imread(probe_path)
        probe_img = cv2.resize(probe_img,(112,112))
        probe_feat = infer.execute(probe_img).reshape(1, -1)
        probe_feat = normalize(probe_feat,axis=1)

        db_feature[img_name] = probe_feat

    save_obj(db_feature,pj(datapath,backbone_type+'_'+model_type))






if __name__ == '__main__':
    main()