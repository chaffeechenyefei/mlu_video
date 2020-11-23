import torch
import numpy as np
import cv2
import os
import pickle
from torch.utils.data import DataLoader
from datasets.plateloader import BasicLoaderV2,BasicLoader

pj = os.path.join

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from eval_utils.inference_api import Inference

def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)



def main():
    backbone_type = 'resnet50_irse_mx'
    model_type = 'model_5932.pth'
    model_pth = pj('./model_output/insight_face_res50irsemx_cosface_emore_dist/',model_type)
    batch_size = 64

    if torch.cuda.is_available():
        use_device = 'cuda'
    else:
        use_device = 'cpu'

    infer = Inference(backbone_type=backbone_type,
                      ckpt_fpath=model_pth,
                      device=use_device)

    # datapath = '/data/yefei/data/ucloud_elavator_face/'
    datapath = '/data/yefei/data/0107_0114_result/'

    # dataset_test = BasicLoaderV2(imgs_dir=datapath,extstr='.jpg')
    dataset_test = BasicLoader(imgs_dir=datapath, extstr='.jpg')
    dataloader_test = DataLoader(dataset=dataset_test, num_workers = 4, batch_size=batch_size, shuffle=False)

    db_feature = {}

    with torch.no_grad():
        for cnt,batch in enumerate(dataloader_test):
            # person_name = '34_技术服务部-韩晨红'
            bimgs = batch['image']#B,C,H,W
            bfdname = batch['name']

            if cnt%100 == 0:
                print('Executing %1.3f...'% (cnt/len(dataloader_test)) )
            else:
                pass

            cur_b_size = bimgs.shape[0]

            probe_feat = infer.execute3(bimgs).reshape(cur_b_size,-1)#[B,F]

            for idx,imgname in enumerate(bfdname):
                db_feature[imgname] = probe_feat[idx].reshape(-1,1)

            #Below is for BasicLoaderV2
            # print(probe_feat.shape)
            # for idx,person_name in enumerate(bfdname):
            #
            #     if person_name in db_feature.keys():
            #         db_feature[person_name]['probe'].append(probe_feat[idx,:].reshape(-1,1))
            #     else:
            #         db_feature[person_name] = {'probe':[probe_feat[idx,:].reshape(-1,1)]}



    save_obj(db_feature,pj(datapath,backbone_type+'_batch_'+model_type))
    print('Done')






if __name__ == '__main__':
    main()