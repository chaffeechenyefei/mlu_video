import torch
import numpy as np
import cv2
import os
import pickle

pj = os.path.join


from .eval_utils.inference_api import Inference

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

    if torch.cuda.is_available():
        use_device = 'cuda:1'
    else:
        use_device = 'cpu'

    infer = Inference(backbone_type=backbone_type,
                      ckpt_fpath=model_pth,
                      device=use_device)

    datapath = '/data/yefei/data/ucloud_elavator_face/'
    # datapath = '/Users/marschen/Ucloud/Data/image_cache1/'
    person_lst = os.listdir(datapath)
    person_lst = [ n for n in person_lst if os.path.isdir(pj(datapath,n))]

    db_feature = {}

    with torch.no_grad():
        for cnt,person_name in enumerate(person_lst):
            # person_name = '34_技术服务部-韩晨红'
            if cnt%10 == 0:
                print('Executing %s...'%person_name)
            else:
                pass

            file_lst = os.listdir(pj(datapath,person_name))
            probe_lst = [ f for f in file_lst if f.endswith('.jpg') ]

            probe_feats = []
            probe_names = []
            for probe_name in probe_lst:
                try:
                    probe_path = pj(datapath,person_name,probe_name)
                    probe_img = cv2.imread(probe_path)
                    if probe_img is not None:
                        probe_feat = infer.execute(probe_img).reshape(-1, 1)
                        probe_feats.append(probe_feat)
                        probe_names.append(probe_name)
                except:
                    print('%s error'%probe_name)
                    continue


            db_feature[person_name] = {
                'probe':probe_feats,#np
                'name':probe_names,
            }
            # break
            #end for

    save_obj(db_feature,pj(datapath,backbone_type+'_'+model_type))
    print('Done')






if __name__ == '__main__':
    main()