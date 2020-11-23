"""
Extract feature batch
Each subfolder contains images of one person
"""
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





def main(args):
    backbone_type = 'resnet50_irse_mx'
    model_type = model_dict[args.model_name]['weights']
    model_pth = pj(model_dict[args.model_name]['path'],model_type)
    model_pth = os.path.abspath(model_pth)
    batch_size = args.batch_size

    if torch.cuda.is_available():
        use_device = 'cuda'
    else:
        use_device = 'cpu'

    infer = Inference(backbone_type=backbone_type,
                      ckpt_fpath=model_pth,
                      device=use_device)

    datapath = args.data
    savepath = args.save

    if os.path.isdir(savepath):
        pass
    else:
        os.mkdir(savepath)
    savepath = pj(savepath,args.model_name)
    if os.path.isdir(savepath):
        pass
    else:
        os.mkdir(savepath)


    dataset_test = BasicLoaderV2(imgs_dir=datapath,extstr='.jpg',center_crop=True,rsz=224,dsz=112,restrict=False)
    # dataset_test = BasicLoader(imgs_dir=datapath, extstr='.jpg')
    dataloader_test = DataLoader(dataset=dataset_test, num_workers = 4, batch_size=batch_size, shuffle=False)

    db_feature = {}

    with torch.no_grad():
        for cnt,batch in enumerate(dataloader_test):
            # person_name = '34_技术服务部-韩晨红'
            bimgs = batch['image']#B,C,H,W
            # bfdname = batch['name']
            fdname = batch['fdname']

            if cnt%10 == 0:
                print('Executing %1.3f...'% (cnt/len(dataloader_test)),end='\r')
            else:
                pass

            cur_b_size = bimgs.shape[0]

            probe_feat = infer.execute3(bimgs).reshape(cur_b_size,-1)#[B,F]

            #Below is for BasicLoaderV2
            # print(probe_feat.shape)
            for idx,person_name in enumerate(fdname):

                if person_name in db_feature.keys():
                    db_feature[person_name].append(probe_feat[idx,:].reshape(1,-1))
                else:
                    db_feature[person_name] = [probe_feat[idx,:].reshape(1,-1)]

    """
    save data into separate files each person to a .npy
    """
    print('#'*5,'merging and saving')
    for person_name in db_feature.keys():
        feats = db_feature[person_name]
        feats = np.concatenate(feats,axis=0)
        np.save(pj(savepath,'%s.npy'%person_name),feats)
    # save_obj(db_feature,pj(datapath,backbone_type+'_batch_'+model_type))
    print('Done')



if __name__ == '__main__':
    cand_model = [model_dict.keys()]
    cand_model = reduce( lambda x,y: '%s/%s'%(x,y), cand_model)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',default='baseline5932',help=cand_model)
    parser.add_argument('--data',default='')
    parser.add_argument('--save',default='')
    parser.add_argument('--batch_size',default=8,type=int)

    args = parser.parse_args()
    main(args=args)