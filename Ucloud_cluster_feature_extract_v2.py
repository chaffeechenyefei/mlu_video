"""
Extract feature batch v2 
input:
Ucloud_elavator_face
person_name = {Dep}-{Name}
data/{person_name}/XXX.jpg 
output:
save/{person_name}.pkl 
    dict:{
        'label':{person_name},
        'feat':[K,D], #K is the number of images in {person_name},
        'neg_label':list[{person_name}],
        # 'label_feat': [1,D],
        # 'neg_label_feat':[N,D],
    }
Note: register image
"""

import sys,os
pj = os.path.join
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
sys.path.append(curPath)
sys.path.append(pj(curPath,'FaceRecog'))
sys.path.append(pj(curPath,'Pytorch_Retinaface'))


import torch,argparse
import numpy as np
import cv2
import pickle
from cfg import model_dict
from functools import reduce
from udftools.functions import face_format
from udftools.SCface.functions import decode_image_name
from torch.utils.data import DataLoader
from FaceRecog.datasets.plateloader import BasicLoaderV2,BasicLoader

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

    if torch.cuda.is_available():
        use_device = 'cuda'
        use_cpu = False
    else:
        use_device = 'cpu'
        use_cpu = True

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



    person_name_folders = [ c for c in os.listdir(datapath) if os.path.isdir(pj(datapath,c)) ]

    with torch.no_grad():
        for cnt_person_name_folder ,person_name_folder in enumerate(person_name_folders):
            print('#'*5,'Executing %s (%d/%d)'%(person_name_folder,cnt_person_name_folder+1,len(person_name_folders)))
            person_name_folder_full = pj(datapath,person_name_folder)

            pos_label = person_name_folder

            dataset_test = BasicLoader(imgs_dir=person_name_folder_full, extstr='.jpg',center_crop=False,dsz=112)
            dataloader_test = DataLoader(dataset=dataset_test, num_workers=4, batch_size=args.batch_size, shuffle=False)

            db_feature = {}
            bfeats = []
            for cnt_batch,batch in enumerate(dataloader_test):
                print('%d/%d...'%(cnt_batch+1,len(dataloader_test)),end='\r')
                bimgs = batch['image'] #[B,C,H,W]
                bnames = batch['name']

                if args.ILA:
                    B = bimgs.shape[0]
                    _bimgs = []

                    rL = max([B-args.nILA+1,1])

                    for k in range(rL):
                        sub_imgs = bimgs[k:k+args.nILA]
                        _bimgs.append(sub_imgs.mean(dim=0))
                    _bimgs = torch.stack(_bimgs,dim=0)
                    bimgs = _bimgs
                else:
                    pass

                bfeat = infer.execute3(X=bimgs).reshape(-1, 512)
                bfeats.append(bfeat)

            bfeats = np.concatenate(bfeats, axis=0)

            #name decoding/feature storing...
            db_feature['feat'] = bfeats
            db_feature['label'] = pos_label
            db_feature['neg_label'] = []

            save_name = '%s.pkl'%person_name_folder
            save_obj(db_feature,pj(savepath,'%s'%save_name))




    print('='*10)
    print('End')
    print('='*10)









if __name__ == '__main__':
    cand_model = [model_dict.keys()]
    cand_model = reduce( lambda x,y: '%s/%s'%(x,y), cand_model)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',default='baseline5932',help=cand_model)
    parser.add_argument('--data',default='')
    parser.add_argument('--save',default='')
    parser.add_argument('--batch_size',default=16,type=int)
    #+image level aggregation
    parser.add_argument('--ILA',action='store_true',help='using image level aggregation or not')
    parser.add_argument('--nILA',default=5,type=int)

    args = parser.parse_args()
    main(args=args)