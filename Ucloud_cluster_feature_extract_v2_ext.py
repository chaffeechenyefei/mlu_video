"""
Extract feature batch v2 ext
'pose_feat' comes from poseNet is used.
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
        'pose_feat':[K,D], #D is the output of poseNet
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
from FaceRecog.datasets.plateloader import BasicLoaderV2,BasicLoader_ext


pj = os.path.join

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


from eval_utils.inference_api import Inference
from Pytorch_Retinaface.retina_infer import RetinaFaceDet
from PoseNet import model as PoseNetTools

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
    _use_device = torch.device(use_device)

    infer = Inference(backbone_type=backbone_type,
                      ckpt_fpath=model_pth,
                      device=use_device)

    #loading pose Net
    args.nCls = 9
    if args.pose_model.endswith('_mt'):
        args.nn_layers = [8, 16, 32, args.nCls+1]
    else:
        args.nn_layers = [8, 16, 32, args.nCls]
    poseCls = getattr(PoseNetTools, args.pose_model)
    poseNet = poseCls(nn_layers=args.nn_layers).to(device=_use_device)
    ckpt_path = os.path.abspath(args.pose_weights)
    PoseNetTools.load_model(model=poseNet, pretrained_path=ckpt_path, load_to_cpu=use_cpu)
    poseNet = poseNet.to(_use_device)
    poseNet.eval()

    #detection net
    weights_faceDet = 'weights/face_det/mobilenet0.25_Final.pth'
    weights_faceDet = os.path.abspath(weights_faceDet)
    faceDet = RetinaFaceDet('mobile0.25', weights_faceDet , use_cpu=use_cpu, backbone_location='')


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

            dataset_test = BasicLoader_ext(imgs_dir=person_name_folder_full, extstr='.jpg',center_crop=False,dsz=112)
            dataloader_test = DataLoader(dataset=dataset_test, num_workers=4, batch_size=args.batch_size, shuffle=False)

            db_feature = {}
            bfeats = []
            bposefeats = []
            for cnt_batch,batch in enumerate(dataloader_test):
                print('%d/%d...'%(cnt_batch+1,len(dataloader_test)),end='\r')
                bimgs = batch['image']
                bnames = batch['name']
                bdetimgs = batch['det_image']

                #face feature
                bfeat = infer.execute3(X=bimgs).reshape(-1, 512)
                bfeats.append(bfeat)

                #pose feature
                bdet_feat = []
                # print('bdet_imgs',bdetimgs.shape)
                detss = faceDet.execute_batch(bdetimgs,threshold=0.75)
                for dets in detss:
                    if len(dets) == 0:
                        max_det = np.zeros((1,15))
                    else:
                        max_face_sz = 0
                        max_det = np.zeros((1,15))
                        for idx, b in enumerate(dets):
                            b = list(map(float, b))
                            pwidth = int((b[2] - b[0]))
                            pheight = int((b[3] - b[1]))
                            if pwidth * pheight > max_face_sz:
                                max_det = b.copy()
                                max_face_sz = pwidth * pheight

                        max_det = np.array(max_det, np.float32).reshape(1, 15)
                    bdet_feat.append(max_det)
                    # print('bdet_feat',bdet_feat)

                bdet_feat = np.concatenate(bdet_feat,axis=0)
                bdet_feat = torch.FloatTensor(bdet_feat).to(_use_device)

                bpose_feat = poseNet.predict(bdet_feat)
                bpose_feat = bpose_feat.cpu().numpy()
                bposefeats.append(bpose_feat)

            bfeats = np.concatenate(bfeats, axis=0)
            bposefeats = np.concatenate(bposefeats,axis=0)
            assert bposefeats.sum() / len(bposefeats) != np.NaN, 'Err nan is found'
            #name decoding/feature storing...
            db_feature['feat'] = bfeats
            db_feature['pose_feat'] = bposefeats
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

    parser.add_argument('--pose_model',default='poseNetv3_mt')
    parser.add_argument('--pose_weights',type=str)

    args = parser.parse_args()
    main(args=args)