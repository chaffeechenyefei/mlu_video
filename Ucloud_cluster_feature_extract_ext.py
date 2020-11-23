"""
Extract feature batch ext
input:
Ucloud_unsupervised_cluster 
person_name = {Dep}-{Name}
data/{timestamp}/{cam}/{ith-cluster}/XXX.jpg 
data/{timestamp}/{cam}/{ith-cluster}/top5/{score}_{person_name}.jpg
data/{timestamp}/{cam}/{ith-cluster}/top5/{???_folder}/{score}_{person_name}.jpg
output:
save/{timestamp}_{cam}_{ith_cluster}.pkl 
    dict:{
        'label':{person_name},
        'feat':[K,D], #K is the number of images in {timestamp}/{ith-cluster},
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
from PoseNet import model as PoseNetTools
from Pytorch_Retinaface.retina_infer import RetinaFaceDet

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



    timestamp_folders = [ c for c in os.listdir(datapath) if os.path.isdir(pj(datapath,c)) ]

    with torch.no_grad():
        for cnt_timestamp_folder,timestamp_folder in enumerate(timestamp_folders):
            print('#'*5,'Executing %s (%d/%d)'%(timestamp_folder,cnt_timestamp_folder+1,len(timestamp_folders)))
            timestamp_folder_full = pj(datapath,timestamp_folder)

            # sub_savepath = pj(savepath,timestamp_folder)
            # if not os.path.exists(sub_savepath):
            #     os.mkdir(sub_savepath)

            cam_folders = [ c for c in os.listdir(timestamp_folder_full) if os.path.isdir(pj(timestamp_folder_full,c)) ]

            for cnt_cam_folder,cam_folder in enumerate(cam_folders):
                print('*' * 2, 'Executing %s (%d/%d)' % (cam_folder, cnt_cam_folder + 1, len(cam_folders)))
                cam_folder_full = pj(timestamp_folder_full,cam_folder)

                # sub_cam_savepath = pj(sub_savepath,cam_folder)
                # if not os.path.exists(sub_cam_savepath):
                #     os.mkdir(sub_cam_savepath)

                cluster_folders = [ c for c in os.listdir(cam_folder_full) if os.path.isdir(pj(cam_folder_full,c)) ]
                for cnt_cluster_folder,cluster_folder in enumerate(cluster_folders):
                    print('*' * 1,'Executing %s (%d/%d)' % (cluster_folder, cnt_cluster_folder + 1, len(cluster_folders)) , end='\r')
                    cluster_folder_full = pj(cam_folder_full,cluster_folder)

                    #get name/label from top5 folder
                    label_folder_full = pj(cluster_folder_full,'top5')
                    assert os.path.exists(label_folder_full), 'Err: folder named top5 is missing <%s>.'%cluster_folder_full
                    pos_folder = [ c for c in os.listdir(label_folder_full) if os.path.isdir(pj(label_folder_full,c)) ]
                    assert len(pos_folder) <= 1, 'Err: too many folders in %s'%label_folder_full

                    if len(pos_folder) == 0:
                        print('Skipped:%s'%cluster_folder_full)
                        continue
                    else:
                        pos_folder_full = pj(label_folder_full,pos_folder[0])
                        neg_labels = [ c.replace('.jpg','') for c in os.listdir(label_folder_full) if c.endswith('.jpg')]
                        pos_labels = [ c.replace('.jpg','') for c in os.listdir(pos_folder_full) if c.endswith('.jpg')]
                        neg_labels = list(set(neg_labels) - set(pos_labels))

                        if len(pos_labels) > 0:
                            pos_label = pos_labels[0]
                            pos_label = pos_label.split('_')[-1]
                        else:
                            pos_label = None

                        neg_labels = [ c.split('_')[-1] for c in neg_labels ]


                        dataset_test = BasicLoader_ext(imgs_dir=cluster_folder_full, extstr='.jpg',center_crop=True,rsz=224,dsz=112)
                        dataloader_test = DataLoader(dataset=dataset_test, num_workers=4, batch_size=args.batch_size, shuffle=False)

                        db_feature = {}
                        bfeats = []
                        bposefeats = []
                        for cnt_batch,batch in enumerate(dataloader_test):
                            # print('%d/%d...'%(cnt_batch+1,len(dataloader_test)),end='\r')
                            bimgs = batch['image']
                            bnames = batch['name']
                            bdetimgs = batch['det_image']

                            #face feature
                            bfeat = infer.execute3(X=bimgs).reshape(-1,512)
                            bfeats.append(bfeat)

                            # pose feature
                            bdet_feat = []
                            # print('bdet_imgs',bdetimgs.shape)
                            detss = faceDet.execute_batch(bdetimgs, threshold=0.75)
                            for dets in detss:
                                if len(dets) == 0:
                                    max_det = np.zeros((1, 15))
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

                            bdet_feat = np.concatenate(bdet_feat, axis=0)
                            bdet_feat = torch.FloatTensor(bdet_feat).to(_use_device)

                            bpose_feat = poseNet.predict(bdet_feat)
                            bpose_feat = bpose_feat.cpu().numpy()
                            bposefeats.append(bpose_feat)

                        bfeats = np.concatenate(bfeats,axis=0)
                        bposefeats = np.concatenate(bposefeats, axis=0)
                        assert bposefeats.sum()/len(bposefeats) != np.NaN, 'Err nan is found'
                        # print(bposefeats.sum()/len(bposefeats), bposefeats.shape,bdet_feat.sum()/len(bdet_feat))

                        #name decoding/feature storing...
                        db_feature['feat'] = bfeats
                        db_feature['label'] = pos_label
                        db_feature['neg_label'] = neg_labels
                        db_feature['pose_feat'] = bposefeats

                        save_name = '%s_%s_%s.pkl'%(timestamp_folder,cam_folder,cluster_folder)
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