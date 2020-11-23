"""
extract features inside each folder and cluster inside
{time}/{cam}/{images} : 1596592801/10B_6_3/*.jpg
feature is stored in {time}.pkl
{cam}.pkl: dict:{'image_name':np.array()}

cluster:
X = [B,D] Y = [B,1(str)]
X = X.unit(dim=1)
sklearn.cluster.AgglomerativeClustering
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',default='multihead7513')
    parser.add_argument('--batch_size',type=int,default=32)
    parser.add_argument('--data')
    parser.add_argument('--save')
    parser.add_argument('--restart',action='store_true',help='If true, all the features will be extracted and refresh the extracted ones in savepath')

    args = parser.parse_args()


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

    past_lst = [c for c in os.listdir(savepath) if c.endswith('.pkl')]
    past_lst = [c.replace('.pkl','') for c in past_lst]

    fd_times = [c for c in os.listdir(datapath) if os.path.isdir(pj(datapath, c))]
    if not args.restart:
        fd_times = list(set(fd_times) - set(past_lst))

    for cnt_fd_time,fd_time in enumerate(fd_times):
        print('#' * 10, 'executing %s (%d/%d)' % (fd_time, cnt_fd_time+1, len(fd_times)), '#' * 10)
        fd_time_full = pj(datapath,fd_time)

        fd_cams = [ c for c in os.listdir(fd_time_full) if os.path.isdir(pj(fd_time_full,c)) ]
        # print(fd_cams)

        db_feat = []
        db_imgpath = []
        db_attr = []



        for cnt_fd_cam,fd_cam in enumerate(fd_cams):
            print('#' * 5, 'executing %s (%d/%d)' % (fd_cam, cnt_fd_cam+1, len(fd_cams)), '#' * 5)
            fd_cam_full = pj(fd_time_full,fd_cam)

            """
            Basic Loader
            """
            dataset_test = BasicLoader(imgs_dir=fd_cam_full, extstr='.jpg',center_crop=True,rsz=224,dsz=112)
            dataloader_test = DataLoader(dataset=dataset_test, num_workers=4, batch_size=batch_size, shuffle=False)

            with torch.no_grad():
                for cnt, batch in enumerate(dataloader_test):
                    # person_name = '34_技术服务部-韩晨红'
                    bimgs = batch['image']  # B,C,H,W
                    bfdname = batch['imgpath']
                    # fdname = batch['fdname']

                    if cnt % 10 == 0:
                        print('%1.3f...' % (cnt / len(dataloader_test)),end='\r')
                    else:
                        pass

                    cur_b_size = bimgs.shape[0]

                    probe_feat = infer.execute3(bimgs).reshape(cur_b_size, -1)  # [B,D]
                    db_feat.append(probe_feat)
                    db_imgpath = db_imgpath + bfdname
                    db_attr = db_attr + [fd_cam]*cur_b_size

        """
        Doing clustering or just saving
        """
        if len(db_feat) > 0:
            db_feat = np.concatenate(db_feat, axis=0)  # [B,D]
            print('\n %d loaded' % db_feat.shape[0])
            print( '#' * 10, 'saving %s (%d/%d)' % (fd_time, cnt_fd_time+1, len(fd_times)), '#' * 10)
            save_obj([db_feat,db_imgpath,db_attr],name=pj(savepath,'%s.pkl'%fd_time))
