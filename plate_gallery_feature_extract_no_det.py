"""
Extract feature batch
input:
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
from Pytorch_Retinaface.retina_infer import RetinaFaceDet
from cfg import model_dict
from functools import reduce
from udftools.functions import face_format

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

    # weights_faceDet = 'weights/face_det/mobilenet0.25_Final.pth'
    # weights_faceDet = os.path.abspath(weights_faceDet)
    # faceDet = RetinaFaceDet('mobile0.25', weights_faceDet , use_cpu=use_cpu, backbone_location='')

    datapath = args.data
    savepath = args.save

    if os.path.isdir(savepath):
        pass
    else:
        os.mkdir(savepath)


    imglst = [ c for c in os.listdir(pj(datapath)) if c.lower().endswith(args.ext) ]

    db_feature = []
    db_label = []
    scale = 1.3
    with torch.no_grad():
        for cnt,imgname in enumerate(imglst):
            #decode image name
            person_id = imgname

            if cnt%10 == 0:
                print('Executing %1.3f...'% ((cnt+1)/len(imglst)) , end='\r')

            img_cv2 = cv2.imread(pj(datapath,imgname))

            if len(img_cv2.shape) == 2:
                img_cv2 = cv2.cvtColor(img_cv2,cv2.COLOR_GRAY2BGR)

            probe_feat = infer.execute(img_cv2).reshape(1,-1)#[B,F]

            db_feature.append(probe_feat)
            db_label.append(person_id)


    # end with torch.no_grad():

    """
    save data into separate files each uuid to a .npy
    """
    print('#'*5,'merging and saving')
    db_feature = np.concatenate(db_feature,axis=0)
    db_label = np.array(db_label)
    save_obj({'X':db_feature,'y':db_label} , pj(savepath,'%s_%s.pkl'%(args.prefix,args.model_name)))
    print('Done')



if __name__ == '__main__':
    cand_model = [model_dict.keys()]
    cand_model = reduce( lambda x,y: '%s/%s'%(x,y), cand_model)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',default='baseline5932',help=cand_model)
    parser.add_argument('--data',default='')
    parser.add_argument('--save',default='')
    parser.add_argument('--prefix',default='gallery',help='{}_xxx.pkl')
    parser.add_argument('--ext',default='.jpg')

    args = parser.parse_args()
    main(args=args)