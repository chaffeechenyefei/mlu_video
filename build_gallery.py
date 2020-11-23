import sys,os
pj = os.path.join
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
sys.path.append(curPath)
sys.path.append(pj(curPath,'FaceRecog'))
sys.path.append(pj(curPath,'Pytorch_Retinaface'))


import torch
from torch.utils.data import DataLoader
import numpy as np
import cv2,math
import pickle
import argparse
from sklearn.preprocessing import normalize

from Pytorch_Retinaface.retina_infer import RetinaFaceDet
from Pytorch_Retinaface.dataset.plateloader import BasicLoader_unsized
from FaceRecog.face_rec_api import Inference
from udftools.illumination import lightNormalizor
from udftools.functions import save_obj,load_obj,face_format,face_normalize,largest_indices
from cfg import model_dict

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

use_device = 'cuda' if torch.cuda.is_available() else 'cpu'
use_cpu = False if torch.cuda.is_available() else True

def _format_batch_faces(img_face):
    img_data = face_normalize(img_face)
    img_data = np.transpose(img_data, axes=[2, 0, 1])  # [C,H,W]
    img_data = torch.FloatTensor(img_data)
    return img_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='framework001')
    ##img source
    parser.add_argument('--image_path',default='/Users/marschen/Ucloud/Data/',help='image folder for detection and recognition')
    parser.add_argument('--ext',default='.jpg')
    parser.add_argument('--sz',default=300,type=int)
    ##detection
    parser.add_argument('--det_net', default='mobile0.25', help='Backbone network mobile0.25 or resnet50 for detection')
    parser.add_argument('--det_weights',default='weights/face_det/mobilenet0.25_Final.pth')
    parser.add_argument('--det_backbone',default='')
    parser.add_argument('--top_k', default=5000, type=int, help='top_k')
    parser.add_argument('--nms_threshold', default=0.7, type=float, help='nms_threshold')
    parser.add_argument('--keep_top_k', default=500, type=int, help='keep_top_k')
    parser.add_argument('--det_threshold', default=0.6, type=float, help='det_threshold')
    ##recognition
    parser.add_argument('--rec_net_name',default='multihead7513')
    # parser.add_argument('--rec_net',default='resnet50_irse_mx')
    # parser.add_argument('--rec_weights', default='weights/face_rec/model_5932.pth')
    parser.add_argument('--rec_threshold',default=0.4,type=float)
    ##console
    parser.add_argument('--draw',action='store_true',default=False,help='Print info on image?')
    parser.add_argument('--save_path',default='/Users/marschen/Ucloud/Data/Data/')
    parser.add_argument('--pcnt',default=100,type=int,help='Print process per ? batches')
    parser.add_argument('--batch_size',default=2,type=int)

    args = parser.parse_args()


    imgfolder = args.image_path
    ext = args.ext
    if not os.path.exists(imgfolder):
        print('Err: %s not exist!'%imgfolder)
        exit(-1)

    dataset_test = BasicLoader_unsized(imgs_dir=imgfolder,extstr='.jpg',sz=args.sz)
    dataloader_test = DataLoader(dataset=dataset_test, num_workers = 4, batch_size=args.batch_size, shuffle=False)

    print('--> Loading face detection model')
    args.det_weights = os.path.abspath(args.det_weights)
    args.det_backbone = os.path.abspath(args.det_backbone)
    faceDet = RetinaFaceDet(args.det_net, args.det_weights, use_cpu=use_cpu,backbone_location=args.det_backbone)

    print('--> Loading face recognition model')
    backbone_type = 'resnet50_irse_mx'
    model_type = model_dict[args.rec_net_name]['weights']
    model_pth = pj(model_dict[args.rec_net_name]['path'],model_type)
    model_pth = os.path.abspath(model_pth)
    faceRec = Inference(backbone_type=backbone_type, ckpt_fpath=model_pth, device=use_device)
    # args.rec_weights = os.path.abspath(args.rec_weights)
    # faceRec = Inference(backbone_type=args.rec_net, ckpt_fpath=args.rec_weights, device=use_device)

    gallery_db = {}

    num_img_done  = 0
    with torch.no_grad():
        for cnt,batch in enumerate(dataloader_test):
            if cnt%args.pcnt == 0:
                print('%1.3f (#%d) processed...'%(cnt/len(dataloader_test),num_img_done),end='\r')


            img_data = batch['image']
            img_names = batch['name']
            img_ori = batch['src']

            detss = faceDet.execute_batch(img_data, threshold=args.det_threshold, topk=args.top_k, keep_topk=args.keep_top_k,
                                          nms_threshold=args.nms_threshold)

            batch_gallery_faces = []
            batch_gallery_names = []

            """
            get max faces from current batch
            """
            for bnt, dets in enumerate(detss):
                img_raw = img_ori[bnt].cpu().numpy()
                img_raw = img_raw.transpose(1,2,0)#[H,W,C]

                max_img_face = None
                max_face_sz = 0
                for idx, b in enumerate(dets):
                    if b[4] < args.det_threshold:
                        continue
                    b = list(map(int, b))
                    scale = 1.3
                    pwidth = int((b[2] - b[0]) * scale)
                    pheight = int((b[3] - b[1]) * scale)
                    pcx = int((b[2] + b[0]) / 2)
                    pcy = int((b[3] + b[1]) / 2)

                    if pwidth*pheight > max_face_sz:
                        max_face_sz = pwidth*pheight
                        img_face = cv2.getRectSubPix(img_raw, (pwidth, pheight), (pcx, pcy))
                        max_img_face = face_format(img_face, 112)

                cv2.imwrite('test.jpg', max_img_face)

                if max_img_face is not None:
                    batch_gallery_faces.append(max_img_face)
                    batch_gallery_names.append(img_names[bnt])
                else:
                    print('%s not face detected'%img_names[bnt])

            """
            batch feature extraction
            """
            batch_gallery_input = list(map( lambda x: _format_batch_faces(x) , batch_gallery_faces))#[C,H,W]
            batch_gallery_input = torch.stack(batch_gallery_input, dim=0)  # [B,C,H,W]
            batch_gallery_feat = faceRec.execute_batch_unit(batch_gallery_input)  # [B,D]

            for i,name in enumerate(batch_gallery_names):
                gallery_db[name] = {
                    'feat':batch_gallery_feat[i].reshape(-1,1),
                    'image':batch_gallery_faces[i],
                }
            num_img_done += len(img_names)
            # break
            # gallery_db = {'user_name': {'feat':normalize(np.random.random((512,1)),axis=0),'image':np.zeros((112,112,3),np.uint8)} }

    gallery_db_name = 'framework_%s' % (args.rec_net_name)
    print('--> saving gallery db into %s'%pj(args.save_path,gallery_db_name))
    save_obj(gallery_db,pj(args.save_path,gallery_db_name))

    print('Done')








