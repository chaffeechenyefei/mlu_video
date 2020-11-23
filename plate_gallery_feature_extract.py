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


from eval_utils.inference_api import Inference,ONNXModel

def save_obj(obj, name ):
    with open( name, 'wb') as f:
        pickle.dump(obj, f)

def load_obj(name ):
    with open(name , 'rb') as f:
        return pickle.load(f)

def _format(img_cv2, format_size=112):
    org_h, org_w = img_cv2.shape[0:2]
    rescale_ratio = format_size / max(org_h, org_w)
    h, w = int(org_h * rescale_ratio), int(org_w * rescale_ratio)
    img_rescaled = cv2.resize(img_cv2, (w, h))
    paste_pos = [int((format_size - w)/2), int((format_size - h)/2)]
    img_format = np.zeros((format_size, format_size, 3), dtype=np.uint8)
    img_format[paste_pos[1]:paste_pos[1]+h, paste_pos[0]:paste_pos[0]+w] = img_rescaled
    return img_format


def _normalize(img_cv2):
    img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    # mean = [123.675, 116.28, 103.53]
    # std = [58.395, 57.12, 57.375]
    mean = 127.5
    std = 127.5
    img_data = np.asarray(img_cv2, dtype=np.float32)
    img_data = img_data - mean
    img_data = img_data / std
    img_data = img_data.astype(np.float32)
    return img_data

def main(args):


    if torch.cuda.is_available():
        use_device = 'cuda'
        use_cpu = False
    else:
        use_device = 'cpu'
        use_cpu = True

    if not args.use_onnx:
        print('using pytorch')
        backbone_type = 'resnet101_irse_mx'
        model_type = model_dict[args.model_name]['weights']
        model_pth = pj(model_dict[args.model_name]['path'], model_type)
        model_pth = os.path.abspath(model_pth)
        infer = Inference(backbone_type=backbone_type,
                      ckpt_fpath=model_pth,
                      device=use_device)
    else:
        print('using onnx')
        model_pth = args.onnx_pth
        model_pth = os.path.abspath(model_pth)
        infer = ONNXModel(onnx_path=model_pth)

    weights_faceDet = 'weights/face_det/mobilenet0.25_Final.pth'
    weights_faceDet = os.path.abspath(weights_faceDet)
    faceDet = RetinaFaceDet('mobile0.25', weights_faceDet , use_cpu=use_cpu, backbone_location='')

    datapath = args.data
    savepath = args.save

    if os.path.isdir(savepath):
        pass
    else:
        os.mkdir(savepath)

    FLG_nface = False
    if args.nface:
        FLG_nface = True
        if not os.path.isdir(args.nface):
            os.mkdir(args.nface)


    imglst = [ c for c in os.listdir(pj(datapath)) if c.lower().endswith('.jpg') ]

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
            img_cv2 = cv2.resize(img_cv2,dsize=None,fx=args.moisac,fy=args.moisac)

            img_cv2_det = cv2.resize(img_cv2,fx=args.det_scale,fy=args.det_scale,dsize=None)

            dets = faceDet.execute(img_cv2_det, threshold=args.det_threshold, topk=5000,
                                   keep_topk=500,
                                   nms_threshold=0.2)

            if dets is None:
                continue
            if len(dets) <= 0:
                continue

            max_face = None
            max_face_sz = 0
            for idx, b in enumerate(dets):
                if b[4] < args.det_threshold:
                    continue
                b = list(map(int, b))
                """
                expand bbox and rescale into unscale size.
                """

                pwidth = int((b[2] - b[0]) * scale / args.det_scale)
                pheight = int((b[3] - b[1]) * scale / args.det_scale)
                pcx = int((b[2] + b[0]) / 2 / args.det_scale)
                pcy = int((b[3] + b[1]) / 2 / args.det_scale)

                if pwidth > max_face_sz:
                    img_face = cv2.getRectSubPix(img_cv2, (pwidth, pheight), (pcx, pcy))
                    img_face = face_format(img_face, 112)
                    max_face = img_face.copy()
                    max_face_sz = pwidth

            # cv2.imwrite('test.jpg',max_face)
            # break
            if max_face is not None:
                max_face = cv2.resize(max_face, dsize=None, fx=args.moisacx112, fy=args.moisacx112)
                if FLG_nface:
                    cv2.imwrite( pj(args.nface,imgname) , max_face )

                if not args.use_onnx:
                    probe_feat = infer.execute(max_face).reshape(1,-1)#[B,F]
                else:
                    img_format = _format(max_face)
                    img_data = _normalize(img_format)
                    img_data = np.transpose(img_data, axes=[2, 0, 1])
                    img_data = np.expand_dims(img_data, axis=0)
                    probe_feat = infer.forward(img_data)

                db_feature.append(probe_feat)
                db_label.append(person_id)


    # end with torch.no_grad():

    """
    save data into separate files each uuid to a .npy
    """
    print('#'*5,'merging and saving')
    db_feature = np.concatenate(db_feature,axis=0)
    db_label = np.array(db_label)
    save_obj({'X':db_feature,'y':db_label} , pj(savepath,'gallery_%s.pkl'%args.model_name))
    print('Done')



if __name__ == '__main__':
    cand_model = [model_dict.keys()]
    cand_model = reduce( lambda x,y: '%s/%s'%(x,y), cand_model)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',default='compress11757',help=cand_model)
    parser.add_argument('--data',default='')
    parser.add_argument('--save',default='')
    parser.add_argument('--det_scale',default=1.0,type=float)
    parser.add_argument('--det_threshold',default=0.8,type=float)
    parser.add_argument('--nface',default='')
    parser.add_argument('--moisac',default=1.0,type=float)
    parser.add_argument('--moisacx112',default=1.0,type=float)

    parser.add_argument('--use_onnx',action='store_true')
    parser.add_argument('--onnx_pth',default='weights/face_rec/onnx_res50irsemx_compress_model_11757.onnx')

    args = parser.parse_args()
    main(args=args)