import sys,os
pj = os.path.join
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
sys.path.append(curPath)
sys.path.append(pj(curPath,'FaceRecog'))
sys.path.append(pj(curPath,'Pytorch_Retinaface'))
sys.path.append(pj(curPath,'DeblurGANv2'))


import torch
import numpy as np
import cv2,math
import pickle
import argparse
from sklearn.preprocessing import normalize


from FaceRecog.face_rec_api import Inference
from Pytorch_Retinaface.retina_infer import RetinaFaceDet,RetinaFaceDetONNX
from DeblurGANv2.predict import Predictor
from udftools.illumination import lightNormalizor
from udftools.functions import save_obj,load_obj,face_format,face_normalize,largest_indices

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

use_device = 'cuda' if torch.cuda.is_available() else 'cpu'
use_cpu = False if torch.cuda.is_available() else True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='framework001')
    ##img source
    parser.add_argument('--image_path',default='/Users/marschen/Ucloud/Data/demos',help='image folder for detection and recognition')
    parser.add_argument('--ext',default='.jpg') #what kind of image should be read in.
    parser.add_argument('--vid_path',default='')#if video is input, then image_path will be ignored.
    ##gallery source
    parser.add_argument('--gallery_path',default='',help='Where to load gallery?') #pkl file that store the feature extracted from gallery image
    ##detection
    parser.add_argument('--det_net', default='mobile0.25', help='Backbone network mobile0.25 or resnet50 for detection') #backbone type
    parser.add_argument('--detonnx_weights',default='weights/face_det/mobilenet_640x460.onnx') #weights/face_det/mobilenet0.25_Final.pth
    parser.add_argument('--det_weights',
                        default='weights/face_det/mobilenet0.25_Final.pth')  # weights/face_det/mobilenet0.25_Final.pth
    parser.add_argument('--det_backbone',default='')#'Pytorch_Retinaface/weights/mobilenetV1X0.25_pretrain.tar' #pretrained model, useless in this version
    parser.add_argument('--top_k', default=5000, type=int, help='top_k')
    parser.add_argument('--nms_threshold', default=0.7, type=float, help='nms_threshold')
    parser.add_argument('--keep_top_k', default=500, type=int, help='keep_top_k')
    parser.add_argument('--det_threshold', default=0.6, type=float, help='det_threshold') #Threshold for bbox( the prob of containing a face)
    parser.add_argument('--det_scale',default=1.0,type=float,help='scale image for detection')#resize the image for quick detection
    ##recognition
    parser.add_argument('--rec_net',default='resnet50_irse_mx') #backbone type
    parser.add_argument('--rec_weights', default='weights/face_rec/model_5932.pth') #weights/face_rec/model_5932.pth
    parser.add_argument('--rec_threshold',default=0.4,type=float) #Threshold for recognition.
    ##deblur
    parser.add_argument('--deblur',action='store_true',default=False,help='Whether doing deblur?')
    parser.add_argument('--deblur_weights',default='weights/deblur_gan/fpn_mobilenet.h5')
    ##console
    parser.add_argument('--toy',action='store_true',default=False,help='If using toy mode, only 100 images will be calced')
    parser.add_argument('--draw',action='store_true',default=False,help='Print info on image?')
    parser.add_argument('--save_path',default='/Users/marschen/Ucloud/Data/demos/result')
    parser.add_argument('--pcnt',default=100,type=int,help='Print process per ? images')
    parser.add_argument('--disp_sz',default=100,type=int,help='Size of the captured image to be shown') #captured face image will be appended on the bottom of origin surveillance image.
    args = parser.parse_args()

    print('--> Loading face detection model')
    args.det_weights = os.path.abspath(args.det_weights)
    faceDetONNX = RetinaFaceDetONNX(args.det_net, args.detonnx_weights, use_cpu=use_cpu,backbone_location=args.det_backbone)
    faceDet = RetinaFaceDet(args.det_net, args.det_weights, use_cpu=use_cpu,
                                    backbone_location=args.det_backbone)

    print('--> Get light normlizor')
    lightCls = lightNormalizor(do_scale=True)

    with torch.no_grad():
        imgfullname = '/Users/marschen/Ucloud/Data/demos/face_image.jpg'
        img_cv2_unscale = cv2.imread(imgfullname)
        if img_cv2_unscale is None:
            print('Err: %s not exist!'%imgfullname)
            exit(-1)
        if len(img_cv2_unscale.shape) < 3:
            img_cv2_unscale = cv2.cvtColor(img_cv2_unscale,cv2.COLOR_GRAY2BGR)

        img_cv2 = cv2.resize(img_cv2_unscale,(640,460))
        print(img_cv2.shape)
        fx = 640/img_cv2_unscale.shape[1]
        fy = 460/img_cv2_unscale.shape[0]

        # img_cv2 = lightCls.run(img_cv2)
        # img_cv2_unscale = lightCls.apply_gamma(img_cv2_unscale)

        img_disp = img_cv2_unscale.copy()

        faceDet.set_default_size(img_cv2.shape)

        detsonnx = faceDetONNX.execute(img_cv2, threshold=args.det_threshold, topk=args.top_k, keep_topk=args.keep_top_k,
                               nms_threshold=args.nms_threshold)
        dets = faceDet.execute(img_cv2, threshold=args.det_threshold, topk=args.top_k, keep_topk=args.keep_top_k,
                               nms_threshold=args.nms_threshold)

        for i,det in enumerate(dets):
            detonnx = detsonnx[i]
            print(det)
            print(detonnx)

        img_faces = []  # BGR raw image
        batch_faces = []  # [B,C,H,W] format
        # fx = 1
        # fy = 1
        for idx, b in enumerate(detsonnx):
            if b[4] < args.det_threshold:
                continue
            b = list(map(int, b))
            """
            expand bbox and rescale into unscale size.
            """
            scale = 1.3
            pwidth = int((b[2] - b[0]) * scale/fx)
            pheight = int((b[3] - b[1]) * scale/fy)
            pcx = int((b[2] + b[0]) / 2 / fx)
            pcy = int((b[3] + b[1]) / 2 / fy)
            # img_face = cv2.getRectSubPix(img_cv2, (pwidth, pheight), (pcx, pcy))
            img_face = cv2.getRectSubPix(img_cv2_unscale, (pwidth, pheight), (pcx, pcy))
            img_face = face_format(img_face, 112)

            img_faces.append(img_face)
        """
        TODO: draw and show
        BGmtx , img_faces , img_disp
        """
        disp_faces = []
        topk = 1
        nB = len(img_faces)
        disp_sz = args.disp_sz

        for i in range(nB):
            probe_face = img_faces[i]

            probe_face = cv2.resize(probe_face, (disp_sz, disp_sz))
            disp_faces.append(probe_face)

        """
        drawing very dirt code
        """
        n_disp = len(disp_faces)
        h, w, _ = img_disp.shape
        nx = math.floor(w / disp_sz)
        ny = math.ceil(n_disp / nx)

        img_res = np.zeros((h + ny * disp_sz, w, 3), np.uint8)
        img_res[:h, :w] = img_disp

        for j in range(ny):
            for i in range(nx):
                offset = j * ny + i
                if offset < n_disp:
                    sh = h + j * disp_sz
                    sw = 0 + i * disp_sz
                    img_res[sh:sh + disp_sz, sw:sw + disp_sz] = disp_faces[offset]

        cv2.imwrite(pj(args.save_path, 'test.jpg'), img_res)
