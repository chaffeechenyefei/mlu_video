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
from Pytorch_Retinaface.retina_infer import RetinaFaceDet
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
    parser.add_argument('--det_weights',default='weights/face_det/mobilenet0.25_Final.pth') #weights/face_det/mobilenet0.25_Final.pth
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

    if args.vid_path == '':
        imgfolder = args.image_path
        ext = args.ext
        if not os.path.exists(imgfolder):
            print('Err: %s not exist!'%imgfolder)
            exit(-1)
        imglst = os.listdir(imgfolder)
        imglst = [ c for c in imglst if c.endswith(ext) and not c.startswith('.')]
    else:
        imglst = []

    if args.toy:
        print('--> It is toy mode!!!')

    print('--> Loading face detection model')
    args.det_weights = os.path.abspath(args.det_weights)
    faceDet = RetinaFaceDet(args.det_net, args.det_weights, use_cpu=use_cpu,backbone_location=args.det_backbone)

    print('--> Loading face recognition model')
    args.rec_weights = os.path.abspath(args.rec_weights)
    faceRec = Inference(backbone_type=args.rec_net, ckpt_fpath=args.rec_weights, device=use_device)

    if args.deblur:
        print('--> Loading deblur model')
        args.deblur_weights = os.path.abspath(args.deblur_weights)
        deblurNet = Predictor(weights_path=args.deblur_weights)

    print('--> Loading face feature from gallery')
    if args.gallery_path:
        if os.path.isfile(args.gallery_path):
            gallery_path = args.gallery_path.split('.pkl')[0]
            gallery_db = load_obj(gallery_path)
        else:
            print('Err: %s not exist!'%args.gallery_path)
            exit(-1)
    else:
        gallery_db = {'user_name': {'feat':normalize(np.random.random((512,1)),axis=0),'image':np.zeros((112,112,3),np.uint8)} }

    gfeat = [ f['feat'] for ky,f in gallery_db.items() ]
    gfeat = np.concatenate(gfeat,axis=1)#[D,nG]
    gname = list(gallery_db.keys())

    print('--> Get light normlizor')
    lightCls = lightNormalizor(do_scale=True)

    # num_img_processed = 0
    print('--> Total #%d images to be search'%len(imglst))
    with torch.no_grad():
        # ============Read from Video=========================
        if os.path.isfile(args.vid_path):
            cap = cv2.VideoCapture(args.vid_path)
            pcnt = 0
            total_face_dets = 0
            while True:
                if pcnt % args.pcnt == 0:
                    print('#%d processed...and %d face detected' % (pcnt,total_face_dets))
                if args.toy and pcnt > 100:
                    print('--> toy mode finished!')
                    break

                fps_scale = 10
                ret,frame = cap.read()
                pcnt += 1

                img_cv2_unscale = frame.copy()
                if img_cv2_unscale is None:
                    print('Err: frame %d not exist!' % pcnt)
                    continue
                if len(img_cv2_unscale.shape) < 3:
                    img_cv2_unscale = cv2.cvtColor(img_cv2_unscale, cv2.COLOR_GRAY2BGR)

                # img_cv2 is used for detection only
                img_cv2 = cv2.resize(img_cv2_unscale, dsize=None, fx=args.det_scale, fy=args.det_scale)
                # preprocessing for illumination of image
                img_cv2 = lightCls.run(img_cv2)
                img_cv2_unscale = lightCls.apply_gamma(img_cv2_unscale)

                img_disp = img_cv2_unscale.copy()

                faceDet.set_default_size(img_cv2.shape)

                dets = faceDet.execute(img_cv2, threshold=args.det_threshold, topk=args.top_k,
                                       keep_topk=args.keep_top_k,
                                       nms_threshold=args.nms_threshold)

                if dets is None:
                    continue
                if len(dets) == 0:
                    continue

                img_faces = []  # BGR raw image
                batch_faces = []  # [B,C,H,W] format
                for idx, b in enumerate(dets):
                    if b[4] < args.det_threshold:
                        continue
                    b = list(map(int, b))
                    """
                    expand bbox and rescale into unscale size.
                    """
                    scale = 1.3
                    pwidth = int((b[2] - b[0]) * scale / args.det_scale)
                    pheight = int((b[3] - b[1]) * scale / args.det_scale)
                    pcx = int((b[2] + b[0]) / 2 / args.det_scale)
                    pcy = int((b[3] + b[1]) / 2 / args.det_scale)
                    img_face = cv2.getRectSubPix(img_cv2_unscale, (pwidth, pheight), (pcx, pcy))
                    img_face = face_format(img_face, 112)

                    """
                    deblur
                    """
                    if args.deblur:
                        img_face = cv2.cvtColor(img_face, cv2.BGR2RGB)
                        img_face = deblurNet(img_face, None)
                        img_face = cv2.cvtColor(img_face, cv2.RGB2BGR)

                    img_faces.append(img_face)

                total_face_dets += len(img_faces)

                """
                wait for batch feature extraction
                """
                for img_face in img_faces:
                    img_data = face_normalize(img_face)
                    img_data = np.transpose(img_data, axes=[2, 0, 1])  # [C,H,W]
                    img_data = torch.FloatTensor(img_data)
                    batch_faces.append(img_data)

                batch_faces = torch.stack(batch_faces, dim=0)  # [B,C,H,W]
                batch_feat = faceRec.execute_batch_unit(batch_faces)  # [B,D]

                """
                cosine distance:
                since all the features are unit normed, it is calculated directly by mat multi.
                """
                BGmtx = batch_feat @ gfeat
                """
                TODO: draw and show
                BGmtx , img_faces , img_disp
                """
                topnames = []
                disp_faces = []
                topk = 1
                nB = BGmtx.shape[0]
                disp_sz = args.disp_sz

                for i in range(nB):
                    topid = largest_indices(BGmtx[i, :], topk)
                    topsim = BGmtx[i, topid[0]].reshape(topk, 1)

                    topname = gname[topid[0].item()]
                    topnames.append(topname)

                    gallery_face = gallery_db[topname]['image']
                    probe_face = img_faces[i]

                    probe_face = cv2.resize(probe_face, (disp_sz, disp_sz))
                    gallery_face = cv2.resize(gallery_face, (disp_sz, disp_sz))
                    disp_face = np.concatenate([probe_face, gallery_face], axis=0)
                    text = "{:.4f}".format(topsim[0].item())
                    if topsim > args.rec_threshold:
                        cv_show_color = (255, 0, 0)
                    else:
                        cv_show_color = (0, 0, 255)

                    cv2.putText(disp_face, text, (0, 12),
                                cv2.FONT_HERSHEY_DUPLEX, 0.5, cv_show_color)
                    disp_faces.append(disp_face)

                """
                drawing, very dirt code
                """
                n_disp = len(disp_faces)
                h, w, _ = img_disp.shape
                nx = math.floor(w / disp_sz)
                ny = math.ceil(n_disp / nx)

                img_res = np.zeros((h + ny * 2 * disp_sz, w, 3), np.uint8)
                img_res[:h, :w] = img_disp

                for j in range(ny):
                    for i in range(nx):
                        offset = j * ny + i
                        if offset < n_disp:
                            sh = h + j * 2 * disp_sz
                            sw = 0 + i * disp_sz
                            img_res[sh:sh + 2 * disp_sz, sw:sw + disp_sz] = disp_faces[offset]

                imgname = 'frame_#%d.jpg'%pcnt
                cv2.imwrite(pj(args.save_path, imgname), img_res)

        # ============Read from Image Path=========================
        else:
            for pcnt,imgname in enumerate(imglst):
                if pcnt%args.pcnt == 0:
                    print('%1.3f (#%d) processed...'%(pcnt/len(imglst),pcnt))
                if args.toy and pcnt > 100:
                    print('--> toy mode finished!')
                    break

                imgfullname = pj(imgfolder,imgname)
                img_cv2_unscale = cv2.imread(imgfullname)
                if img_cv2_unscale is None:
                    print('Err: %s not exist!'%imgfullname)
                    continue
                if len(img_cv2_unscale.shape) < 3:
                    img_cv2_unscale = cv2.cvtColor(img_cv2_unscale,cv2.COLOR_GRAY2BGR)

                #img_cv2 is used for detection only
                img_cv2 = cv2.resize(img_cv2_unscale,dsize=None,fx=args.det_scale,fy=args.det_scale)
                img_cv2 = lightCls.run(img_cv2)
                img_cv2_unscale = lightCls.apply_gamma(img_cv2_unscale)

                img_disp = img_cv2_unscale.copy()

                faceDet.set_default_size(img_cv2.shape)

                dets = faceDet.execute(img_cv2, threshold=args.det_threshold, topk=args.top_k, keep_topk=args.keep_top_k,
                                       nms_threshold=args.nms_threshold)

                if dets is None:
                    continue
                if len(dets) == 0:
                    continue

                img_faces = []  # BGR raw image
                batch_faces = []  # [B,C,H,W] format
                for idx, b in enumerate(dets):
                    if b[4] < args.det_threshold:
                        continue
                    b = list(map(int, b))
                    """
                    expand bbox and rescale into unscale size.
                    """
                    scale = 1.3
                    pwidth = int((b[2] - b[0]) * scale/args.det_scale)
                    pheight = int((b[3] - b[1]) * scale/args.det_scale)
                    pcx = int((b[2] + b[0]) / 2 / args.det_scale)
                    pcy = int((b[3] + b[1]) / 2 / args.det_scale)
                    img_face = cv2.getRectSubPix(img_cv2_unscale, (pwidth, pheight), (pcx, pcy))
                    img_face = face_format(img_face, 112)

                    """
                    deblur
                    """
                    if args.deblur:
                        img_face = cv2.cvtColor(img_face,cv2.BGR2RGB)
                        img_face = deblurNet(img_face, None)
                        img_face = cv2.cvtColor(img_face,cv2.RGB2BGR)

                    img_faces.append(img_face)

                """
                wait for batch feature extraction
                """
                for img_face in img_faces:
                    img_data = face_normalize(img_face)
                    img_data = np.transpose(img_data, axes=[2, 0, 1])  # [C,H,W]
                    img_data = torch.FloatTensor(img_data)
                    batch_faces.append(img_data)


                batch_faces = torch.stack(batch_faces, dim=0)  # [B,C,H,W]
                batch_feat = faceRec.execute_batch_unit(batch_faces)  # [B,D]

                BGmtx = batch_feat @ gfeat
                """
                TODO: draw and show
                BGmtx , img_faces , img_disp
                """
                topnames = []
                disp_faces = []
                topk = 1
                nB = BGmtx.shape[0]
                disp_sz = args.disp_sz

                for i in range(nB):
                    topid = largest_indices(BGmtx[i, :], topk)
                    topsim = BGmtx[i, topid[0]].reshape(topk, 1)

                    topname = gname[topid[0].item()]
                    topnames.append(topname)

                    gallery_face = gallery_db[topname]['image']
                    probe_face = img_faces[i]

                    probe_face = cv2.resize(probe_face, (disp_sz, disp_sz))
                    gallery_face = cv2.resize(gallery_face, (disp_sz, disp_sz))
                    disp_face = np.concatenate([probe_face, gallery_face], axis=0)
                    text = "{:.4f}".format(topsim[0].item())
                    if topsim > args.rec_threshold:
                        cv_show_color = (255, 0, 0)
                    else:
                        cv_show_color = (0, 0, 255)

                    cv2.putText(disp_face, text, (0, 12),
                                cv2.FONT_HERSHEY_DUPLEX, 0.5, cv_show_color)
                    disp_faces.append(disp_face)

                """
                drawing very dirt code
                """
                n_disp = len(disp_faces)
                h, w, _ = img_disp.shape
                nx = math.floor(w / disp_sz)
                ny = math.ceil(n_disp / nx)

                img_res = np.zeros((h + ny * 2 * disp_sz, w, 3), np.uint8)
                img_res[:h, :w] = img_disp

                for j in range(ny):
                    for i in range(nx):
                        offset = j * ny + i
                        if offset < n_disp:
                            sh = h + j * 2 * disp_sz
                            sw = 0 + i * disp_sz
                            img_res[sh:sh + 2 * disp_sz, sw:sw + disp_sz] = disp_faces[offset]

                cv2.imwrite(pj(args.save_path, imgname), img_res)
