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

from cfg import model_dict

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

use_device = 'cuda' if torch.cuda.is_available() else 'cpu'
use_cpu = False if torch.cuda.is_available() else True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='framework001')
    ##img source
    parser.add_argument('--vid_path',default='rtsp://admin:8848chaffee@192.168.183.63:554/h264/ch1/main/av_stream')#if video is input, then image_path will be ignored.
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
    # parser.add_argument('--rec_net',default='resnet50_irse_mx') #backbone type
    # parser.add_argument('--rec_weights', default='weights/face_rec/model_5932.pth') #weights/face_rec/model_5932.pth
    parser.add_argument('--rec_net_name', default='multihead7513')
    parser.add_argument('--rec_threshold',default=0.4,type=float) #Threshold for recognition.
    ##console
    parser.add_argument('--toy',action='store_true',default=False,help='If using toy mode, only 100 images will be calced')
    parser.add_argument('--save_path',default='/Users/marschen/Ucloud/Data/demos/result')
    parser.add_argument('--pcnt',default=100,type=int,help='Print process per ? images')
    parser.add_argument('--disp_sz',default=100,type=int,help='Size of the captured image to be shown') #captured face image will be appended on the bottom of origin surveillance image.
    args = parser.parse_args()

    imglst = []

    if args.toy:
        print('--> It is toy mode!!!')

    print('--> Loading face detection model')
    args.det_weights = os.path.abspath(args.det_weights)
    faceDet = RetinaFaceDet(args.det_net, args.det_weights, use_cpu=use_cpu,backbone_location=args.det_backbone)

    # print('--> Loading face recognition model')
    # backbone_type = 'resnet50_irse_mx'
    # model_type = model_dict[args.rec_net_name]['weights']
    # model_pth = pj(model_dict[args.rec_net_name]['path'],model_type)
    # model_pth = os.path.abspath(model_pth)
    # faceRec = Inference(backbone_type=backbone_type, ckpt_fpath=model_pth, device=use_device)


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

    # print('--> Get light normlizor')
    # lightCls = lightNormalizor(do_scale=True)

    cv2.namedWindow('demo')

    # num_img_processed = 0
    print('--> Total #%d images to be search'%len(imglst))
    with torch.no_grad():
        # ============Read from RSTP=========================
        print('Opening from %s'%args.vid_path)
        cap = cv2.VideoCapture(args.vid_path)
        pcnt = 0
        total_face_dets = 0

        if cap.isOpened():
            pass
        else:
            print('## this video is broken and skipped')
            exit(-1)
        vid_fps = cap.get(cv2.CAP_PROP_FPS)
        print('##FPS:%d' % vid_fps)
        # total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        # print('##Total Frames:%d' % total_frames)
        if vid_fps > 60:
            print('##FPS is too big !! Will skip this video')
            exit(-1)

        FLG_skip = True
        fps_scale = 10
        cnt_skip = 0
        while (cap.isOpened()):
            if pcnt % args.pcnt == 0:
                print('#%d processed...and %d face detected' % (pcnt,total_face_dets),end='\r')
            if args.toy and pcnt > 100:
                print('--> toy mode finished!')
                break

            ret,frame = cap.read()
            pcnt += 1

            if FLG_skip:
                cnt_skip += 1
                if cnt_skip > fps_scale:
                    FLG_skip = False
                    cnt_skip = 0
                continue

            if frame is None:
                print('Err: frame %d not exist!' % pcnt)
                continue

            img_cv2_unscale = frame.copy()
            if len(img_cv2_unscale.shape) < 3:
                img_cv2_unscale = cv2.cvtColor(img_cv2_unscale, cv2.COLOR_GRAY2BGR)

            # img_cv2 is used for detection only
            img_cv2 = cv2.resize(img_cv2_unscale, dsize=None, fx=args.det_scale, fy=args.det_scale)

            # img_disp = img_cv2_unscale.copy()
            # img_disp = cv2.resize(img_cv2_unscale,dsize=None,fx=0.35,fy=0.35)

            faceDet.set_default_size(img_cv2.shape)

            dets = faceDet.execute(img_cv2, threshold=args.det_threshold, topk=args.top_k,
                                   keep_topk=args.keep_top_k,
                                   nms_threshold=args.nms_threshold)




            if dets is None:
                FLG_skip = True
            if len(dets) == 0:
                FLG_skip = True
            if FLG_skip:
                cv2.imshow('demo', img_cv2)
                if cv2.waitKey(20) & 0xFF == ord('q'):
                    break
                continue

            for b in dets:
                text = "face: {:.4f}".format(b[4])
                b = list(map(int, b))
                cv2.rectangle(img_cv2, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cx = b[0]
                cy = b[1] + 12
                cv2.putText(img_cv2, text, (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                # landms
                cv2.circle(img_cv2, (b[5], b[6]), 2, (0, 0, 255), 4)
                cv2.circle(img_cv2, (b[7], b[8]), 2, (0, 255, 255), 4)
                cv2.circle(img_cv2, (b[9], b[10]), 2, (255, 0, 255), 4)
                cv2.circle(img_cv2, (b[11], b[12]), 2, (0, 255, 0), 4)
                cv2.circle(img_cv2, (b[13], b[14]), 2, (255, 0, 0), 4)

            cv2.imshow('demo',img_cv2)

            if cv2.waitKey(20) & 0xFF == ord('q'):
                break

            # img_faces = []  # BGR raw image
            # batch_faces = []  # [B,C,H,W] format
            # for idx, b in enumerate(dets):
            #     if b[4] < args.det_threshold:
            #         continue
            #     b = list(map(int, b))
            #     """
            #     expand bbox and rescale into unscale size.
            #     """
            #     scale = 1.3
            #     pwidth = int((b[2] - b[0]) * scale / args.det_scale)
            #     pheight = int((b[3] - b[1]) * scale / args.det_scale)
            #     pcx = int((b[2] + b[0]) / 2 / args.det_scale)
            #     pcy = int((b[3] + b[1]) / 2 / args.det_scale)
            #     img_face = cv2.getRectSubPix(img_cv2_unscale, (pwidth, pheight), (pcx, pcy))
            #     img_face = face_format(img_face, 112)
            #
            #     img_faces.append(img_face)
            #
            # total_face_dets += len(img_faces)
            #
            # """
            # wait for batch feature extraction
            # """
            # for img_face in img_faces:
            #     img_data = face_normalize(img_face)
            #     img_data = np.transpose(img_data, axes=[2, 0, 1])  # [C,H,W]
            #     img_data = torch.FloatTensor(img_data)
            #     batch_faces.append(img_data)
            #
            # batch_faces = torch.stack(batch_faces, dim=0)  # [B,C,H,W]
            # batch_feat = faceRec.execute_batch_unit(batch_faces)  # [B,D]
            #
            # """
            # cosine distance:
            # since all the features are unit normed, it is calculated directly by mat multi.
            # """
            # BGmtx = batch_feat @ gfeat
            # """
            # TODO: draw and show
            # BGmtx , img_faces , img_disp
            # """
            # topnames = []
            # disp_faces = []
            # topk = 1
            # nB = BGmtx.shape[0]
            # disp_sz = args.disp_sz
            #
            # for i in range(nB):
            #     topid = largest_indices(BGmtx[i, :], topk)
            #     topsim = BGmtx[i, topid[0]].reshape(topk, 1)
            #
            #     topname = gname[topid[0].item()]
            #     topnames.append(topname)
            #
            #     gallery_face = gallery_db[topname]['image']
            #     probe_face = img_faces[i]
            #
            #     probe_face = cv2.resize(probe_face, (disp_sz, disp_sz))
            #     gallery_face = cv2.resize(gallery_face, (disp_sz, disp_sz))
            #     disp_face = np.concatenate([probe_face, gallery_face], axis=0)
            #     text = "{:.4f}".format(topsim[0].item())
            #     if topsim > args.rec_threshold:
            #         cv_show_color = (255, 0, 0)
            #     else:
            #         cv_show_color = (0, 0, 255)
            #
            #     cv2.putText(disp_face, text, (0, 12),
            #                 cv2.FONT_HERSHEY_DUPLEX, 0.5, cv_show_color)
            #     disp_faces.append(disp_face)
            #
            # """
            # drawing, very dirt code
            # """
            # n_disp = len(disp_faces)
            # h, w, _ = img_disp.shape
            # nx = math.floor(w / disp_sz)
            # ny = math.ceil(n_disp / nx)
            #
            # img_res = np.zeros((h + ny * 2 * disp_sz, w, 3), np.uint8)
            # img_res[:h, :w] = img_disp
            #
            # for j in range(ny):
            #     for i in range(nx):
            #         offset = j * ny + i
            #         if offset < n_disp:
            #             sh = h + j * 2 * disp_sz
            #             sw = 0 + i * disp_sz
            #             img_res[sh:sh + 2 * disp_sz, sw:sw + disp_sz] = disp_faces[offset]
            #
            # imgname = 'frame_#%06d.jpg'%pcnt
            # cv2.imwrite(pj(args.save_path, imgname), img_res)


    cap.release()
    print('='*5,'END','='*5)