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
import pickle,time,datetime
import argparse
from sklearn.preprocessing import normalize


from FaceRecog.face_rec_api import Inference
from Pytorch_Retinaface.retina_infer import RetinaFaceDet
from DeblurGANv2.predict import Predictor
from udftools.illumination import lightNormalizor
from udftools.functions import save_obj,load_obj,face_format,timer,datetime_verify,filter_time,timestamp_to_timestr

#include 3rd party
import easyocr

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

use_device = 'cuda' if torch.cuda.is_available() else 'cpu'
use_cpu = False if torch.cuda.is_available() else True

torch.backends.cudnn.benchmark=True

def display_time(avg_time:dict):
    print('===Timing===')
    for key in avg_time.keys():
        if len(avg_time[key]) > 0:
            avg_used_time = np.array(avg_time[key], np.float32)
            avg_used_time = avg_used_time.mean()
            avg_time[key] = []
            print('%s: %1.3f secs'%(key,avg_used_time))


special_cam = {
    '10B_6_0':{'det_scale':0.25,'FLG_no_time':True},
    '10B_7_7':{'det_scale':0.25,'FLG_no_time':True},
}


# cam_with_no_time = ['10B_6_0','10B_7_7']#Cam inside this folder will be executed even without timestamp

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='extract_face_in_video')
    ##img source
    parser.add_argument('--vid_path',default='')#This path is folder containing several folders
    parser.add_argument('--vid_ext',default='.mp4')
    parser.add_argument('--vid_head',default='10B',type=str)
    # ##gallery source
    # parser.add_argument('--gallery_path',default='',help='Where to load gallery?') #pkl file that store the feature extracted from gallery image
    ##detection
    parser.add_argument('--det_net', default='mobile0.25', help='Backbone network mobile0.25 or resnet50 for detection') #backbone type
    parser.add_argument('--det_weights',default='weights/face_det/mobilenet0.25_Final.pth') #weights/face_det/mobilenet0.25_Final.pth
    parser.add_argument('--det_backbone',default='')#'Pytorch_Retinaface/weights/mobilenetV1X0.25_pretrain.tar' #pretrained model, useless in this version
    parser.add_argument('--top_k', default=5000, type=int, help='top_k')
    parser.add_argument('--nms_threshold', default=0.7, type=float, help='nms_threshold')
    parser.add_argument('--keep_top_k', default=500, type=int, help='keep_top_k')
    parser.add_argument('--det_threshold', default=0.6, type=float, help='det_threshold') #Threshold for bbox( the prob of containing a face)
    parser.add_argument('--det_scale',default=1.0,type=float,help='scale image for detection')#resize the image for quick detection
    # ##recognition
    # parser.add_argument('--rec_net',default='resnet50_irse_mx') #backbone type
    # parser.add_argument('--rec_weights', default='weights/face_rec/model_5932.pth') #weights/face_rec/model_5932.pth
    # parser.add_argument('--rec_threshold',default=0.4,type=float) #Threshold for recognition.
    ##deblur
    parser.add_argument('--deblur',action='store_true',default=False,help='Whether doing deblur?')
    parser.add_argument('--deblur_weights',default='DeblurGANv2/weights/fpn_mobilenet.h5')
    ##console
    parser.add_argument('--toy',action='store_true',default=False,help='If using toy mode, only 100 images will be calced')
    parser.add_argument('--save_path',default='/Users/marschen/Ucloud/Data/error_analysis/shuyuan_example/shuyan_result/')
    parser.add_argument('--pcnt',default=100,type=int,help='Print process per ? images')
    parser.add_argument('--face_sz',default=224,type=int,help='Size of face to be saved')
    parser.add_argument('--face_scale',default=2.6,type=float,help='112x112 with bbox*{1.3}')
    args = parser.parse_args()

    cam_with_no_time = [ k for k in special_cam.keys() if special_cam[k]['FLG_no_time'] ]

    if args.toy:
        print('--> It is toy mode!!!')

    print('--> Loading face detection model')
    args.det_weights = os.path.abspath(args.det_weights)
    faceDet = RetinaFaceDet(args.det_net, args.det_weights, use_cpu=use_cpu,backbone_location=args.det_backbone)

    print('--> Loading ocr model')
    ocrNet = easyocr.Reader(['en'], gpu= True)

    print('--> Checking save path')
    if os.path.exists(args.save_path):
        print('--> %s exist'%args.save_path)
    else:
        print('--> creating %s' % args.save_path)
        os.mkdir(args.save_path)

    # print('--> Loading face recognition model')
    # args.rec_weights = os.path.abspath(args.rec_weights)
    # faceRec = Inference(backbone_type=args.rec_net, ckpt_fpath=args.rec_weights, device=use_device)

    """
    check new folders in vid
    """
    cand_lst = [ c for c in os.listdir(args.vid_path) if os.path.isdir(pj(args.vid_path,c)) and not c.endswith('_face') ]

    past_lst = []
    log_path = pj(args.vid_path,'log.txt')
    if os.path.exists(log_path):
        with open(log_path,'r') as f:
            past_lst = f.readlines()
        past_lst = [ c.replace('\n','') for c in past_lst ]
    else:
        pass

    cand_lst = list( set(cand_lst) - set(past_lst) )

    print('='*40)
    print('Total %d folders'%len(cand_lst))
    for c in cand_lst:
        print(c)
    print('=' * 40)


    for folder in cand_lst:
        print('#'*40)
        print('# FOLDER: %s'%folder)
        print('#' * 40)
        vid_path = pj(args.vid_path,folder)

        vid_lst = [ c for c in os.listdir(vid_path) if c.endswith(args.vid_ext) and c.startswith(args.vid_head)]
        print('--> Total %d videos inside'%len(vid_lst))

        # transfer fold timestamp into timestr
        folder_time = timestamp_to_timestr(timestamp=int(folder))
        print('--> folder timestamp %s transferred to %s' % (folder, folder_time))

        save_path = pj(args.save_path,folder)
        if os.path.exists(save_path):
            pass
        else:
            os.mkdir(save_path)

        if args.deblur:
            print('--> Loading deblur model')
            args.deblur_weights = os.path.abspath(args.deblur_weights)
            deblurNet = Predictor(weights_path=args.deblur_weights)

        # print('--> Get light normlizor')
        # lightCls = lightNormalizor(do_scale=True)

        # num_img_processed = 0
        # print('--> Total #%d images to be search'%len(imglst))
        for vid_file in vid_lst:
            print('#'*20)
            print('## executing %s...'%vid_file)
            print('#' * 20)
            vid_fullname = pj( vid_path, vid_file )
            #create fold for saving result.
            result_fd = vid_file.split(args.vid_ext)[0]
            if os.path.isdir(pj(save_path,result_fd)):
                pass
            else:
                os.mkdir(pj(save_path,result_fd))

            tob = timer(display=False)
            # ============Read from Video=========================
            with torch.no_grad():
                # ============Read from Video=========================
                if os.path.isfile(vid_fullname):
                    cap = cv2.VideoCapture(vid_fullname)
                    if cap.isOpened():
                        pass
                    else:
                        print('## this video is broken and skipped')
                        continue
                    vid_fps = cap.get(cv2.CAP_PROP_FPS)
                    print('##FPS:%d'%vid_fps)
                    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    print('##Total Frames:%d'%total_frames)
                    if vid_fps > 60:
                        print('##FPS is too big !! Will skip this video')
                        continue
                    if total_frames > 3600*vid_fps*3:
                        print('##Total Frames more than 2 hours!! Will skip this video')
                        continue


                    pcnt = 0
                    total_face_dets = 0
                    cont_FLG = True
                    skip_frames = 10
                    cnt_skiped = 0

                    skip_x_head_frames = 10

                    ocr_rec_step = vid_fps*10
                    ocr_rec_cnt = 0
                    last_ocr_time = None
                    last_ocr_frame = 0

                    if result_fd in cam_with_no_time:
                        last_ocr_time = folder_time

                    det_scale = args.det_scale if result_fd not in special_cam.keys() else special_cam[result_fd][
                        'det_scale']

                    avg_time = {'reading':[],
                                'illumination':[],
                                'detection':[],
                                'post_detection':[],
                                'ocr': []
                                }
                    while True:
                        if pcnt % args.pcnt == 0:
                            print('#%1.2f processed...and %d face detected' % ((1.0*pcnt)/total_frames,total_face_dets))
                            display_time(avg_time=avg_time)
                        if args.toy and pcnt > 100:
                            print('--> toy mode finished!')
                            break

                        if pcnt >= total_frames:
                            break

                        tob.start()
                        ret,frame = cap.read()
                        tim_read = tob.eclapse()
                        avg_time['reading'].append(tim_read)

                        pcnt += 1
                        ocr_rec_cnt+=1

                        if frame is None:
                            continue

                        if pcnt < skip_x_head_frames:
                            """
                            skip fisrt x frames
                            """
                            continue

                        if not cont_FLG:
                            """
                            skip x frames for acceleration
                            """
                            cnt_skiped += 1
                            if cnt_skiped >= skip_frames:
                                cnt_skiped = 0
                                cont_FLG = True
                            else:
                                continue

                        img_cv2_unscale = frame.copy()
                        if img_cv2_unscale is None:
                            print('Err: frame %d not exist!' % pcnt)
                            continue
                        if len(img_cv2_unscale.shape) < 3:
                            img_cv2_unscale = cv2.cvtColor(img_cv2_unscale, cv2.COLOR_GRAY2BGR)

                        """
                        get time of image
                        """
                        if pcnt > vid_fps*20 and last_ocr_time is None:
                            print('Error! Time cannot be get. Check the position of time texture.')
                            break

                        tim_FLG = True
                        if result_fd in cam_with_no_time:
                            tim_FLG = False
                        else:
                            if last_ocr_time is None or ocr_rec_cnt > ocr_rec_step:
                                tob.start()
                                text_img = img_cv2_unscale[:50,-500:]
                                text_result = ocrNet.readtext(text_img,detail=0)
                                tim_ocr = tob.eclapse()
                                avg_time['ocr'].append(tim_ocr)

                                if len(text_result) > 0:
                                    text_time = filter_time(text_result[0])
                                    tim_FLG = datetime_verify(text_time)
                                else:
                                    tim_FLG = False

                                ocr_rec_cnt = 0
                            else:
                                tim_FLG = False

                        if tim_FLG:
                            framename = text_time.replace(' ','#')
                            framename = 'frame#%d#%s'%(pcnt,framename)
                            last_ocr_time = text_time
                            last_ocr_frame = pcnt
                        else:
                            if last_ocr_time:
                                eclapse_seconds = int((pcnt-last_ocr_frame)/vid_fps)
                                dt = datetime.datetime.strptime(last_ocr_time,"%Y-%m-%d %H:%M:%S")
                                dt = dt + datetime.timedelta(seconds=eclapse_seconds)
                                cur_pred_time = dt.strftime("%Y-%m-%d %H:%M:%S")
                                framename = cur_pred_time.replace(' ','#')
                                framename = 'frame#%d#%s'%(pcnt,framename)
                            else:
                                # last_ocr_time is missing
                                continue




                        # img_cv2 is used for detection only
                        img_cv2 = cv2.resize(img_cv2_unscale, dsize=None, fx=det_scale, fy=det_scale)
                        # preprocessing for illumination of image
                        tob.start()
                        # img_cv2 = lightCls.run(img_cv2)
                        # img_cv2_unscale = lightCls.apply_gamma(img_cv2_unscale)
                        tim_illu = tob.eclapse()
                        avg_time['illumination'].append(tim_illu)

                        img_disp = img_cv2_unscale.copy()

                        faceDet.set_default_size(img_cv2.shape)

                        tob.start()
                        dets = faceDet.execute(img_cv2, threshold=args.det_threshold, topk=args.top_k,
                                               keep_topk=args.keep_top_k,
                                               nms_threshold=args.nms_threshold)
                        tim_det = tob.eclapse()
                        avg_time['detection'].append(tim_det)

                        if dets is None:
                            cont_FLG = False
                            continue
                        if len(dets) == 0:
                            cont_FLG = False
                            continue

                        tob.start()
                        img_faces = []  # BGR raw image
                        batch_faces = []  # [B,C,H,W] format
                        for idx, b in enumerate(dets):
                            if b[4] < args.det_threshold:
                                continue
                            b = list(map(int, b))
                            """
                            expand bbox and rescale into unscale size.
                            """
                            scale = args.face_scale
                            pwidth = int((b[2] - b[0]) * scale / det_scale)
                            pheight = int((b[3] - b[1]) * scale / det_scale)
                            pcx = int((b[2] + b[0]) / 2 / det_scale)
                            pcy = int((b[3] + b[1]) / 2 / det_scale)
                            img_face = cv2.getRectSubPix(img_cv2_unscale, (pwidth, pheight), (pcx, pcy))
                            img_face = face_format(img_face, args.face_sz)

                            """
                            deblur
                            """
                            if args.deblur:
                                img_face = cv2.cvtColor(img_face, cv2.BGR2RGB)
                                img_face = deblurNet(img_face, None)
                                img_face = cv2.cvtColor(img_face, cv2.RGB2BGR)

                            img_faces.append(img_face)

                        total_face_dets += len(img_faces)

                        for ith,img_face in enumerate(img_faces):
                            imgname = '%s#%d#.jpg' % (framename,ith)
                            cv2.imwrite(pj(save_path, result_fd,imgname), img_face)
                        tim_post = tob.eclapse()
                        avg_time['post_detection'].append(tim_post)


            print('#' * 20)
            print('End')


        print('#'*40)
        print('# FOLDER: %s FINISHED'%folder)
        print('#' * 40)

        with open(log_path,'a') as f:
            f.write('%s\n'%folder)


    print('#'*20,'ALL END','#'*20)