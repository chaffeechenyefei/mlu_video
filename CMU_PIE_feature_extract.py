"""
Extract feature batch
input:
cmu-pie formatted:
{person_id}_{xxx}_{cam_pose}.jpg
cam_pose: INT
output:
xxx.pkl
    {
    'feat': [N,4+1+2*5] np.array
    'label': [N,] np.array, class_id
    'label-cam_pose':{label-id:cam_pose}
    }
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


def save_obj(obj, name ):
    with open( name, 'wb') as f:
        pickle.dump(obj, f)

def load_obj(name ):
    with open(name , 'rb') as f:
        return pickle.load(f)

# label_to_campose={
# 				0:{'id':[2,25,37],'flip_label':4},
# 				1:{'id':[5],'flip_label':3},
# 				2:{'id':[7,9,27],'flip_label':2},
# 				3:{'id':[29],'flip_label':1},
# 				4:{'id':[11,14,31],'flip_label':0},
# 				5:{'id':[22],'flip_label':6},
# 				6:{'id':[34],'flip_label':5},
# 				}

label_to_campose = {
                0: {'id': [2, 25, 37], 'flip_label': 6, 'name': 'Big Right'},
                1: {'id': [5], 'flip_label': 5, 'name': 'Right'},
                2: {'id': [7], 'flip_label': 2, 'name': 'Frontal-Up'},
                3: {'id': [27], 'flip_label': 3, 'name': 'Frontal'},
                4: {'id': [9], 'flip_label': 4, 'name': 'Frontal-Down'},
                5: {'id': [29], 'flip_label': 1, 'name': 'Left'},
                6: {'id': [11, 14, 31], 'flip_label': 0, 'name': 'Big Left'},

                7: {'id': [22], 'flip_label': 8, 'name': 'Non-Frontal'},
                8: {'id': [34], 'flip_label': 7, 'name': 'Non-Frontal'},
            }


def main(args):
    if torch.cuda.is_available():
        use_device = 'cuda'
        use_cpu = False
    else:
        use_device = 'cpu'
        use_cpu = True

    weights_faceDet = 'weights/face_det/mobilenet0.25_Final.pth'
    weights_faceDet = os.path.abspath(weights_faceDet)
    faceDet = RetinaFaceDet('mobile0.25', weights_faceDet , use_cpu=use_cpu, backbone_location='')

    datapath = args.data
    savepath = args.save

    if os.path.isdir(savepath):
        pass
    else:
        os.mkdir(savepath)

    imglst = [ c for c in os.listdir(pj(datapath)) if c.lower().endswith('.jpg') ]

    db_feature = []
    db_label = []
    scale = 1.3
    with torch.no_grad():
        for cnt,imgname in enumerate(imglst):
            #decode image name
            ximgname = imgname.replace('.jpg','').split('_')
            person_id = ximgname[0]
            cam_pose = int(ximgname[-1])

            if cam_pose > 50:
                #Esp, condition
                continue

            if cnt%10 == 0:
                print('Executing %1.3f...'% ((cnt+1)/len(imglst)) , end='\r')

            img_cv2 = cv2.imread(pj(datapath,imgname))

            img_cv2_det = cv2.resize(img_cv2,fx=args.det_scale,fy=args.det_scale,dsize=None)

            dets = faceDet.execute(img_cv2_det, threshold=args.det_threshold, topk=5000,
                                   keep_topk=500,
                                   nms_threshold=0.7)

            if dets is None:
                continue
            if len(dets) <= 0:
                continue

            max_face_sz = 0
            max_det = None

            for idx, b in enumerate(dets):
                if b[4] < args.det_threshold:
                    continue
                b = list(map(float, b))
                """
                expand bbox and rescale into unscale size.
                """

                pwidth = int((b[2] - b[0]) )
                pheight = int((b[3] - b[1]) )
                pcx = int((b[2] + b[0]) / 2 )
                pcy = int((b[3] + b[1]) / 2 )

                if pwidth*pheight > max_face_sz:
                    max_det = b.copy()
                    max_face_sz = pwidth*pheight

            max_det = np.array(max_det,np.float32).reshape(1,15)
            # max_det (1,10+5) array [x0,y0,x1,y1,score,leyex,leyey,reyex,reyey,nx,ny,lmx,lmy,rmx,rmy]
            #尺度归一化
            # max_det = np.array(max_det).reshape(1,10)
            # max_det[:,:2] = max_det[:,:2] / pwidth
            # max_det[:,1::2] = max_det[:,1::2] / pheight
            # #中心化
            # nx = max_det[0,4]
            # ny = max_det[0,5]
            # max_det[:,:2] = max_det[:,:2] - nx
            # max_det[:,1::2] = max_det[:,1::2] - ny

            db_feature.append(max_det)
            db_label.append(cam_pose)

            # info from retinaface
            # cv2.rectangle(img_raw0, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            # # landms
            # cv2.circle(img_cv2_det, (max_det[5], max_det[6]), 1, (0, 0, 255), 4) #Left Eye
            # # cv2.circle(img_cv2_det, (max_det[7], max_det[8]), 1, (0, 255, 255), 4) #Right Eye
            # cv2.circle(img_cv2_det, (max_det[9], max_det[10]), 1, (0, 255, 0), 4) #Nose
            # # cv2.circle(img_cv2_det, (max_det[11], max_det[12]), 1, (0, 255, 0), 4) # Left Mouth
            # cv2.circle(img_cv2_det, (max_det[13], max_det[14]), 1, (255, 0, 0), 4) #Right Mouth
            # cv2.imwrite('test.jpg',img_cv2_det)
            # break

    # cam_pose_set = list(set(db_label))
    # cam_pose_label = dict(zip(cam_pose_set,range(len(cam_pose_set))))
    # label_cam_pose = dict(zip(range(len(cam_pose_set)),cam_pose_set))
    label_cam_pose = label_to_campose
    cam_pose_label = {}
    for k in label_cam_pose.keys():
        cam_ids = label_cam_pose[k]['id']
        for cam_id in cam_ids:
            cam_pose_label[cam_id] = {
                'label':k,
                'flip_label':label_cam_pose[k]['flip_label']
            }
    print('{label:cam_pose}')
    print(label_cam_pose)


    db_label = [ cam_pose_label[c]['label'] for c in db_label ]


    # end with torch.no_grad():

    """
    save data into separate files each uuid to a .npy
    """
    print('#'*5,'merging and saving')
    db_feature = np.concatenate(db_feature, axis=0)
    db_label = np.array(db_label)

    nCls = len(label_cam_pose)
    for i in range(nCls):
        print('%d-th label = #%d' % (i, (db_label == i).sum()))

    save_obj({'feat':db_feature,'label':db_label,'label_cam_pose':label_cam_pose} , pj(savepath,'cmu_landmark_pose.pkl'))
    print('Done')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',default='')
    parser.add_argument('--save',default='')
    parser.add_argument('--det_scale',default=1.0,type=float)
    parser.add_argument('--det_threshold',default=0.75,type=float)

    args = parser.parse_args()
    main(args=args)