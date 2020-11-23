import os,cv2
import argparse
import numpy as np
import torch

from Pytorch_Retinaface.retina_infer import RetinaFaceOnnxModel
from FaceRecog.face_rec_api import FFEOnnxModel
import json

pj = os.path.join


if __name__ == '__main__':

    det_model_pth = 'weights/face_det/mobilenet_640x460_prepost.onnx'
    facedetNet = RetinaFaceOnnxModel(det_model_pth)

    rec_model_pth = 'weights/face_rec/face_rec_112x112_m7513.onnx'
    facerecNet = FFEOnnxModel(rec_model_pth,use_fp16=False)

    img_full_name = '/Users/marschen/Ucloud/Data/人工智能部-袁亮_0000.jpg'
    img_cv = cv2.imread(img_full_name)

    ddets = facedetNet.execute(img_cv=img_cv)

    max_face = None
    bbox = [0]*15
    max_size = 0
    expand_scale = 1.3
    for b in ddets:
        b = list(map(int,b))

        if b[0] > bbox[0]:
            bbox = b

    if bbox[4]>0:
        x0, y0, x1, y1 = b[:4]
        _w = (x1 - x0)*expand_scale
        _h = (y1 - y0)*expand_scale
        cx = (x0+x1)/2
        cy = (y0+y1)/2

        _x = cx - _w/2
        _y = cy - _h/2

        _w = int(_w)
        _h = int(_h)
        cx = int(cx)
        cy = int(cy)
        print(x0,y0,x1,y1)
        print( int(_x), int(_y), int(_w), int(_h))
        max_face = cv2.getRectSubPix( img_cv, (_w,_h) , (cx,cy))
        cv2.imwrite('test.png', max_face)

        feat_bag = facerecNet.execute(max_face)
        norm_feat = feat_bag['norm_feat']
        feat = feat_bag['feat']
        print(feat.shape,norm_feat.shape)

        feat_json = {'feat': feat.tolist(), 'norm_feat':norm_feat.tolist() }

        f = open('feat.json', 'w', encoding='utf-8')
        json.dump(feat_json,f)
        f.close()


    # direct rec
    img_full_name = '/Users/marschen/Ucloud/Data/rec_pre2.bmp'
    img_cv = cv2.imread(img_full_name)
    feat_bag2 = facerecNet.execute(img_cv)
    norm_feat2 = feat_bag2['norm_feat']
    feat2 = feat_bag2['feat']

    feat_json = {'feat': feat2.tolist(), 'norm_feat': norm_feat2.tolist()}

    f = open('feat2.json', 'w', encoding='utf-8')
    json.dump(feat_json, f)
    f.close()

    similarity = norm_feat2@norm_feat.transpose()
    print(similarity)














