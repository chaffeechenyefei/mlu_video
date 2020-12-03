import sys                                                                                                                                                                                                           
import os                                                                                                                                                                                                            
                                                                                                                                                                                                                     

from flask import Flask, jsonify, request                                                                                                                                                                            
import json                                                                                                                                                                                                          
import numpy as np                                                                                                                                                                                                   
from sklearn.preprocessing import normalize
from mlu_inference import *
import base64                                                                                                                                                                                                        
import cv2                                                                                                                                                                                                           
import traceback                                                                                                                                                                                                     
from pathlib import Path                                                                                                                                                                                             
from PIL import Image                                                                                                                                                                                                
import torch                                                                                                                                                                                                         app = Flask(__name__)                                                                                                                                                                                                
                                                                                                                                                                                                                     
cv2.useOptimized()                                                                                                                                                                                                   
                                                                                                                                                                                                                     

#detect
if os.environ['MODE'] == 'cpu':
    detect_algo = mlu_face_det_inference(weights='/root/weights/face_det/mobilenet0.25_Final.pth',use_mlu=False,use_jit=True)
else:
    detect_algo = mlu_face_det_inference(weights='/root/weights/face_det/retinaface_mlu_int8.pth', use_mlu=True, use_jit=True)

# cpu 
#cpu_detss = cpu_face_det_model.execute(img_cv2,dst_size=[w,h])
#if len(cpu_detss) > 0:
#    print(cpu_detss[0].shape,cpu_detss[0])
# load model
# mlu detect
#mlu_detss = mlu_face_det_model.execute(img_cv2,dst_size=[w,h])
#    if len(mlu_detss) > 0:
#        print(mlu_detss[0].shape,mlu_detss[0])
#cal 
if os.environ['MODE'] == 'cpu':
    rec_algo = mlu_face_rec_inference(weights='/root/weights/face_rec/r101irse_model_3173.pth',use_mlu=False,use_jit=True)
else:
    rec_algo = mlu_face_rec_inference(weights='/root/weights/face_rec/resnet101_mlu_int8.pth',use_mlu=True,use_jit=True)
                                             
def read64(encoded_data):                                                                                                                                                                                            
   nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)                                                                                                                                                   
   img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)                                                                                                                                                                       
   return img                                               

@app.route('/detect', methods=['POST'])
def detect():
    try:
        import time
        now = time.time()
        img = read64(request.json['img'])
        height, width, _ = img.shape
        print("from b64 spent: ", time.time() - now)
        res = detect_algo.execute(img, dst_size=[width, height])
        print("detect spent: ", time.time() - now)
        if len(res) == 0:
            return jsonify({'code':0, 'data': [jsonify({'bbox':[]})]})
        else:
            pos = res[0]
            scale = 1.3
            new_pos = []
            for b in pos:
                b = list(map(int, b))
                pwidth = int((b[2] - b[0]) * scale)
                pheight = int((b[3] - b[1]) * scale)
                pcx = int((b[2] + b[0]) / 2)
                pcy = int((b[3] + b[1]) / 2)
                new_pos.append([pwidth, pheight, int(pcx-pwidth/2), int(pcy - pheight/2)])
            return jsonify({'code':0, 'data': [json.dumps({'bbox': new_pos})]})
    except Exception as e:
        traceback.print_exc(e)
        return jsonify({'msg':str(traceback.extract_tb(e)), 'code': -1})

@app.route('/rec', methods=['POST'])
def rec():
    try:
        import time
        now = time.time()
        img = request.json['img']
        base64_imgs = [read64(i) for i in img]
        print("from b64 spent: ", time.time() - now)
        res = rec_algo.execute(base64_imgs)
        res = normalize(res, axis=1)
        print("rec spent: ", time.time() - now)
        return jsonify({'code':0, 'data': res.tolist()})
    except Exception as e:
        traceback.print_exc(e)
        return jsonify({'msg':str(traceback.extract_tb(e)), 'code': -1})

if __name__ == '__main__':
    print("server started")
    app.run(threaded=False, processes=1, port=7999, host='0.0.0.0')
