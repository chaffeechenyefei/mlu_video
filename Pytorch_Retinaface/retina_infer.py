from __future__ import print_function
import os,sys
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import torch.nn as nn
import json

# curPath = os.path.abspath(os.path.dirname(__file__))
# sys.path.append(curPath)



from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm, batch_decode_landm, batch_decode
import onnxruntime

pj = os.path.join

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import time
class timer(object):
    def __init__(self,it:str='',display=True):
        self._start = 0
        self._end = 0
        self._name = it
        self._display = display
        pass

    def start(self,it:str=None):
        self._start = time.time()
        if it is not None:
            self._name = it
        else:
            pass

    def end(self):
        self._end = time.time()

    def diff(self):
        tt = self._end-self._start
        return tt

    def eclapse(self):
        self.end()
        tt = self.diff()
        if self._display:
            print('<<%s>> eclapse: %f sec...'%(self._name,tt))
        return tt


def store(data):
    with open('data.json', 'w') as fw:
        # 将字典转化为字符串
        # json_str = json.dumps(data)
        # fw.write(json_str)
        # 上面两句等同于下面这句
        json.dump(data, fw)
        # load json data from file


def load():
    with open('data.json', 'r') as f:
        data = json.load(f)
        return data

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model



class ONNXModel():
    def __init__(self, onnx_path):
        """
        :param onnx_path:
        """
        self.onnx_session = onnxruntime.InferenceSession(onnx_path)
        self.input_name = self.get_input_name(self.onnx_session)
        self.output_name = self.get_output_name(self.onnx_session)
        print("input_name:{}".format(self.input_name))
        print("output_name:{}".format(self.output_name))

    def get_output_name(self, onnx_session):
        """
        output_name = onnx_session.get_outputs()[0].name
        :param onnx_session:
        :return:
        """
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_name(self, onnx_session):
        """
        input_name = onnx_session.get_inputs()[0].name
        :param onnx_session:
        :return:
        """
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_input_feed(self, input_name, image_tensor):
        """
        input_feed={self.input_name: image_tensor}
        :param input_name:
        :param image_tensor:
        :return:
        """
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_tensor
        return input_feed

    def forward(self, image_tensor):
        '''
        image_tensor = image.transpose(2, 0, 1)
        image_tensor = image_tensor[np.newaxis, :]
        onnx_session.run([output_name], {input_name: x})
        :param image_tensor:
        :return:
        '''
        # 输入数据的类型必须与模型一致,以下三种写法都是可以的
        # scores, boxes = self.onnx_session.run(None, {self.input_name: image_tensor})
        # scores, boxes = self.onnx_session.run(self.output_name, input_feed={self.input_name: image_tensor})
        input_feed = self.get_input_feed(self.input_name, image_tensor)
        output = self.onnx_session.run(self.output_name, input_feed=input_feed)
        return output

class RetinaFaceDetONNX(object):
    def __init__(self,model_type="mobile0.25",model_path="./weights/mobilenet0.25_Final.pth",
                 backbone_location="./weights/mobilenetV1X0.25_pretrain.tar",use_cpu=True):
        self.cfg = None
        if model_type == "mobile0.25":
            self.cfg = cfg_mnet
        elif model_type == "resnet50":
            self.cfg = cfg_re50

        self.net = ONNXModel(model_path)
        self.device = torch.device("cpu" if use_cpu else "cuda")

        self._priors = None
        self.im_width = 0
        self.im_height = 0
        self.im_nch = 0

    def set_default_size(self,imgshape=[640,480,3]):#[H,W,nCh]
        im_height, im_width, im_nch = imgshape
        if im_height == self.im_height and im_width == self.im_width and self._priors is not None:
            pass
        else:
            self.im_height,self.im_width,self.im_nch = imgshape
            priorbox = PriorBox(self.cfg,image_size=(self.im_height,self.im_width))
            self._priors = priorbox.forward()

    def execute(self,img_cv,threshold=0.6,topk=5000,keep_topk=750,nms_threshold=0.7):
        resize = 1
        with torch.no_grad():
            img = np.float32(img_cv)

            im_height, im_width, _ = img.shape
            scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
            img -= (104, 117, 123)
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).unsqueeze(0)
            img = img.to(self.device)
            scale = scale.to(self.device)

            _img = img.cpu().numpy()
            loc, conf, landms = self.net.forward(_img)  # forward pass
            loc = torch.FloatTensor(loc)
            conf = torch.FloatTensor(conf)
            landms = torch.FloatTensor(landms)

            if im_height == self.im_height and im_width == self.im_width and self._priors is not None:
                pass
            else:
                self.set_default_size([im_height,im_width,3])

            priors = self._priors

            priors = priors.to(self.device)
            prior_data = priors.data
            boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
            boxes = boxes * scale / resize
            boxes = boxes.cpu().numpy()
            scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
            landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg['variance'])
            scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                   img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                   img.shape[3], img.shape[2]])
            scale1 = scale1.to(self.device)
            landms = landms * scale1 / resize
            landms = landms.cpu().numpy()

            # ignore low scores
            inds = np.where(scores > threshold)[0]
            boxes = boxes[inds]
            landms = landms[inds]
            scores = scores[inds]

            # keep top-K before NMS
            order = scores.argsort()[::-1][:topk]
            boxes = boxes[order]
            landms = landms[order]
            scores = scores[order]

            # do NMS
            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            keep = py_cpu_nms(dets, nms_threshold)
            # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
            dets = dets[keep, :]
            landms = landms[keep]

            # keep top-K faster NMS
            dets = dets[:keep_topk, :]
            landms = landms[:keep_topk, :]

            # x0,y0,x1,y1,score,landmarks...
            dets = np.concatenate((dets, landms), axis=1)

            return dets


class RetinaFaceDet(object):
    def __init__(self,model_type="mobile0.25",model_path="./weights/mobilenet0.25_Final.pth",
                 backbone_location="./weights/mobilenetV1X0.25_pretrain.tar",use_cpu=True,loading=True):
        self.cfg = None
        self.use_cpu = use_cpu
        self.model_path = model_path
        if model_type == "mobile0.25":
            self.cfg = cfg_mnet
        elif model_type == "resnet50":
            self.cfg = cfg_re50
        self.device = torch.device("cpu" if use_cpu else "cuda")
        self.net = RetinaFace(cfg=self.cfg,phase="test",backbone_location=backbone_location)
        if loading:
            self.loading()

        self._priors = None
        self.im_width = 0
        self.im_height = 0
        self.im_nch = 0

    def _get_model(self):
        self.net.eval()
        return self.net

    def loading(self):
        self.net = load_model(self.net, self.model_path, self.use_cpu)
        self.net.eval()
        self.net = self.net.to(self.device)

    def set_default_size(self,imgshape=[640,480,3]):#[H,W,nCh]
        im_height, im_width, im_nch = imgshape
        if im_height == self.im_height and im_width == self.im_width and self._priors is not None:
            pass
        else:
            self.im_height,self.im_width,self.im_nch = imgshape
            """
            priorbox shape [-1,4]; dim0: number of predicted bbox from network; dim1:[x_center,y_center,w,h]
            priorbox存储的内容分别是bbox中心点的位置以及人脸预设的最小尺寸，长宽比例通过variance解决
            这里的数值都是相对图像尺寸而言的相对值，取值在(0,1)之间
            """
            priorbox = PriorBox(self.cfg,image_size=(self.im_height,self.im_width))
            self._priors = priorbox.forward()

    def execute_batch(self,img_batch,threshold=0.6,topk=5000,keep_topk=750,nms_threshold=0.7):
        resize = 1
        with torch.no_grad():
            img_batch = img_batch.to(self.device)
            locs,confs,landmss = self.net(img_batch)

            nB,nCh,im_height, im_width = img_batch.shape

            scale = torch.Tensor([im_width, im_height, im_width, im_height])
            scale = scale.to(self.device)

            if im_height == self.im_height and im_width == self.im_width and self._priors is not None:
                pass
            else:
                self.set_default_size([im_height,im_width,nCh])

            priors = self._priors
            priors = priors.to(self.device)
            prior_data = priors.data

            detss = []
            """
            以bbox的location为例子，最终要得到的是：
                bbox_center_x
                bbox_center_y
                bbox_w
                bbox_h
            但是，直接预测这些数值是困难的，所以需要脱离图像的尺寸，压缩到0-1的范围，所以我们改为预测：_bbox_center_x，_bbox_center_y，_bbox_w，_bbox_w，他们的关系如下：
                bbox_center_x = (_bbox_center_x)*imgW
                bbox_center_y = (_bbox_center_y)*imgH
                bbox_w = (_bbox_w)*imgW
                bbox_h = (_bbox_w)*imgH
            进一步，引入anchor的概念，即预先设定多个bbox的中心和最小的人脸长宽。我们只预测真实值与预设值之间的比例、偏移关系，
            模型预测结果为[x_offset,y_offset,w_scale,h_scale]
            预设bbox为[x_center,y_center,face_w,face_h] 即prior_data
            vx,vy控制人脸的长宽比
            他们之间相互关系为：
                _bbox_center_x = x_center + x_offset*face_w*vx
                _bbox_center_y = y_center + y_offset*face_h*vy
                _bbox_w = face_w*exp(w_scale*vx)
                _bbox_h = face_h*exp(h_scale*vy)            
            
            最终得到：
                bbox_center_x = (x_center + x_offset*face_w*vx)*imgW
                bbox_center_y = (y_center + y_offset*face_h*vy)*imgH
                bbox_w = (face_w*exp(w_scale*vx))*imgW
                bbox_h = (face_h*exp(h_scale*vy))*imgH
            """

            for idx in range(nB):
                loc = locs[idx]
                conf = confs[idx]
                landms = landmss[idx]

                """
                对loc而言，网络输出的是shape为[-1,4]的矩阵，dim1是[x_offset,y_offset,w_scale,h_scale],需要通过decode进行恢复到正常的bbox
                loc: [-1,4]; dim1: [x_offset,y_offset,w_scale,h_scale]
                prior_data: [-1,4]; dim1:[x_center,y_center,face_w,face_h] 
                虽然这里的face_w!= face_h,但本质上是相等的，因为是face_w/face_h相对图像尺寸的值。所以，本质上这里是正方形的anchor，需要variance来控制长宽比。
                variance: [vx,vy] 控制长宽比例
                _bbox_center_x = x_center + x_offset*face_w*vx
                _bbox_center_y = y_center + y_offset*face_h*vy
                _bbox_w = face_w*exp(w_scale*vx)
                _bbox_h = face_h*exp(h_scale*vy)
                进一步，转为left top corner x, left top corner y, right bottom corner x, right bottom corner y的形式 
                """
                boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
                """
                之前的结果都是normalize的结果，即(0,1)，因此，需要重新rescale回去。
                这个scale即图像的大小。
                """
                boxes = boxes * scale / resize
                boxes = boxes.cpu().numpy()
                scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
                """
                基本原理同上
                """
                landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg['variance'])
                scale1 = torch.Tensor([im_width, im_height, im_width, im_height,
                                       im_width, im_height, im_width, im_height,
                                       im_width, im_height])
                scale1 = scale1.to(self.device)
                landms = landms * scale1 / resize
                landms = landms.cpu().numpy()

                # ignore low scores
                inds = np.where(scores > threshold)[0]
                boxes = boxes[inds]
                landms = landms[inds]
                scores = scores[inds]

                # keep top-K before NMS
                order = scores.argsort()[::-1][:topk]
                boxes = boxes[order]
                landms = landms[order]
                scores = scores[order]

                # do NMS
                dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
                """
                py_cpu_nms 非极大抑制
                基本原理，假设dets是一个队列，每次取队列第一个元素(pop)，并加入到keep的list中，将该元素与dets队列中其它元素比较，剔除bbox交集大于nms_threshold的元素。
                然后不断循环，直到dets为空。
                """
                keep = py_cpu_nms(dets, nms_threshold)
                # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
                dets = dets[keep, :]
                landms = landms[keep]

                # keep top-K faster NMS
                dets = dets[:keep_topk, :]
                landms = landms[:keep_topk, :]

                # x0,y0,x1,y1,score,landmarks...
                dets = np.concatenate((dets, landms), axis=1)

                detss.append(dets)
        return detss




    def execute(self,img_cv,threshold=0.6,topk=5000,keep_topk=750,nms_threshold=0.7):
        resize = 1
        with torch.no_grad():
            img = np.float32(img_cv)

            im_height, im_width, _ = img.shape
            scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
            img -= (104, 117, 123)
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).unsqueeze(0)
            img = img.to(self.device)
            scale = scale.to(self.device)

            loc, conf, landms = self.net(img)  # forward pass

            if im_height == self.im_height and im_width == self.im_width and self._priors is not None:
                pass
            else:
                self.set_default_size([im_height,im_width,3])

            priors = self._priors

            priors = priors.to(self.device)
            prior_data = priors.data
            boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
            boxes = boxes * scale / resize
            boxes = boxes.cpu().numpy()
            scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
            landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg['variance'])
            scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                   img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                   img.shape[3], img.shape[2]])
            scale1 = scale1.to(self.device)
            landms = landms * scale1 / resize
            landms = landms.cpu().numpy()

            # ignore low scores
            inds = np.where(scores > threshold)[0]
            boxes = boxes[inds]
            landms = landms[inds]
            scores = scores[inds]

            # keep top-K before NMS
            order = scores.argsort()[::-1][:topk]
            boxes = boxes[order]
            landms = landms[order]
            scores = scores[order]

            # do NMS
            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            keep = py_cpu_nms(dets, nms_threshold)
            # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
            dets = dets[keep, :]
            landms = landms[keep]

            # keep top-K faster NMS
            dets = dets[:keep_topk, :]
            landms = landms[:keep_topk, :]

            # x0,y0,x1,y1,score,landmarks...
            dets = np.concatenate((dets, landms), axis=1)

            return dets

    def execute_debug(self,img_cv,threshold=0.6,topk=5000,keep_topk=750,nms_threshold=0.7):
        resize = 1
        dtime = {'detection': [],
                    'nms': [],
                    'decode': [],
                    }
        tob = timer(display=False)
        with torch.no_grad():
            img = np.float32(img_cv)

            im_height, im_width, _ = img.shape
            scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
            # img -= (104, 117, 123)
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).unsqueeze(0)
            img = img.to(self.device)
            scale = scale.to(self.device)

            data = img.reshape(-1).tolist()
            jsd = {
                "data":[
                    {
                        "INPUT":{
                            "content":data,
                            "shape":[460,640]
                        }
                    }
                ]
            }

            # jsdata = json.dumps( jsd )
            store(jsd)

            tob.start()
            loc, conf, landms = self.net(img)  # forward pass
            utm = tob.eclapse()
            dtime['detection'].append(utm)

            if im_height == self.im_height and im_width == self.im_width and self._priors is not None:
                pass
            else:
                self.set_default_size([im_height,im_width,3])

            priors = self._priors

            tob.start()

            priors = priors.to(self.device)
            prior_data = priors.data
            print('nin=', prior_data.shape[0])
            print('loc.data',loc.data.squeeze(0)[:2,:])
            boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
            print('boxes',boxes[:2,:])
            boxes = boxes * scale / resize
            boxes = boxes.cpu().numpy()
            scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
            print('landms.data',landms.data.squeeze(0)[:2,:])
            landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg['variance'])
            print('landms', landms[:2, :])
            scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                   img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                   img.shape[3], img.shape[2]])
            scale1 = scale1.to(self.device)
            landms = landms * scale1 / resize
            landms = landms.cpu().numpy()

            # ignore low scores
            inds = np.where(scores > threshold)[0]
            boxes = boxes[inds]
            landms = landms[inds]
            scores = scores[inds]
            print('nout=', scores.shape[0])


            # keep top-K before NMS
            order = scores.argsort()[::-1][:topk]
            boxes = boxes[order]
            landms = landms[order]
            scores = scores[order]

            utm = tob.eclapse()
            dtime['decode'].append(utm)

            tob.start()
            # do NMS

            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            print('nms in=', dets.shape[0])
            keep = py_cpu_nms(dets, nms_threshold)
            # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
            dets = dets[keep, :]
            print('nms out=', dets.shape[0])
            landms = landms[keep]

            # keep top-K faster NMS
            dets = dets[:keep_topk, :]
            landms = landms[:keep_topk, :]

            # x0,y0,x1,y1,score,landmarks...
            dets = np.concatenate((dets, landms), axis=1)
            utm = tob.eclapse()
            dtime['nms'].append(utm)

            return dets,dtime

class RetinaFaceDetv2(nn.Module):
    def __init__(self,model_type="mobile0.25",model_path="./weights/mobilenet0.25_Final.pth",use_cpu=True):
        super().__init__()
        self.cfg = None
        self.cfg = cfg_mnet
        self.net = RetinaFace(cfg=self.cfg,phase="test")
        self.net = load_model(self.net, model_path, use_cpu)
        self.net.eval()

        self.device = torch.device("cpu" if use_cpu else "cuda")
        self.net = self.net.to(self.device)
        self._priors = None
        self.im_width = 640
        self.im_height = 460
        self.im_nch = 0

        self.set_default_size([self.im_height,self.im_width,3])
        self.scale = torch.Tensor([self.im_width, self.im_height]*2)
        self.scale1 = torch.Tensor([self.im_width, self.im_height] * 5)

    def set_default_size(self,imgshape=[480,640,3]):
        im_height, im_width, im_nch = imgshape
        if im_height == self.im_height and im_width == self.im_width and self._priors is not None:
            pass
        else:
            self.im_height,self.im_width,self.im_nch = imgshape
            """
            priorbox shape [-1,4]; dim0: number of predicted bbox from network; dim1:[x_center,y_center,w,h]
            priorbox存储的内容分别是bbox中心点的位置以及人脸预设的最小尺寸，长宽比例通过variance解决
            这里的数值都是相对图像尺寸而言的相对值，取值在(0,1)之间
            """
            priorbox = PriorBox(self.cfg,image_size=(self.im_height,self.im_width))
            self._priors = priorbox.forward().unsqueeze(dim=0)

    def forward(self,img_batch):
        # threshold = 0.75
        # topk = 5000
        # keep_topk = 750
        # nms_threshold = 0.7
        with torch.no_grad():
            img_batch = img_batch.to(self.device)
            #[B,C,H,W]->[B,H,W,C]
            img_batch = img_batch.permute(0,2,3,1)
            img_batch -= torch.FloatTensor([104, 117, 123]).to(self.device)
            img_batch = img_batch.permute(0,3,1,2)

            locs,confs,landmss = self.net(img_batch)

            # # nB,nCh,_, _ = img_batch.shape
            priors = self._priors
            priors = priors.to(self.device)
            # #
            #
            # priors = torch.ones_like(self._priors).to(self.device)
            boxes = batch_decode(locs, priors, self.cfg['variance'])
            boxes = boxes * self.scale.to(self.device)
            # boxes = boxes.cpu().numpy()
            # scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
            scores = confs[:,:,1]
            # print('inner',scores.shape)
            landms = batch_decode_landm(landmss, priors, self.cfg['variance'])
            scale1 = self.scale1.to(self.device)
            landms = landms * scale1


                # landms = landms.cpu().numpy()

                # ignore low scores
                # inds = torch.where(scores > threshold)[0]
            # inds = scores > threshold
            # boxes = boxes[inds]
            # landms = landms[inds]
            # scores = scores[inds]

                # # keep top-K before NMS
                # order = scores.argsort()[::-1][:topk]
                # boxes = boxes[order]
                # landms = landms[order]
                # scores = scores[order]


        return boxes,scores,landms


class RetinaFaceOnnxModel(object):
    def __init__(self, onnx_weights, threshold=0.8, topk=5000, keep_topk=750, nms_threshold=0.2, H = 460, W=640 ):
        self.threshold = threshold
        self.topk = topk
        self.keep_topk = keep_topk
        self.nms_threshold = nms_threshold
        self.H = H
        self.W = W
        self.detnet = ONNXModel(onnx_weights)
        self.dratio = H / W

    def execute(self,img_cv):
        #resize into [dh,dw,3], but keep the ratio of w/h constant
        h,w = img_cv.shape[:2]
        ratio = h/w
        format_img = np.zeros((self.H,self.W,3),np.uint8)
        if ratio > self.dratio:
            #w should be filled
            _h = self.H
            scale = _h/h
            _w = int(w*scale)
        else:
            #h should be filled
            _w = self.W
            scale = _w/w
            _h = int(h*scale)

        scale_img_cv = cv2.resize(img_cv,dsize=(_w,_h))
        format_img[:_h,:_w,:] = scale_img_cv[:,:,:]

        # convert [h,w,c] to [b,c,h,w] and float32
        format_img = format_img.astype(np.float32)
        format_img = format_img.transpose(2,0,1)
        format_img = format_img[np.newaxis,:,:,:] #[b,c,h,w]

        locs, confs, landmss = self.detnet.forward(format_img)

        # squeeze because only b = 1 :: locs = [b,12200,4], confs = [b,12200], landmss=[b,12200,10]
        locs = locs.squeeze()
        confs = confs.squeeze()
        landmss = landmss.squeeze()

        # post execution which should be done outside the model
        # ignore low scores
        inds = np.where(confs > self.threshold)[0]

        locs = locs[inds]
        landmss = landmss[inds]
        confs = confs[inds]

        # keep top-K before NMS
        order = confs.argsort()[::-1][:self.topk]
        locs = locs[order]
        landmss = landmss[order]
        confs = confs[order]

        # do NMS
        dets = np.hstack((locs, confs[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landmss = landmss[keep]

        # keep top-K faster NMS
        dets = dets[:self.keep_topk, :]
        landmss = landmss[:self.keep_topk, :]

        # x0,y0,x1,y1,score,landmarks...
        dets = np.concatenate((dets, landmss), axis=1)

        dets[:4] = dets[:4]/scale
        dets[5:] = dets[5:]/scale

        return dets



def format_det_img(img_cv):
    img = np.float32(img_cv)
    im_height, im_width, _ = img.shape
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img)
    return img


def display_time(avg_time: dict):
    print('===Timing===')
    for key in avg_time.keys():
        if len(avg_time[key]) > 0:
            avg_used_time = np.array(avg_time[key], np.float32)
            avg_used_time = avg_used_time.mean()
            avg_time[key] = []
            print('%s: %1.3f secs' % (key, avg_used_time))

def _format(img_cv2, format_size=112):
    org_h, org_w = img_cv2.shape[0:2]
    rescale_ratio = format_size / max(org_h, org_w)
    h, w = int(org_h * rescale_ratio), int(org_w * rescale_ratio)
    img_rescaled = cv2.resize(img_cv2, (w, h))
    paste_pos = [int((format_size - w) / 2), int((format_size - h) / 2)]
    img_format = np.zeros((format_size, format_size, 3), dtype=np.uint8)
    img_format[paste_pos[1]:paste_pos[1] + h, paste_pos[0]:paste_pos[0] + w] = img_rescaled
    return img_format


def _normalize_retinaface(img_cv2,mlu=False):
    # img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    # mean = [123.675, 116.28, 103.53]
    # std = [58.395, 57.12, 57.375]
    img_cv2 = cv2.resize(img_cv2,dsize=(512,512))
    img_data = np.asarray(img_cv2, dtype=np.float32)
    if mlu:
        return img_data #[0,255]
    else:
        mean = (104, 117, 123)
        img_data = img_data - mean
        img_data = img_data.astype(np.float32) #[0,1] normalized
    return img_data

def preprocess_retinaface(img_cv2, mlu=False):
    img_format = _format(img_cv2)
    img_data = _normalize_retinaface(img_format,mlu=mlu)
    img_data = np.transpose(img_data, axes=[2, 0, 1])
    img_data = np.expand_dims(img_data, axis=0)
    img_t = torch.from_numpy(img_data)
    return img_t

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retinaface')

    parser.add_argument('-m', '--trained_model', default='../weights/face_det/mobilenet0.25_Final.pth',
                        # mobilenet0.25_Final Resnet50_Final
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
    parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
    parser.add_argument('--top_k', default=5000, type=int, help='top_k')
    parser.add_argument('--nms_threshold', default=0.7, type=float, help='nms_threshold')
    parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
    parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
    parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
    args = parser.parse_args()

    args.cpu = True
    # cudnn.benchmark = True
    faceDet = RetinaFaceDet(args.network,args.trained_model,use_cpu=args.cpu)
    faceDet.set_default_size([512,512,3])
    model = faceDet.net

    with torch.no_grad():
        image_path = "/Users/marschen/Ucloud/Project/git/mlu_videofacerec/Pytorch_Retinaface/5ea952c344c2683545af4566_1.jpg"
        img_cv = cv2.imread(image_path)
        img_cv = preprocess_retinaface(img_cv,mlu=False)

        loc,conf,landms = model(img_cv)

        conf = conf.data.cpu().numpy()
        print(conf.shape)
        s = conf.max()
        print(s)
    # dtimes = {}
    # for imgname in imglst:
    #     imgname = '人工智能部-袁亮_0001.jpg'
    #     imgfullname0 = pj(image_path,imgname)
    #     # imgfullname1 = pj(image_path, imgname1)
    #     img_raw0 = cv2.imread(imgfullname0, cv2.IMREAD_COLOR)
    #     print(img_raw0.shape)
    #     img_raw0 = cv2.resize(img_raw0,(640,460))
    #     print(img_raw0.shape)
        # img_raw00 = format_det_img(img_raw0)
        # img_raw1 = cv2.imread(imgfullname1, cv2.IMREAD_COLOR)
        # img_raw1 = cv2.resize(img_raw1, (1920//2, 1080//2))
        # img_raw11 = format_det_img(img_raw1)

        # img_batch = torch.stack([img_raw00,img_raw11],dim=0)
        # print(img_batch.shape)
        # detss = faceDet.execute_batch(img_batch,threshold=0.6, topk=args.top_k, keep_topk=args.keep_top_k,
        #                        nms_threshold=args.nms_threshold)

        # print(loc.shape,conf.shape,landms.shape)


        # dets,dtime = faceDet.execute_debug(img_raw0, threshold=0.75, topk=args.top_k, keep_topk=args.keep_top_k,
        #                        nms_threshold=args.nms_threshold)

    #     print(dets)
    #
    #     # show image
    #     if args.save_image:
    #         for b in dets:
    #             if b[4] < args.vis_thres:
    #                 continue
    #             text = "{:.4f}".format(b[4])
    #             b = list(map(int, b))
    #             cv2.rectangle(img_raw0, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
    #             cx = b[0]
    #             cy = b[1] + 12
    #             cv2.putText(img_raw0, text, (cx, cy),
    #                         cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
    #
    #             # landms
    #             cv2.circle(img_raw0, (b[5], b[6]), 2, (0, 0, 255), 4)
    #             cv2.circle(img_raw0, (b[7], b[8]), 2, (0, 255, 255), 4)
    #             cv2.circle(img_raw0, (b[9], b[10]), 2, (255, 0, 255), 4)
    #             cv2.circle(img_raw0, (b[11], b[12]), 2, (0, 255, 0), 4)
    #             cv2.circle(img_raw0, (b[13], b[14]), 2, (255, 0, 0), 4)
    #         # save image
    #
    #         savename = pj(image_path, 'result', imgname)
    #         cv2.imwrite(savename, img_raw0)
    #
    #     for k in dtime.keys():
    #         if k in dtimes.keys():
    #             dtimes[k] += dtime[k]
    #         else:
    #             dtimes[k] = dtime[k]
    #
    #     break
    #
    #
    # display_time(dtimes)

