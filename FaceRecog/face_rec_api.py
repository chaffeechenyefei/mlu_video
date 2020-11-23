import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import os
import numpy as np
from checkpoint_mgr.checkpoint_mgr import CheckpointMgr
from model_design_v2.model_mgr import FaceNetMgr

from sklearn.preprocessing import normalize

import onnxruntime

class Inference(object):

    def __init__(self,
                 backbone_type='resnet101',
                 ckpt_fpath='/data/output/insight_face_res101_emore_dist_v1/model_1936.pth',
                 device='cuda:0',
                 # device='cpu',
                 ):
        super(Inference, self).__init__()
        self.backbone_type = backbone_type
        self.ckpt_fpath = ckpt_fpath
        self.device = device
        self.model = self._load_model()


    def _load_model(self):

        model = FaceNetMgr(backbone_type=self.backbone_type).get_model()

        if os.path.isdir(self.ckpt_fpath):
            checkpoint_op = CheckpointMgr(ckpt_dir=self.ckpt_fpath)
            checkpoint_op.load_checkpoint(model=model, warm_load=True, map_location=self.device)
        else:
            print('load_ckpt_fpath: {}'.format(self.ckpt_fpath))
            save_dict = torch.load(self.ckpt_fpath, map_location=self.device)
            # save_dict = save_dict['state_dict'] if 'state_dict' in save_dict.keys() else save_dict
            save_dict = {key.replace('module.', ''): val for key, val in save_dict.items()}
            model.load_state_dict(save_dict)
        model.eval()
        if 'cuda' in self.device:
            model.cuda()
        return model

    def _format(self, img_cv2, format_size=112):
        org_h, org_w = img_cv2.shape[0:2]
        rescale_ratio = format_size / max(org_h, org_w)
        h, w = int(org_h * rescale_ratio), int(org_w * rescale_ratio)
        img_rescaled = cv2.resize(img_cv2, (w, h))
        paste_pos = [int((format_size - w)/2), int((format_size - h)/2)]
        img_format = np.zeros((format_size, format_size, 3), dtype=np.uint8)
        img_format[paste_pos[1]:paste_pos[1]+h, paste_pos[0]:paste_pos[0]+w] = img_rescaled
        return img_format


    def _normalize(self, img_cv2):
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

    def execute(self, img_cv2):
        img_format = self._format(img_cv2)
        img_data = self._normalize(img_format)
        img_data = np.transpose(img_data, axes=[2, 0, 1])
        img_data = np.expand_dims(img_data, axis=0)
        # print('img_data: {}'.format(img_data[0][0][0]))
        img_t = torch.from_numpy(img_data)
        if 'cuda' in self.device:
            img_t = img_t.cuda()
        embedd = self.model(img_t)
        # embedd = F.normalize(embedd)
        embedd = embedd.data.cpu().numpy()[0]
        return embedd

    def execute2(self, X):
        # img_format = self._format(img_cv2)
        # img_data = self._normalize(img_format)
        # img_data = np.transpose(img_data, axes=[2, 0, 1])
        # img_data = np.expand_dims(img_data, axis=0)
        # print('img_data: {}'.format(img_data[0][0][0]))
        # img_t = torch.from_numpy(img_data)
        if 'cuda' in self.device:
            X = X.cuda()
        embedd = self.model(X)
        # embedd = F.normalize(embedd)
        # embedd = embedd.data.cpu().numpy()[0]
        return embedd,X

    def execute3(self, X):
        if 'cuda' in self.device:
            X = X.cuda()
        embedd = self.model(X)
        embedd = embedd.data.cpu().numpy()
        return embedd

    def execute_batch_unit(self, X):
        if 'cuda' in self.device:
            X = X.cuda()
        embedd = self.model(X)#[B,D]
        embedd = F.normalize(embedd,dim=1)
        embedd = embedd.data.cpu().numpy()
        return embedd


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

import json
class FFEOnnxModel(object):
    def __init__(self,onnx_weights,use_fp16=False):
        self.ffenet = ONNXModel(onnx_path=onnx_weights)
        self.use_fp16 = use_fp16

    def _format(self, img_cv2, format_size=112):
        org_h, org_w = img_cv2.shape[0:2]
        rescale_ratio = format_size / max(org_h, org_w)
        h, w = int(org_h * rescale_ratio), int(org_w * rescale_ratio)
        img_rescaled = cv2.resize(img_cv2, (w, h))

        # #test code
        # tmp_img = img_rescaled.copy()
        # tmp_img = self._normalize(tmp_img)
        # feat_json = {'image':tmp_img.tolist()}
        # f = open('image.json', 'w', encoding='utf-8')
        # json.dump(feat_json, f)
        # f.close()

        paste_pos = [int((format_size - w)/2), int((format_size - h)/2)]
        img_format = np.zeros((format_size, format_size, 3), dtype=np.uint8)
        img_format[paste_pos[1]:paste_pos[1]+h, paste_pos[0]:paste_pos[0]+w] = img_rescaled
        return img_format


    def _normalize(self, img_cv2):
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

    def execute(self, img_cv):
        img_format = self._format(img_cv)
        img_data = self._normalize(img_format)
        img_data = np.transpose(img_data, axes=[2, 0, 1])
        img_data = np.expand_dims(img_data, axis=0)
        if self.use_fp16:
            img_data = img_data.astype(np.float16)
        feat = self.ffenet.forward(img_data)
        return {'norm_feat':normalize(feat[0],axis=1),
                'feat':feat[0]}




if __name__=='__main__':

    infer = Inference(backbone_type='resnet50_irse_mx',
                      ckpt_fpath='/Users/marschen/Ucloud/Project/FaceRecog/model_output/insight_face_res50irsemx_cosface_emore_dist/model_1240.pth',
                      device='cpu')
    # img_fpath = '../camera_con/test.jpg'
    img_fpath = '../tupu_data/0169_0000.jpg'
    if os.path.exists(img_fpath):
        img_cv2 = cv2.imread(img_fpath)
        embedd = infer.execute(img_cv2)
        print(embedd)
    else:
        print('Empty')



