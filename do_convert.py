import sys,os
pj = os.path.join
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
sys.path.append(curPath)
sys.path.append(pj(curPath,'FaceRecog'))
sys.path.append(pj(curPath,'Pytorch_Retinaface'))

from FaceRecog.deploy import convert_to_onnx as fr_convert_to_onnx
from Pytorch_Retinaface import convert_to_onnx as det_convert_to_onnx



print('--> converting onnx for FaceRecog')
backbone_type = 'resnet50_irse_mx'
# save_dir = 'FaceRecog/model_output/insight_face_res50irsemx_cosface_emore_dist'
# ckpt_fpath = 'FaceRecog/model_output/insight_face_res50irsemx_cosface_emore_dist/model_5932.pth'
# model_onnx_path='FaceRecog/model_output/insight_face_res50irsemx_cosface_emore_dist_112x112_m5932.onnx'

save_dir = 'weights/face_rec/'
ckpt_fpath = 'weights/face_rec/model_7513.pth'
model_onnx_path='weights/face_rec/face_rec_112x112_m7513.onnx'

save_dir = os.path.abspath(save_dir)
ckpt_fpath = os.path.abspath(ckpt_fpath)
model_onnx_path = os.path.abspath(model_onnx_path)

fr_convert_to_onnx.main(backbone_type,save_dir,ckpt_fpath,model_onnx_path)

# print('--> check oonx for FaceRecog')

# print('--> converting onnx for Pytorch Retinaface')
# network = "mobile0.25"
# imgW = 640
# imgH = 460
# # trained_model = 'Pytorch_Retinaface/weights/mobilenet0.25_Final.pth'
# # output_onnx = 'Pytorch_Retinaface/weights/mobilenet_%dx%d.onnx'%(imgW,imgH)
# trained_model = 'weights/face_det/mobilenet0.25_Final.pth'
# output_onnx = 'weights/face_det/mobilenet_%dx%d.onnx'%(imgW,imgH)
#
# trained_model = os.path.abspath(trained_model)
# output_onnx = os.path.abspath(output_onnx)
# det_convert_to_onnx.main(network,trained_model,output_onnx,imgH=imgH,imgW=imgW,use_cpu=True)


##check
print('--> check oonx for FaceRecog')
# fp16_model_onnx_path = os.path.abspath('weights/face_rec/fp16_face_rec_112x112_m7513.onnx')
# import onnx
# onnx_model = onnx.load(fp16_model_onnx_path)
# onnx.checker.check_model(onnx_model)

# print('Done')


import onnxruntime
import numpy as np
import torch
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

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


import onnxmltools
from onnxmltools.utils.float16_converter import convert_float_to_float16

# fp16_model_onnx_path = os.path.abspath('weights/face_rec/fp16_face_rec_112x112_m7513.onnx')
# onnx_model = onnxmltools.utils.load_model(model_onnx_path)
# onnx_model = convert_float_to_float16(onnx_model)
# onnxmltools.utils.save_model(onnx_model, fp16_model_onnx_path)
#
rec_onnx = ONNXModel(model_onnx_path)
B = 5
sz = 112

rand_Input = np.random.random((B,3,sz,sz)).astype(np.float32)
x_onnx = rec_onnx.forward(rand_Input)

from FaceRecog.face_rec_api import Inference
#
use_device = 'cuda' if torch.cuda.is_available() else 'cpu'
use_cpu = False if torch.cuda.is_available() else True
device = torch.device(use_device)

faceRec = Inference(backbone_type='resnet50_irse_mx', ckpt_fpath=ckpt_fpath, device=use_device)
x_pt = faceRec.execute3(torch.FloatTensor(rand_Input.astype(np.float32)))
print(x_pt.shape)
print(x_onnx[0].shape)
print('--> difference:%f'%((x_onnx[0] - x_pt)**2).sum().item())


##check
print('--> check oonx for RetinaFace')
# fp16_output_onnx = os.path.abspath('weights/face_det/fp16_mobilenet_640x460.onnx')
# import onnx
# onnx_model = onnx.load(fp16_output_onnx)
# onnx.checker.check_model(onnx_model)
# print('Done')

# onnx_model = onnxmltools.utils.load_model(output_onnx)
# onnx_model = convert_float_to_float16(onnx_model)
# onnxmltools.utils.save_model(onnx_model, fp16_output_onnx)


# det_onnx = ONNXModel(output_onnx)
# B = 5
# w = 640
# h = 460
#
# rand_Input = np.random.random((B,3,h,w)).astype(np.float32)
# x_onnx = det_onnx.forward(rand_Input)
# locs,confs,landmss = x_onnx
#
# print(locs.shape,confs.shape,landmss.shape)
#
# from Pytorch_Retinaface.retina_infer import RetinaFaceDet
# faceDet = RetinaFaceDet('mobile0.25', trained_model, use_cpu=use_cpu,backbone_location=None)
# rand_Input = torch.FloatTensor(rand_Input).to(device)
# _locs,_confs,_landmss = faceDet.net(rand_Input)
#
# _locs = _locs.cpu().numpy()
# _confs = _confs.cpu().numpy()
# _landmss = _landmss.cpu().numpy()
#
# print(_locs.shape,_confs.shape,_landmss.shape)
# print('--> loc difference:%f'%((locs - _locs)**2).sum().item())
# print('--> confs difference:%f'%((confs - _confs)**2).sum().item())
# print('--> landmss difference:%f'%((landmss - _landmss)**2).sum().item())

print('Done')
