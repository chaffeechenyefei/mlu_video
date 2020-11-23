import torch
import torch.onnx as onnx
import os
from model_design_v2 import resnet
from model_design_v2.model_mgr import FaceNetMgr
from checkpoint_mgr.checkpoint_mgr import CheckpointMgr
import numpy as np

def load_model(backbone_type='resnet_face101',
               save_dir='/data/output/insight_face_resface101_emore_dist_v1',
               ckpt_fpath=None):
    model = FaceNetMgr(backbone_type=backbone_type).get_model()
    # load weights
    checkpoint_op = CheckpointMgr(ckpt_dir=save_dir)
    checkpoint_op.load_checkpoint(model=model,
                                  ckpt_fpath=ckpt_fpath,
                                  warm_load=False,
                                  map_location='cpu')
    model.eval()
    return model


def output_to_onnx(model, model_onnx_path='insight_face_resface101_size128_emore_v1.onnx'):
    print('n_params: {}'.format(len([p for p in model.parameters()])))
    n_params = sum(p.nelement() for p in model.parameters())
    print('n_vals: {}'.format(n_params))
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 112, 112)
        input_names = ['input']
        output_names = ['output']
        output = onnx.export(model=model,
                             args=dummy_input,
                             f=model_onnx_path,
                             verbose=False,
                             opset_version=11,
                             input_names=input_names, output_names=output_names,
                             dynamic_axes={'input' : {0 : 'batch_size'},
                                    'output' : {0 : 'batch_size'}}
                             )


def output_alin():
    import cv2
    import numpy as np

    model = resnet.resnet50()
    model.eval()

    save_dir = '/data/output/insight_face_resnet50_v2'
    checkpoint_op = CheckpointMgr(ckpt_dir=save_dir)
    checkpoint_op.load_checkpoint(model=model, map_location='cpu')
    img_fpath = './test.jpg'
    img_cv2 = cv2.imread(img_fpath)
    print(img_cv2.shape)
    img_data = np.asarray(img_cv2, dtype=np.float32)
    img_data = np.transpose(img_data, (2, 0, 1))
    img_data = np.expand_dims(img_data, axis=0)
    img_t = torch.from_numpy(img_data)
    output = model(img_t)
    print(output.size())
    feature = output.data.numpy()[0]
    with open('feature.txt', 'w') as f:
        lines = [str(val) + '\n' for val in feature]
        f.writelines(lines)


def main(backbone_type,save_dir,ckpt_fpath,model_onnx_path):
    # model = load_model(backbone_type='resnet50_irse',
    #                    save_dir='/data/output/insight_face_res50irse_emore_dist_v1',
    #                    ckpt_fpath='/data/output/insight_face_res50irse_emore_dist_v1/model_0.pth')
    #
    # output_to_onnx(model=model,
    #                model_onnx_path='insightface_res50irse_size112_emore_v1.onnx')
    # import tensorflow as tf
    # output_alin()

    # ####################################################################
    #
    # model = load_model(backbone_type='resnet50_irse_v2',
    #                    save_dir='/data/output/insight_face_res50irsev2_emore_dist/best_model4hegui',
    #                    ckpt_fpath='/data/output/insight_face_res50irsev2_emore_dist/best_model4hegui/1589829865_hegui_0.959.pth')
    # output_to_onnx(model=model,
    #                model_onnx_path='insightface_res50irsev2_cosface_size112_emore_20200525.onnx')

    ####################################################################

    # model = load_model(backbone_type='resnet50_ir',
    #                    save_dir='/data/output/insight_face_res50ir_cosface_emore_dist',
    #                    ckpt_fpath='/data/output/insight_face_res50ir_cosface_emore_dist/model_204.pth')
    # output_to_onnx(model=model,
    #                model_onnx_path='insightface_res50ir_cosface_size112_emore_20200522.onnx')

    ####################################################################

    # model = load_model(backbone_type='resnet101_ir',
    #                    save_dir='/data/output/insight_face_res101ir_cosface_emore_dist',
    #                    ckpt_fpath='/data/output/insight_face_res101ir_cosface_emore_dist/model_549.pth')
    # output_to_onnx(model=model,
    #                model_onnx_path='insightface_res101ir_cosface_size112_emore_20200522pm.onnx')

    ####################################################################

    # model = load_model(backbone_type='resnet50_irse_v3',
    #                    save_dir='/data/output/insight_face_res50irsev3_cosface_emore_dist',)
    #                    # ckpt_fpath='/data/output/insight_face_res50irsev2_emore_dist/best_model4hegui/1589829865_hegui_0.959.pth')
    # output_to_onnx(model=model,
    #                model_onnx_path='insightface_res50irsev3_cosface_size112_emore_20200525night.onnx')

    # ####################################################################
    torch.set_grad_enabled(False)
    model = load_model(backbone_type=backbone_type,
                       save_dir=save_dir,
                       ckpt_fpath=ckpt_fpath
                       )
    output_to_onnx(model=model,
                   model_onnx_path=model_onnx_path)

    ####################################################################
    # model = load_model(backbone_type='resnet101_irse_mx',
    #                    save_dir='/data/output/insight_face_res101irsemx_cosface_emore_dist',
    #                    # ckpt_fpath='/data/output/insight_face_res50irsemx_cosface_emore_dist/best_model4tupu/1590813364_tupu_0.970.pth'
    #                    )
    # output_to_onnx(model=model,
    #                model_onnx_path='insightface_res101irsemx_cosface_size112_emore_20200602.onnx')


if __name__ == '__main__':
    main()
