import torch
import torch.onnx as onnx
import os
from checkpoint_mgr.checkpoint_mgr import CheckpointMgr
from model_design_v2.model_mgr import FaceNetMgr


def load_model(backbone_type='resnet_face101',
               save_dir='/data/output/insight_face_resface101_emore_dist_v1',
               ckpt_fpath=None):
    # load model defination
    model = FaceNetMgr(backbone_type=backbone_type).get_model()
    # load weights
    checkpoint_op = CheckpointMgr(ckpt_dir=save_dir)
    checkpoint_op.load_checkpoint(model=model,
                                  ckpt_fpath=ckpt_fpath,
                                  warm_load=True,
                                  map_location='cpu')
    model.eval()
    return model


def output_to_onnx(model, model_onnx_path='insight_face_resface101_size128_emore_v1.onnx'):
    print('n_params: {}'.format(len([p for p in model.parameters()])))
    n_params = sum(p.nelement() for p in model.parameters())
    print('n_vals: {}'.format(n_params))

    dummy_input = torch.randn(1, 3, 112, 112)
    output = onnx.export(model=model,
                         args=dummy_input,
                         f=model_onnx_path,
                         verbose=True,
                         opset_version=10)

def onnx_to_tmfile(onnx_fpath, tmfile_fpath):
    onnx2tmfile_tool = '/data/FaceRecog/deploy/convert_model_to_tm'
    convert_cmd = '{} -f onnx -m {} -o {}'.format(onnx2tmfile_tool, onnx_fpath, tmfile_fpath)

    result = os.system(convert_cmd)
    print('onnx2tmfile: {}'.format(result))

def main():

    backbone_type = 'resnet50_irse_mx'
    save_dir = '/data/output/insight_face_res50irsemx_cosface_emore_dist'
    ckpt_fpath = None
    onnx_fpath = 'temp_for_gen_tmfile.onnx'
    tmfile_fpath = '{}_{}.tmfile'.format(backbone_type, '0526')

    model = load_model(backbone_type=backbone_type,
                       save_dir=save_dir,
                       ckpt_fpath=ckpt_fpath)
    output_to_onnx(model=model,
                   model_onnx_path=onnx_fpath)
    onnx_to_tmfile(onnx_fpath=onnx_fpath,
                   tmfile_fpath=tmfile_fpath)


if __name__ == '__main__':
    main()

