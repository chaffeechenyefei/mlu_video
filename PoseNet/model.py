import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import os
from typing import Dict


class poseNetv1(nn.Module):
    def __init__(self, nn_layers: list = [8, 16, 32, 13], dropout_keep_ratio=0.9):
        super().__init__()
        assert len(nn_layers) >= 2, 'Err@poseNetv1: Number of nn_layers too small.'
        assert nn_layers[0] == 8, 'Err@poseNetv1: First element must be 8.'
        self.fin = nn_layers[0]
        self.fout = nn_layers[-1]
        self.net = nn.Sequential()
        self.dropout_keep_ratio = dropout_keep_ratio
        for dth, neurons in enumerate(nn_layers):
            if dth == len(nn_layers) - 1:
                break
            else:
                if dth == len(nn_layers) - 2:
                    self.net.add_module('dropout', nn.Dropout(p=dropout_keep_ratio))

                self.net.add_module('linear_%d_%dx%d' % (dth, nn_layers[dth], nn_layers[dth + 1]),
                                    nn.Linear(nn_layers[dth], nn_layers[dth + 1], bias=True))
                self.net.add_module('leaky_%d' % dth, nn.LeakyReLU())

    def format(self, x):
        bbox = x[:, :4]
        landmk = x[:, 5:]  # [B,10]
        width = (bbox[:, 2] - bbox[:, 0]).view(-1, 1)+1e-3  # [B,1]
        height = (bbox[:, 3] - bbox[:, 1]).view(-1, 1)+1e-3
        cx = ((bbox[:, 2] + bbox[:, 0]) / 2).view(-1, 1)  # [B,1]
        cy = ((bbox[:, 3] + bbox[:, 1]) / 2).view(-1, 1)

        # scale to [0,1]
        landmk[:, ::2] = (landmk[:, ::2] - cx) / width
        landmk[:, 1::2] = (landmk[:, 1::2] - cy) / height
        # set nose to center point
        nx = 1.0 * landmk[:, 4].view(-1, 1)
        ny = 1.0 * landmk[:, 5].view(-1, 1)
        landmk[:, ::2] = landmk[:, ::2] - nx
        landmk[:, 1::2] = landmk[:, 1::2] - ny

        res = torch.cat([landmk[:, :4], landmk[:, 6:]], dim=1).view(-1, 2 * 4)
        return res

    def forward(self, x):
        """
        :param x: [B,D]; D is composed of [x0,y0,x1,y1,score,lex,ley,rex,rey,nx,ny,lmx,lmy,rmx,rmy] 
        :return: class

        """
        x = self.format(x)
        y = self.net(x).view(-1, self.fout)
        return y

    def predict(self, x):
        x = self.forward(x)
        y = F.softmax(x, dim=1)
        return y

    def predictidx(self, x):
        x = self.forward(x)
        _, y = torch.topk(x, k=1, dim=1, largest=True, sorted=True)
        return y

    def predictidx_v2(self, x, frontal_position=3):
        x = self.forward(x)
        v, y = torch.topk(x, k=1, dim=1, largest=True, sorted=True)
        return {
            'label': y,
            'prob': v,
            'frontal_prob': x[:, frontal_position]
        }

    def loss(self):
        return nn.CrossEntropyLoss(reduction='mean')

    def label_explanar(self, ncls: int):
        if ncls == 13:
            return {0: 34, 1: 2, 2: 5, 3: 37, 4: 7, 5: 9, 6: 11, 7: 14, 8: 22, 9: 25, 10: 27, 11: 29, 12: 31}
        elif ncls == 7:
            return {
                0: {'id': [2, 25, 37], 'flip_label': 4, 'name': 'Big Right'},
                1: {'id': [5], 'flip_label': 3, 'name': 'Right'},
                2: {'id': [7, 9, 27], 'flip_label': 2, 'name': 'Frontal'},
                3: {'id': [29], 'flip_label': 1, 'name': 'Left'},
                4: {'id': [11, 14, 31], 'flip_label': 0, 'name': 'Big Left'},
                5: {'id': [22], 'flip_label': 6, 'name': 'Non-Frontal'},
                6: {'id': [34], 'flip_label': 5, 'name': 'Non-Frontal'},
            }
        elif ncls == 9:
            return {
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
        else:
            print('Err no such experiment')
            return None


class poseNetv2_mt(poseNetv1):
    """
    _mt: multi-target with pose classification and angle regression
    """

    def __init__(self, nn_layers: list = [8, 16, 32, 13], dropout_keep_ratio=0.9):
        super(poseNetv2_mt, self).__init__(nn_layers=nn_layers, dropout_keep_ratio=dropout_keep_ratio)

    def forward(self, x):
        """
        :param x: [B,D]; D is composed of [x0,y0,x1,y1,score,lex,ley,rex,rey,nx,ny,lmx,lmy,rmx,rmy] 
        :return: class

        """
        x = self.format(x)
        y = self.net(x).view(-1, self.fout)
        return y

    def predictidx(self, x):
        x = self.forward(x)
        _, y = torch.topk(x[:, :-1], k=1, dim=1, largest=True, sorted=True)
        return y, x[:, -1].view(-1, 1)  # cls,pose

    def predictidx_v2(self, x, frontal_position=3):
        x = self.forward(x)
        v, y = torch.topk(x[:, :-1], k=1, dim=1, largest=True, sorted=True)
        return {
            'label': y,
            'prob': v,
            'frontal_prob': x[:, frontal_position],
            'arc_angle': x[:, -1].view(-1, 1)
        }

    def predict(self, x):
        x = self.forward(x)
        y = F.softmax(x[:, :-1], dim=1)
        return y, x[:, -1].view(-1, 1)

    def loss(self):
        return nn.CrossEntropyLoss(reduction='mean'), nn.L1Loss(reduction='mean')


class poseNetv3_mt(poseNetv1):
    """
    _mt: multi-target with pose classification and angle regression
    """

    def __init__(self, nn_layers: list = [8, 16, 32, 13], dropout_keep_ratio=0.9):
        super().__init__()
        assert len(nn_layers) >= 2, 'Err@poseNetv1: Number of nn_layers too small.'
        assert nn_layers[0] == 8, 'Err@poseNetv1: First element must be 8.'
        self.fin = nn_layers[0]
        self.fout = nn_layers[-1]
        self.net = nn.Sequential()
        self.dropout_keep_ratio = dropout_keep_ratio
        for dth, neurons in enumerate(nn_layers):
            if dth == len(nn_layers) - 1:
                break
            else:
                if dth == len(nn_layers) - 2:
                    self.net.add_module('dropout', nn.Dropout(p=dropout_keep_ratio))
                    self.net.add_module('linear_%d_%dx%d' % (dth, nn_layers[dth], nn_layers[dth + 1]),
                                        nn.Linear(nn_layers[dth], nn_layers[dth + 1], bias=True))
                else:
                    self.net.add_module('linear_%d_%dx%d' % (dth, nn_layers[dth], nn_layers[dth + 1]),
                                        nn.Linear(nn_layers[dth], nn_layers[dth + 1], bias=True))
                    self.net.add_module('leaky_%d' % dth, nn.LeakyReLU())
        self.tanh = nn.Tanh()

    def forward(self, x):
        """
        :param x: [B,D]; D is composed of [x0,y0,x1,y1,score,lex,ley,rex,rey,nx,ny,lmx,lmy,rmx,rmy] 
        :return: class

        """
        x = self.format(x)
        y = self.net(x).view(-1, self.fout)
        y[:, -1] = self.tanh(y[:, -1]) * math.pi
        return y

    def predictidx(self, x):
        x = self.forward(x)
        _, y = torch.topk(x[:, :-1], k=1, dim=1, largest=True, sorted=True)
        return y, x[:, -1].view(-1, 1)  # cls,pose

    def predictidx_v2(self, x, frontal_position=3):
        x = self.forward(x)
        prob = F.softmax(x[:, :-1], dim=1)
        v, y = torch.topk(prob, k=1, dim=1, largest=True, sorted=True)
        return {
            'label': y,
            'prob': v,
            'frontal_prob': prob[:, frontal_position],
            'arc_angle': x[:, -1].view(-1, 1)
        }

    def predict(self, x):
        y = self.forward(x)
        y[:,:-1] = F.softmax(y[:, :-1], dim=1)
        return y

    def loss(self):
        return nn.CrossEntropyLoss(reduction='mean'), nn.MSELoss(reduction='mean')


def adjust_lr(epoch, optimizer, decay_epoch, decay=0.1):
    if epoch % decay_epoch == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = decay * param_group['lr']


def save_model(model, epoch, step, output,args=None):
    try:
        os.mkdir(output)
        print('Create checkpoint dir {}'.format(output))
    except:
        pass
    if args is None:
        torch.save(model.state_dict(), os.path.join(output, '{}_{}.pth'.format(epoch, step)))
        print('{}_{}.pth has been saved in {}'.format(epoch, step, output))
    else:
        torch.save(model.state_dict(), os.path.join(output, '{}_{}_{}.pth'.format(args.model, epoch, step)))
        print('{}_{}_{}.pth has been saved in {}'.format(args.model,epoch, step, output))


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

def load_model_with_dict_replace(model:nn.Module, path , old_phs:str , new_phs:str='', lastLayer:bool=True) -> Dict:
    """
    change the pre-desc of dict
    :param model: 
    :param path: 
    :return: 
    """
    device = torch.device('cpu')
    state = torch.load(str(path),map_location=device)
    sz_old_phs = len(old_phs)
    if lastLayer:
        new_state = { str(new_phs+k[sz_old_phs:]):v for k,v in state.items() }
        model.load_state_dict(new_state)
    else:
        cur_state = model.state_dict()
        new_state = {str(new_phs + k[sz_old_phs:]): v for k, v in state.items() if 'last_linear' not in k }
        cur_state.update(new_state)
        model.load_state_dict(cur_state)
    print('Load model from ' + str(path) )
    return state

# import onnxruntime
# class ONNXModel():
#     def __init__(self, onnx_path):
#         """
#         :param onnx_path:
#         """
#         self.onnx_session = onnxruntime.InferenceSession(onnx_path)
#         self.input_name = self.get_input_name(self.onnx_session)
#         self.output_name = self.get_output_name(self.onnx_session)
#         print("input_name:{}".format(self.input_name))
#         print("output_name:{}".format(self.output_name))
#
#     def get_output_name(self, onnx_session):
#         """
#         output_name = onnx_session.get_outputs()[0].name
#         :param onnx_session:
#         :return:
#         """
#         output_name = []
#         for node in onnx_session.get_outputs():
#             output_name.append(node.name)
#         return output_name
#
#     def get_input_name(self, onnx_session):
#         """
#         input_name = onnx_session.get_inputs()[0].name
#         :param onnx_session:
#         :return:
#         """
#         input_name = []
#         for node in onnx_session.get_inputs():
#             input_name.append(node.name)
#         return input_name
#
#     def get_input_feed(self, input_name, image_tensor):
#         """
#         input_feed={self.input_name: image_tensor}
#         :param input_name:
#         :param image_tensor:
#         :return:
#         """
#         input_feed = {}
#         for name in input_name:
#             input_feed[name] = image_tensor
#         return input_feed
#
#     def forward(self, image_tensor):
#         '''
#         image_tensor = image.transpose(2, 0, 1)
#         image_tensor = image_tensor[np.newaxis, :]
#         onnx_session.run([output_name], {input_name: x})
#         :param image_tensor:
#         :return:
#         '''
#         # 输入数据的类型必须与模型一致,以下三种写法都是可以的
#         # scores, boxes = self.onnx_session.run(None, {self.input_name: image_tensor})
#         # scores, boxes = self.onnx_session.run(self.output_name, input_feed={self.input_name: image_tensor})
#         input_feed = self.get_input_feed(self.input_name, image_tensor)
#         output = self.onnx_session.run(self.output_name, input_feed=input_feed)
#         return output


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()