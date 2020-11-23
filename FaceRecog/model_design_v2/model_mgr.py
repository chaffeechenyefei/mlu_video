import torch
import torch.nn as nn
import torch.nn.functional as F

from model_design_v2 import resnet
from model_design_v2 import focal_loss
from model_design_v2 import metrics
from model_design_v2 import arc_face as arc_face_pt
from model_design_v3 import combined_margin, cosine_face

from model_design_v3 import resnest
from model_design_v3 import arc_face
from model_design_v3 import backbone as resnet_ir_se
from model_design_v3 import resnet_irse_refit as resnet_refit
from model_design_v3 import resnet_irse_mx as resnet_mx
from model_design_v3 import resnet_irse_mx2 as resnet_mx2
from model_design_v4 import resnet_irse_activate as resnet_act


class FaceNetMgr(object):

    def __init__(self,
                 backbone_type='resnet50',
                 loss_type='arcface',
                 embedding_size=512,
                 n_person=10177,
                 lr=0.01,
                 lr_decay=0.6,
                 use_focal_loss=False,
                 frozen=[],  # ['extraction', 'classification']
                 optim_method='SGD',
                 ):
        super(FaceNetMgr, self).__init__()
        if backbone_type == 'resnet50':
            self.model = resnet.resnet50()
        elif backbone_type == 'resnet101':
            self.model = resnet.resnet101()
        elif backbone_type == 'resnet152':
            self.model = resnet.resnet152()
        elif backbone_type == 'resnet_face101':
            self.model = resnet.resnet_face101()
        elif backbone_type == 'resnest101':
            self.model = resnest.resnest101()
        elif backbone_type == 'resnet101_ir':
            self.model = resnet_ir_se.Backbone(num_layers=100, drop_ratio=0.4, mode='ir')
        elif backbone_type == 'resnet50_irse':
            self.model = resnet_ir_se.Backbone(num_layers=50, drop_ratio=0.4, mode='ir_se')
        elif backbone_type == 'resnet34_irse':
            self.model = resnet.resnet_face34(use_se=True)
        elif backbone_type == 'resnet101_irse':
            self.model = resnet.resnet_face101(use_se=True)
        elif backbone_type == 'resnet50_irse_v2':
            self.model = resnet.resnet_face50(use_se=True)
        elif backbone_type == 'resnet50_ir':
            # self.model = resnet.resnet_face50(use_se=False)
            self.model = resnet_ir_se.Backbone(num_layers=50, drop_ratio=0.4, mode='ir')
        elif backbone_type == 'resnet50_irse_v3':
            self.model = resnet_refit.resnet_face50(use_se=True)
        elif backbone_type == 'resnet50_irse_mx':
            self.model = resnet_mx.resnet_face50(use_se=True)
        elif backbone_type == 'resnet50_irse_mx2':
            self.model = resnet_mx2.resnet_face50(use_se=True)
        elif backbone_type == 'resnet50_irse_act':
            self.model = resnet_act.resnet_face50(use_se=True)
        elif backbone_type == 'resnet101_irse_mx':
            self.model = resnet_mx.resnet_face101(use_se=True)
        elif backbone_type == 'resnet50_irse_mx_compress':
            self.model = resnet_mx.resnet_face50_compress(use_se=True, mode='normal')
        elif backbone_type == 'resnet50_irse_mx_spatial':
            self.model = resnet_mx.resnet_face50_spatial(use_se=True)
        else:
            print('### ERROR invalid backbone_type: {}'.format(backbone_type))
            exit(-1)

        # self.metric_fc = metrics.ArcMarginProduct(in_features=embedding_size,
        #                                           out_features=n_person,
        #                                           s=64,
        #                                           m=0.5,
        #                                           easy_margin=False)

        # loss_type = loss_type if loss_type is not None else 'arcface'
        print('LOSS_TYPE: {}'.format(loss_type))
        if loss_type == 'arcface':
            # self.metric_fc = arc_face_pt.Arcface(embedding_size=512,
            #                                      classnum=n_person,
            #                                      s=64.0,
            #                                      m=0.5)
            self.metric_fc = arc_face.ArcFace(embedding_size=512,
                                              n_cls=n_person,
                                              scale=64.0,
                                              margin=0.5)
        elif loss_type == 'combined_margin':
            self.metric_fc = combined_margin.CombinedMargin(embedding_size=512,
                                                            n_cls=n_person)

        elif loss_type == 'cosine_face':
            self.metric_fc = cosine_face.CosineFace(in_features=embedding_size,
                                                    out_features=n_person,
                                                    s=64.0,
                                                    m=0.4)


        else:
            print('### loss_type invalid.')
            exit(1)

        if use_focal_loss:
            print('USE focal_loss.')
            self.criterion = focal_loss.FocalLoss(gamma=2.0)
        else:
            self.criterion = nn.CrossEntropyLoss()

        # trainable models
        trainable_models = {}
        if 'extraction' not in frozen:
            print('Active: {}'.format(backbone_type))
            trainable_models.setdefault('extraction', self.model)
        if 'classification' not in frozen:
            print('Active: {}'.format('classification'))
            trainable_models.setdefault('classification', self.metric_fc)

        # trainable params
        # trainable_params = TrainParamsController(weight_decay=1e-5).filter(trainable_models=[trainable_models['extraction']],
        #                                                                    no_decay_models=[trainable_models['classification']])
        #
        trainable_params = TrainParamsController(weight_decay=0.0).filter(trainable_models=trainable_models.values())
        # trainable_params = TrainParamsController(weight_decay=0.0).set_training_prop(
        #     trainable_models=trainable_models.values(),
        #     lr=(1e-2, 1e-3)
        # )

        # trainable_params = TrainParamsController(weight_decay=0.0).filter_by_key(
        #     trainable_models=trainable_models.values(),
        #     active_keys=['body.0.ir_layers'], #['se', 'prelu'],
        # )

        # optimizer
        assert optim_method in ['SGD', 'Adam', 'RMSprop']
        if optim_method == 'SGD':
            self.optimizer = torch.optim.SGD(
                trainable_params,
                lr=lr,
                momentum=0.9,
                # weight_decay=5e-4
            )
        elif optim_method == 'Adam':
            self.optimizer = torch.optim.Adam(
                trainable_params,
                lr=lr,
                weight_decay=5e-4
            )
        elif optim_method == 'RMSprop':
            self.optimizer = torch.optim.RMSprop(
                trainable_params,
                lr=lr,
                alpha=0.99,
                eps=1e-08,
                weight_decay=5e-4,
                momentum=0.9,
                centered=False
            )
        else:
            print('### ERROR, invalid optim_method: {}.'.format(optim_method))
            exit(1)

        self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer,
                                                         step_size=1,
                                                         gamma=lr_decay)

    def get_model(self):
        return self.model

    def __call__(self):
        return self.model, self.metric_fc, self.criterion, self.optimizer, self.scheduler


class TrainParamsController(object):

    def __init__(self, weight_decay=1e-5):
        super(TrainParamsController, self).__init__()
        self.weight_decay = weight_decay

    def filter(self, trainable_models, no_decay_models=[]):
        decay_params = []
        no_decay_params = []
        # decay models
        for model in trainable_models:
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if 'bn' in name or len(param.shape) == 1:
                    no_decay_params.append(param)
                    # print('--- weigh_decay_drop: {}'.format(name))
                else:
                    decay_params.append(param)
        # no_decay_models
        for model in no_decay_models:
            for name, param in model.named_parameters():
                no_decay_params.append(param)

        # resnet
        # if 'extraction' in trainable_models.keys():
        # for name, param in trainable_models[0].named_parameters():
        #     if not param.requires_grad:
        #             continue
        #     if 'bn' in name or len(param.shape) == 1:
        #         no_decay_params.append(param)
        #         # print('--- weigh_decay_drop: {}'.format(name))
        #     else:
        #         decay_params.append(param)

        print('n_no_decay: {}'.format(len(no_decay_params)))
        return [
            {'params': no_decay_params, 'weight_decay': 0.},
            {'params': decay_params, 'weight_decay': self.weight_decay}
        ]
        # return [
        #     {'params': no_decay_params, 'weight_decay': 0., 'lr':1e-3},
        #     {'params': decay_params, 'weight_decay': self.weight_decay}]

    def filter_by_key(self, trainable_models, active_keys=[]):
        decay_params = []
        no_decay_params = []
        # decay models
        for model in trainable_models:
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                remain_flag = False
                for key in active_keys:
                    if key in name:
                        remain_flag = True
                        break
                if not remain_flag:
                    continue
                else:
                    print('Active_param: {}'.format(name))
                if 'bn' in name or len(param.shape) == 1:
                    no_decay_params.append(param)
                    # print('--- weigh_decay_drop: {}'.format(name))
                else:
                    decay_params.append(param)
        print('n_active_param: {}'.format(len(decay_params) + len(no_decay_params)))
        print('n_no_decay: {}'.format(len(no_decay_params)))
        return [
            {'params': no_decay_params, 'weight_decay': 0.},
            {'params': decay_params, 'weight_decay': self.weight_decay}]

    def bn_filter(self, modules):
        if not isinstance(modules, list):
            modules = [*modules.modules()]
        paras_only_bn = []
        paras_wo_bn = []
        for layer in modules:
            if 'model' in str(layer.__class__):
                continue
            if 'container' in str(layer.__class__):
                continue
            else:
                if 'batchnorm' in str(layer.__class__):
                    paras_only_bn.extend([*layer.parameters()])
                else:
                    paras_wo_bn.extend([*layer.parameters()])
        # return paras_only_bn, paras_wo_bn
        return [
            {'params': paras_wo_bn, 'weight_decay': self.weight_decay},
            {'params': paras_only_bn}
        ]

    def set_training_prop(self, trainable_models, lr):
        if isinstance(lr, int):
            lrs = [lr] * len(trainable_models)
        elif isinstance(lr, tuple) or isinstance(lr, list):
            assert len(lr) == len(trainable_models)
            lrs = lr
        else:
            print('### ERROR: invalid learning rate: {}, reset to 1e-3.'.format(lr))
            lrs = [1e-3] * len(trainable_models)
        train_params = []
        for model_idx, model, lr in zip(range(len(trainable_models)), trainable_models, lrs):
            decay_params = []
            no_decay_params = []
            for name, param in model.named_parameters():
                if not param.requeired_grad:
                    continue
                if len(param.shape) == 1:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
            if len(decay_params) > 0:
                train_params.append(
                    {
                        'params': decay_params,
                        'weight_decay': self.weight_decay,
                        'lr': lr
                    }
                )
            if len(no_decay_params) > 0:
                train_params.append(
                    {
                        'params': no_decay_params,
                        'weight_decay': 0.0,
                        'lr': lr
                    }
                )
            print('n_no_decay for model_{}: {}'.format(model_idx, len(no_decay_params)))
        return train_params

        # resnet
        # if 'extraction' in trainable_models.keys():
        # for name, param in trainable_models[0].named_parameters():
        #     if not param.requires_grad:
        #             continue
        #     if 'bn' in name or len(param.shape) == 1:
        #         no_decay_params.append(param)
        #         # print('--- weigh_decay_drop: {}'.format(name))
        #     else:
        #         decay_params.append(param)

        return [
            {'params': no_decay_params, 'weight_decay': 0., 'lr': 1e-3},
            {'params': decay_params, 'weight_decay': self.weight_decay}]

    # def regular_loss_without_bn_and_bias(self, model):
    #     reg_loss = 0
    #     for name, param in model.named_parameters():
    #         if 'bn' not in name and len(param.shape) > 1:
    #             reg_loss += torch.norm(param)
    #         else:
    #             print('--- weight_decay_drop: {}'.format(name))
    #     return self.weight_decay * reg_loss  # Manual add weight decay

    # def add_weight_decay_without_bn_and_bias(self, model, skip_list=()):
    #     decay = []
    #     no_decay = []
    #     for name, param in model.named_parameters():
    #         if not param.requires_grad:
    #             continue
    #         if len(param.shape) == 1 or name in skip_list:
    #             no_decay.append(param)
    #         else:
    #             decay.append(param)
    #     return [
    #         {'params': no_decay, 'weight_decay': 0.},
    #         {'params': decay, 'weight_decay': self.weight_decay}]


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      # padding=0, #'valid'
                      padding=1,  # 'same',
                      # padding=2, # 'mirror'
                      bias=False, ),
            nn.BatchNorm2d(num_features=32)
        )

    def forward(self, x):
        x = self.module(x)
        return x


if __name__ == '__main__':

    model = Model()
    inputs = torch.randn(2, 32, 128, 128, dtype=torch.float32)
    output = model(inputs)
    print('output: ', output.size())
    for name, param in model.named_parameters():
        print(name, param.size())
    if isinstance(model, list):
        modules = model
    else:
        modules = [*model.modules()]
    for layer in modules:
        print(str(layer.__class__))
