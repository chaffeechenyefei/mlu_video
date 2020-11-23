import torch
import torch.nn as nn
import torch.nn.functional as F


class OptimizerMgr(object):
    
    def __init__(self):
        super(OptimizerMgr, self).__init__()


    def get_optimizer(self,
                      optim_type,
                      optim_params=dict(lr=1e-3, weight_decay=0.0),
                      train_params=None,
                      models=None,
                      frozen_scope=None,
                      ):
        # optimizer
        optimizer = getattr(torch.optim, optim_type)
        if optim_params is None:
            optim_params = dict(lr=1e-3, weight_decay=1e-4)
        print(optimizer)

        # model parameters


        if frozen_scope is None:
            pass
        elif isinstance(frozen_scope, list):
            # selected_params
            for name, val in model.named_parameters():
                if any([scope in name for scope in frozen_scope]):
                    val.requires_grad = False
                    print('freeze: {}'.format(name))
        elif isinstance(frozen_scope, str):
            for name, val in model.named_parameters():
                if frozen_scope in name:
                    val.requires_grad = False
                    print('freeze: {}'.format(name))
        else:
            raise KeyError('valid frozen_scope !')



        selected_params = [val for k, val in model.named_parameters() if val.requires_grad]
        print('selected_params: {}'.format(len(selected_params)))
        optim_params.setdefault('params', selected_params)
        # print(optim_params)
        optimizer = optim_obj(**optim_params)
        print(optimizer)
        return optimizer

    def __call__(self, model, optim_type, optim_params=None, frozen_scope=None):




        
    def test_mgr(self, model, frozen_scope='resnet50'):

        # method 1, it works
        '''
        for name, child in model.named_children():
            print('-------------------- name: ', name)
            if frozen_scope in name:
            # if True:
                print('frozen [{}]'.format(name))
                for param in child.parameters():
                    param.requires_grad = False
                    print(param.size())
        '''

        # model_dict = model.state_dict()
        # for key, val in model_dict.items():
        #     print(key)
        #     print(val.requires_grad)
        #     param = model.__getattr__(name=key)
        #     print(param)

        # method 2, it works
        for name, val in model.named_parameters():
            print('name: ', name)
            print(val.requires_grad)
            if frozen_scope in name.split('.'):
                val.requires_grad = False
            # if frozen_scope in name:
            #     val.requires_grad = False

        print('#################### net.parameters() ####################')
        # for val in model.parameters():
        #     print(val.requires_grad)

        selected_params = []
        for k, val in model.named_parameters():
            print('name: ', k)
            print(val.requires_grad)
            if val.requires_grad:
                selected_params.append(val)

        # selected_params = filter(lambda p:p
        # .requires_grad, model.parameters())
        # print(selected_params)
        # for param in selected_params:
        #     print(param)
            
        # for i in range(1, self.frozen_stages + 1):
        #     m = getattr(self, 'layer{}'.format(i))
        #     m.eval_utils()
        #     for param in m.parameters():
        #         param.requires_grad = False

        # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
        #                              lr=1e-3,
        #                              amsgrad=True)
        optimizer = torch.optim.Adam(#params=model.parameters(),
                                    params=selected_params,
                                     lr=1e-3,
                                     weight_decay=1e-6)

        
    def save_opt_params(self):
        torch.save(
            self.optimizer.state_dict(), save_path /
                                         ('optimizer_{}_accuracy:{}_step:{}_{}.pth'.format(get_time(), accuracy,
                                                                                           self.step, extra)))

    def load_opt_params(self):
        self.optimizer.load_state_dict(torch.load(save_path / 'optimizer_{}'.format(fixed_str)))




if __name__=='__main__':

    from model_design.cls.cls_net import ResNet101_Cls
    model = ResNet101_Cls(n_cls=3)
    
    optim_mgr = OptimizerMgr()
    # optim_mgr.test_mgr(model=model)

    optimizer = optim_mgr(model=model,
                          optim_type='Adam',
                          frozen_scope=['resnet50.layer1', 'resnet50.layer2', 'resnet50.layer3'])
    

