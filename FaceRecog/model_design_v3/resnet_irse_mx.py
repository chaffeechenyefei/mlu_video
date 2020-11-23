import torch
import torch.nn as nn
import math
import torch.nn.utils.weight_norm as weight_norm
import torch.nn.functional as F
from sklearn.cluster import KMeans as skKmeans

################################################
"""
compress version
"""
################################################
def generate_K_alpha(m:nn.Module):
    if type(m) == FastConv2d_multi:
        m.generate_K_alpha()

class FastConv2d_multi(nn.Module):
    def __init__(self, cin, cL, cout, ksize, padding, stride, bias: bool = False, mode='load'):
        """
        alpha [cout,cL]
        K [cL,cin,ksize,ksize]
        """
        super().__init__()
        self.cin = cin
        self.cout = cout
        self.cL = cL
        self.ksize = ksize
        self.padding = padding
        self.stride = stride
        if bias:
            self.bias = nn.Parameter(torch.rand(cout))
        else:
            self.bias = None

        self.mode = mode
        self.conv1 = nn.Conv2d(cin,cL,ksize,padding=padding,stride=stride,bias=False)
        self.conv2= nn.Conv2d(cL,cout,1,padding=0,stride=1,bias=False)

        #conv1.weight = [cL,cin,ksize,ksize]
        #conv2.weight = [cout,cL,1,1]

        if mode == 'load':
            self.register_buffer( 'weight' ,torch.rand(cout, cin, ksize, ksize))


    def generate_K_alpha(self, max_iter=10):
        """
        should be used in mode 'load' and weight should be given
        """
        assert self.mode == 'load', 'Err::K can only be generated with given weights'
        """
        \arg\min | W - K*alpha |_F
        """
        with torch.no_grad():
            d = self.cin * self.ksize * self.ksize
            # [cout,cin,k,k] -> [cout,d]
            W = self.weight.view(self.cout, d)
            # K = [cL,d]
            _W = W.cpu().numpy()
            kmeans = skKmeans(n_clusters=self.cL, max_iter=600)
            kmeans.fit(_W)
            K = kmeans.cluster_centers_
            K = torch.FloatTensor(K).to(self.weight.device)
            # add parameter 1
            # self.K = nn.Parameter(K.reshape(self.cL,self.cin,self.ksize,self.ksize))

            # K=[d,cL] W=[d,cout]
            W = W.transpose(0, 1)
            K = K.transpose(0, 1)
            #             K = F.normalize(K,dim=0)

            for n_iter in range(max_iter):
                pKtK = torch.pinverse(K.transpose(0, 1) @ K)
                I = torch.diag(torch.ones(self.cL)).to(K.device)
                pKtK += 0.001 * I
                alpha = pKtK @ (K.transpose(0, 1)) @ W
                alpha = alpha.clamp(-0.5, 0.5)

                pxxt = torch.pinverse(
                    alpha @ alpha.transpose(0, 1) + 0.01 * torch.diag(torch.ones(self.cL)).to(K.device))
                K = W @ alpha.transpose(0, 1) @ pxxt

                loss = ((K @ alpha - W) ** 2).sum()

            K = K.transpose(0, 1).reshape(self.cL, self.cin, self.ksize, self.ksize)
            alpha = alpha.transpose(0,1).view(self.cout,self.cL,1,1)
            alpha = alpha.clamp(-0.5,0.5)
            # self.K = nn.Parameter(K.transpose(0, 1).reshape(self.cL, self.cin, self.ksize, self.ksize))
            # self.alpha = nn.Parameter(alpha)
            print('Final templating loss = %1.2f for shaping:' % loss, self.weight.shape, 'to', K.shape)

            self.conv1.weight = nn.Parameter(K)
            self.conv2.weight = nn.Parameter(alpha)

    def forward(self, x):
        """
        x: [B,cin,h,w]
        """
        y = self.conv1(x)
        y = self.conv2(y)

        return y



class IRSEBlock_compress(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 stride=(1, 1),
                 bottle_neck=False,
                 use_se=True,
                 mode = 'load'):
        super(IRSEBlock_compress, self).__init__()
        self.mode = mode
        _stride = stride[0]

        if out_channel <= 128:
            self.cL = out_channel//2
        else:
            self.cL = 64

        if bottle_neck:
            self.ir_layers = nn.Sequential(
                nn.BatchNorm2d(num_features=in_channel),
                nn.Conv2d(in_channels=in_channel,
                          out_channels=out_channel//4,
                          kernel_size=(1, 1),
                          stride=(1, 1),
                          padding=0,
                          bias=False),
                nn.BatchNorm2d(num_features=out_channel//4),
                nn.PReLU(num_parameters=out_channel//4),
                nn.Conv2d(in_channels=out_channel//4,
                          out_channels=out_channel//4,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1,
                          bias=False),
                nn.BatchNorm2d(num_features=out_channel//4),
                nn.PReLU(num_parameters=out_channel//4),
                nn.Conv2d(in_channels=out_channel//4,
                          out_channels=out_channel,
                          kernel_size=(1, 1),
                          stride=stride,
                          padding=0,
                          bias=False),
                nn.BatchNorm2d(num_features=out_channel)
            )
        else:
            self.ir_layers = nn.Sequential(
                nn.BatchNorm2d(num_features=in_channel),

                FastConv2d_multi(cin=in_channel,cL=self.cL,cout=out_channel,ksize=3,stride=1,padding=1,bias=False,mode=mode),
                nn.BatchNorm2d(num_features=out_channel),
                nn.PReLU(num_parameters=out_channel),
                FastConv2d_multi(cin=out_channel,cL=self.cL,cout=out_channel,ksize=3,stride=_stride,padding=1,bias=False,mode=mode),
                nn.BatchNorm2d(num_features=out_channel)
            )
        self.use_se = use_se
        if self.use_se:
            self.se = SEBlock(n_channel=out_channel)
        if stride[0] > 1 or stride[1] > 1 or in_channel != out_channel:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channels=in_channel,
                          out_channels=out_channel,
                          kernel_size=(1, 1),
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(num_features=out_channel)
            )
        else:
            self.shortcut_layer = None

    def forward(self, x):
        if self.shortcut_layer is not None:
            shortcut = self.shortcut_layer(x)
        else:
            shortcut = x
        y = self.ir_layers(x)
        if self.use_se:
            y = self.se(y)
        return y + shortcut


class SEResNetIR_compress(nn.Module):
    def __init__(self,
                 unit_blocks=[3, 4, 6, 3],
                 unit_depths=[64, 128, 256, 512],
                 bottle_neck=False,
                 use_se=True,
                 drop_ratio=0.4,
                 embedding_size=512,
                 compress_block_ind = [1,2],
                 mode = 'load' ):
        super(SEResNetIR_compress, self).__init__()
        self.unit_blocks = unit_blocks
        self.unit_depths = unit_depths
        self.bottle_neck = bottle_neck
        self.use_se = use_se
        self.drop_ratio = drop_ratio
        self.compress_block_ind = compress_block_ind
        self.mode = mode

        self.head = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=64,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.PReLU(num_parameters=64)
        )
        self.body = self._build_body()
        self.tail = nn.Sequential(
            nn.BatchNorm2d(num_features=self.unit_depths[-1]),
            nn.Dropout(p=self.drop_ratio),
        )
        self.tail_fc = nn.Linear(self.unit_depths[-1] * 7 * 7, embedding_size)
        self.tail_bn = nn.BatchNorm1d(embedding_size)

        for m in self.body:
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def initial_weights(self):
        print('doing initial for shrinking')
        self.apply(generate_K_alpha)

    def _build_body(self):
        modules = []
        for unit_idx, n_block in enumerate(self.unit_blocks):
            in_channel = self.unit_depths[unit_idx - 1] if unit_idx > 0 else 64
            depth = self.unit_depths[unit_idx]
            if unit_idx in self.compress_block_ind:
                irse_module = IRSEBlock_compress
            else:
                irse_module = IRSEBlock

            modules.append(
                irse_module(in_channel=in_channel,
                          out_channel=depth,
                          stride=(2, 2),
                          bottle_neck=self.bottle_neck,
                          use_se=self.use_se,mode=self.mode)
            )
            for _ in range(n_block-1):
                modules.append(
                    irse_module(in_channel=depth,
                              out_channel=depth,
                              stride=(1, 1),
                              bottle_neck=self.bottle_neck,
                              use_se=self.use_se, mode=self.mode)
                )
        body = nn.Sequential(*modules)
        return body

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.tail_fc(x)
        x = self.tail_bn(x)
        # x = F.normalize(x)
        return x

################################################
"""
normal
"""
################################################
class SEBlock(nn.Module):
    def __init__(self, n_channel, reduction=16):
        super(SEBlock, self).__init__()
        self.se_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Conv2d(in_channels=n_channel,
                      out_channels=n_channel // reduction,
                      kernel_size=(1, 1),
                      stride=(1, 1),
                      padding=0,
                      bias=True),
            nn.PReLU(num_parameters=n_channel // reduction),
            nn.Conv2d(in_channels=n_channel // reduction,
                      out_channels=n_channel,
                      kernel_size=(1, 1),
                      stride=(1, 1),
                      padding=0,
                      bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.se_layers(x)
        return x * y


class IRSEBlock(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 stride=(1, 1),
                 bottle_neck=False,
                 use_se=True, mode='load'):
        super(IRSEBlock, self).__init__()
        if bottle_neck:
            self.ir_layers = nn.Sequential(
                nn.BatchNorm2d(num_features=in_channel),
                nn.Conv2d(in_channels=in_channel,
                          out_channels=out_channel//4,
                          kernel_size=(1, 1),
                          stride=(1, 1),
                          padding=0,
                          bias=False),
                nn.BatchNorm2d(num_features=out_channel//4),
                nn.PReLU(num_parameters=out_channel//4),
                nn.Conv2d(in_channels=out_channel//4,
                          out_channels=out_channel//4,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1,
                          bias=False),
                nn.BatchNorm2d(num_features=out_channel//4),
                nn.PReLU(num_parameters=out_channel//4),
                nn.Conv2d(in_channels=out_channel//4,
                          out_channels=out_channel,
                          kernel_size=(1, 1),
                          stride=stride,
                          padding=0,
                          bias=False),
                nn.BatchNorm2d(num_features=out_channel)
            )
        else:
            self.ir_layers = nn.Sequential(
                nn.BatchNorm2d(num_features=in_channel),
                nn.Conv2d(in_channels=in_channel,
                          out_channels=out_channel,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1,
                          bias=False),
                nn.BatchNorm2d(num_features=out_channel),
                nn.PReLU(num_parameters=out_channel),
                nn.Conv2d(in_channels=out_channel,
                          out_channels=out_channel,
                          kernel_size=(3, 3),
                          stride=stride,
                          padding=1,
                          bias=False),
                nn.BatchNorm2d(num_features=out_channel)
            )
        self.use_se = use_se
        if self.use_se:
            self.se = SEBlock(n_channel=out_channel)
        if stride[0] > 1 or stride[1] > 1 or in_channel != out_channel:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channels=in_channel,
                          out_channels=out_channel,
                          kernel_size=(1, 1),
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(num_features=out_channel)
            )
        else:
            self.shortcut_layer = None

    def forward(self, x):
        if self.shortcut_layer is not None:
            shortcut = self.shortcut_layer(x)
        else:
            shortcut = x
        y = self.ir_layers(x)
        if self.use_se:
            y = self.se(y)
        return y + shortcut


class SEResNetIR(nn.Module):
    def __init__(self,
                 unit_blocks=[3, 4, 6, 3],
                 unit_depths=[64, 128, 256, 512],
                 bottle_neck=False,
                 use_se=True,
                 drop_ratio=0.4,
                 embedding_size=512):
        super(SEResNetIR, self).__init__()
        self.unit_blocks = unit_blocks
        self.unit_depths = unit_depths
        self.bottle_neck = bottle_neck
        self.use_se = use_se
        self.drop_ratio = drop_ratio

        self.head = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=64,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.PReLU(num_parameters=64)
        )
        self.body = self._build_body()
        self.tail = nn.Sequential(
            nn.BatchNorm2d(num_features=self.unit_depths[-1]),
            nn.Dropout(p=self.drop_ratio),
        )
        self.tail_fc = nn.Linear(self.unit_depths[-1] * 7 * 7, embedding_size)
        self.tail_bn = nn.BatchNorm1d(embedding_size)

        for m in self.body:
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _build_body(self):
        modules = []
        for unit_idx, n_block in enumerate(self.unit_blocks):
            in_channel = self.unit_depths[unit_idx - 1] if unit_idx > 0 else 64
            depth = self.unit_depths[unit_idx]
            modules.append(
                IRSEBlock(in_channel=in_channel,
                          out_channel=depth,
                          stride=(2, 2),
                          bottle_neck=self.bottle_neck,
                          use_se=self.use_se)
            )
            for _ in range(n_block-1):
                modules.append(
                    IRSEBlock(in_channel=depth,
                              out_channel=depth,
                              stride=(1, 1),
                              bottle_neck=self.bottle_neck,
                              use_se=self.use_se)
                )
        body = nn.Sequential(*modules)
        return body

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.tail_fc(x)
        x = self.tail_bn(x)
        # x = F.normalize(x)
        return x


################################################
"""
spatial aggregation
"""
################################################
class SEResNetIR_spatial(nn.Module):
    def __init__(self,
                 unit_blocks=[3, 4, 6, 3],
                 unit_depths=[64, 128, 256, 512],
                 bottle_neck=False,
                 use_se=True,
                 drop_ratio=0.4,
                 embedding_size=512):
        super(SEResNetIR_spatial, self).__init__()
        self.unit_blocks = unit_blocks
        self.unit_depths = unit_depths
        self.bottle_neck = bottle_neck
        self.use_se = use_se
        self.drop_ratio = drop_ratio

        self.head = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=64,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.PReLU(num_parameters=64)
        )
        self.body = self._build_body()
        self.tail = nn.Sequential(
            nn.BatchNorm2d(num_features=self.unit_depths[-1]),
            nn.Dropout(p=self.drop_ratio),
        )

        self.tail_att = nn.Sequential(#output [B,1,7,7]
            nn.Conv2d(in_channels=self.unit_depths[-1],out_channels=64,kernel_size=7,padding=3,stride=1),
            nn.BatchNorm2d(num_features=64),
            nn.PReLU(num_parameters=64),
            nn.Conv2d(in_channels=64,out_channels=1,kernel_size=1,padding=0,stride=1),
            nn.BatchNorm2d(num_features=1),
            nn.Sigmoid(),
        )

        self.tail_spatial = nn.AdaptiveAvgPool2d(1)#[B,C,1,1]
        self.tail_fc_shaped = nn.Linear(self.unit_depths[-1], embedding_size)
        self.tail_bn = nn.BatchNorm1d(embedding_size)

        for m in self.body:
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _build_body(self):
        modules = []
        for unit_idx, n_block in enumerate(self.unit_blocks):
            in_channel = self.unit_depths[unit_idx - 1] if unit_idx > 0 else 64
            depth = self.unit_depths[unit_idx]
            modules.append(
                IRSEBlock(in_channel=in_channel,
                          out_channel=depth,
                          stride=(2, 2),
                          bottle_neck=self.bottle_neck,
                          use_se=self.use_se)
            )
            for _ in range(n_block - 1):
                modules.append(
                    IRSEBlock(in_channel=depth,
                              out_channel=depth,
                              stride=(1, 1),
                              bottle_neck=self.bottle_neck,
                              use_se=self.use_se)
                )
        body = nn.Sequential(*modules)
        return body

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)

        w = self.tail_att(x)#[B,1,7,7]
        # print(w.shape,x.shape)
        x = w*x
        x = self.tail_spatial(x).squeeze() #[B,C,1,1]->[B,C]
        # x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.tail_fc_shaped(x)
        x = self.tail_bn(x)
        # x = F.normalize(x)
        return x


###################    SE_ResNet_IR   #####################

def resnet_face18(use_se=True, **kwargs):
    model = SEResNetIR(unit_blocks=[2, 2, 2, 2], unit_depths=[64, 128, 256, 512], use_se=use_se)
    return model


def resnet_face34(use_se=True):
    model = SEResNetIR(unit_blocks=[3, 4, 6, 3], unit_depths=[64, 128, 256, 512], use_se=use_se)
    return model


def resnet_face50(use_se=True, **kwargs):
    model = SEResNetIR(unit_blocks=[3, 4, 14, 3], unit_depths=[64, 128, 256, 512], use_se=use_se)
    return model

def resnet_face50_compress(use_se=True, mode='load' ,**kwargs):
    model = SEResNetIR_compress(unit_blocks=[3, 4, 14, 3], unit_depths=[64, 128, 256, 512], use_se=use_se, mode=mode)
    return model

def resnet_face50_compress_v2(use_se=True, compress_block_ind = [0,1,2,3] ,mode='load' ,**kwargs):
    model = SEResNetIR_compress(unit_blocks=[3, 4, 14, 3], unit_depths=[64, 128, 256, 512],
                                use_se=use_se, mode=mode, compress_block_ind=compress_block_ind)
    return model

def resnet_face50_spatial(use_se=True,**kwargs):
    model = SEResNetIR_spatial(unit_blocks=[3, 4, 14, 3], unit_depths=[64, 128, 256, 512], use_se=use_se)
    return model

def resnet_face101(use_se=True, **kwargs):
    model = SEResNetIR(unit_blocks=[3, 13, 30, 3], unit_depths=[64, 128, 256, 512], use_se=use_se)
    return model
#
def resnet_face152(bottle_neck=True, use_se=True, **kwargs):
    model = SEResNetIR(unit_blocks=[3, 8, 36, 3],
                       unit_depths=[256, 512, 1024, 2048],
                       bottle_neck=bottle_neck,
                       use_se=use_se)
    return model



def main():
    model = SEResNetIR_spatial()
    input_x = torch.randn(2, 3, 112, 112)
    output = model(input_x)
    print(output.size())


if __name__ == '__main__':
    main()