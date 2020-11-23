import torch
import torch.nn as nn
import math
import torch.nn.utils.weight_norm as weight_norm
import torch.nn.functional as F


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
                 use_se=True):
        super(IRSEBlock, self).__init__()
        self.ir_layers_0 = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channel),
            nn.Conv2d(in_channels=in_channel,
                      out_channels=out_channel,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(num_features=out_channel),
            nn.PReLU(num_parameters=out_channel),
        )
        # self.ir_prelu = nn.LeakyReLU()
        self.ir_layers_1 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel,
                      out_channels=out_channel,
                      kernel_size=(3, 3),
                      stride=stride,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(num_features=out_channel)
        )
        self.ir_prelu = nn.PReLU(num_parameters=out_channel)

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
        y = self.ir_layers_0(x)
        y = self.ir_layers_1(y)
        if self.use_se:
            y = self.se(y)
        return self.ir_prelu(y + shortcut)


class SEResNetIR(nn.Module):
    def __init__(self,
                 unit_blocks=[3, 4, 6, 3],
                 unit_depths=[64, 128, 256, 512],
                 use_se=True,
                 drop_ratio=0.4,
                 embedding_size=512):
        super(SEResNetIR, self).__init__()
        self.unit_blocks = unit_blocks
        self.unit_depths = unit_depths
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
                          use_se=self.use_se)
            )
            for _ in range(n_block - 1):
                modules.append(
                    IRSEBlock(in_channel=depth,
                              out_channel=depth,
                              stride=(1, 1),
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


def resnet_face101(use_se=True, **kwargs):
    model = SEResNetIR(unit_blocks=[3, 13, 30, 3], unit_depths=[64, 128, 256, 512], use_se=use_se)
    return model


def main():
    model = resnet_face18()

    input_x = torch.randn(2, 3, 112, 112)
    output = model(input_x)
    print(output.size())
    for name, param in model.named_parameters():
        print(name, param.size(), len(param.shape))

    modules = [*model.modules()]
    for layer in modules:
        print(str(layer.__class__))


if __name__ == '__main__':
    main()
