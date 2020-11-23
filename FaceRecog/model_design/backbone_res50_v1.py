import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(Bottleneck, self).__init__()
        if in_channel == out_channel:
            self.shortcut_layer = nn.MaxPool2d(kernel_size=1, stride=stride)
        else:
            self.shortcut_layer = nn.Sequential(nn.Conv2d(in_channels=in_channel,
                                                          out_channels=out_channel,
                                                          kernel_size=1,
                                                          stride=stride,
                                                          bias=False),
                                                nn.BatchNorm2d(num_features=out_channel))
        self.res_layer = nn.Sequential(nn.Conv2d(in_channels=in_channel,
                                                 out_channels=out_channel,
                                                 kernel_size=3,
                                                 stride=1,
                                                 padding=1,
                                                 bias=False),
                                       nn.BatchNorm2d(num_features=out_channel),
                                       nn.PReLU(num_parameters=out_channel),
                                       nn.Conv2d(in_channels=out_channel,
                                                 out_channels=out_channel,
                                                 kernel_size=3,
                                                 stride=stride,
                                                 padding=1,
                                                 bias=False),
                                       nn.BatchNorm2d(num_features=out_channel)
                                       )

    def forward(self, input):
        shortcut = self.shortcut_layer(input)
        residual = self.res_layer(input)
        return shortcut + residual


class ResNet50(nn.Module):

    def __init__(self, dropout_ratio=0.1):
        super(ResNet50, self).__init__()
        self.head_conv = nn.Conv2d(in_channels=3,
                                   out_channels=32,
                                   kernel_size=3,
                                   stride=1,
                                   bias=False)

        self.body_blocks = self._body_blocks()

        self.tail_bn1 = nn.BatchNorm2d(num_features=512)
        self.tail_dropout = nn.Dropout(p=dropout_ratio)
        self.tail_fc = nn.Linear(in_features=512 * 7 * 7, out_features=512)
        self.tail_bn2 = nn.BatchNorm1d(num_features=512)

    def _body_blocks(self):
        blocks = nn.ModuleList()
        # block1
        blocks.append(Bottleneck(in_channel=32,
                                 out_channel=64,
                                 stride=2))
        for i in range(2):
            blocks.append(Bottleneck(in_channel=64,
                                     out_channel=64,
                                     stride=1))
        # block2
        blocks.append(Bottleneck(in_channel=64,
                                 out_channel=128,
                                 stride=2))
        for i in range(3):
            blocks.append(Bottleneck(in_channel=128,
                                     out_channel=128,
                                     stride=1))

        # block3
        blocks.append(Bottleneck(in_channel=128,
                                 out_channel=256,
                                 stride=2))
        for i in range(13):
            blocks.append(Bottleneck(in_channel=256,
                                     out_channel=256,
                                     stride=1))

        # block4
        blocks.append(Bottleneck(in_channel=256,
                                 out_channel=512,
                                 stride=2))
        for i in range(2):
            blocks.append(Bottleneck(in_channel=512,
                                     out_channel=512,
                                     stride=1))
        return blocks

    def forward(self, input):
        feature = self.head_conv(input)
        for block in self.body_blocks:
            feature = block(feature)
        feature = self.tail_bn1(feature)
        feature = self.tail_dropout(feature)
        feature = feature.view(input.size()[0], -1)
        feature = self.tail_fc(feature)
        feature = self.tail_bn2(feature)
        return feature


if __name__ == '__main__':
    model = ResNet50()
    # model = Bottleneck(in_channel=3, out_channel=6, stride=2)

    input = torch.randn(2, 3, 112, 112, dtype=torch.float32)
    output = model(input)
    print(output.size())
