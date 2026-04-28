from torchvision import models as vision_models
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CoordConv2d(nn.Conv2d, nn.Module):
    """
    2D Coordinate Convolution

    Source: An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution
    https://arxiv.org/abs/1807.03247
    (e.g. adds 2 channels per input feature map corresponding to (x, y) location on map)
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode='zeros',
        coord_encoding='position',
    ):
        """
        Args:
            in_channels: number of channels of the input tensor [C, H, W]
            out_channels: number of output channels of the layer
            kernel_size: convolution kernel size
            stride: conv stride
            padding: conv padding
            dilation: conv dilation
            groups: conv groups
            bias: conv bias
            padding_mode: conv padding mode
            coord_encoding: type of coordinate encoding. currently only 'position' is implemented
        """

        assert(coord_encoding in ['position'])
        self.coord_encoding = coord_encoding
        if coord_encoding == 'position':
            in_channels += 2  # two extra channel for positional encoding
            self._position_enc = None  # position encoding
        else:
            raise Exception("CoordConv2d: coord encoding {} not implemented".format(self.coord_encoding))
        nn.Conv2d.__init__(
            self,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode
        )

    def output_shape(self, input_shape):
        """
        Function to compute output shape from inputs to this module.

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """

        # adds 2 to channel dimension
        return [input_shape[0] + 2] + input_shape[1:]

    def forward(self, input):
        b, c, h, w = input.shape
        if self.coord_encoding == 'position':
            if self._position_enc is None:
                pos_y, pos_x = torch.meshgrid(torch.arange(h), torch.arange(w))
                pos_y = pos_y.float().to(input.device) / float(h)
                pos_x = pos_x.float().to(input.device) / float(w)
                self._position_enc = torch.stack((pos_y, pos_x)).unsqueeze(0)
            pos_enc = self._position_enc.expand(b, -1, -1, -1)
            input = torch.cat((input, pos_enc), dim=1)
        return super(CoordConv2d, self).forward(input)
#
#
#
# class ConvBase(nn.Module):
#     """
#     Base class for ConvNets.
#     """
#     def __init__(self):
#         super(ConvBase, self).__init__()
#
#     # dirty hack - re-implement to pass the buck onto subclasses from ABC parent
#     def output_shape(self, input_shape):
#         """
#         Function to compute output shape from inputs to this module.
#
#         Args:
#             input_shape (iterable of int): shape of input. Does not include batch dimension.
#                 Some modules may not need this argument, if their output does not depend
#                 on the size of the input, or if they assume fixed size input.
#
#         Returns:
#             out_shape ([int]): list of integers corresponding to output shape
#         """
#         raise NotImplementedError
#
#     def forward(self, inputs):
#         x = self.nets(inputs)
#
#         if list(self.output_shape(list(inputs.shape)[1:])) != list(x.shape)[1:]:
#             raise ValueError('Size mismatch: expect size %s, but got size %s' % (
#                 str(self.output_shape(list(inputs.shape)[1:])), str(list(x.shape)[1:]))
#             )
#         return x
#
# class ResNet18Conv(ConvBase):
#     """
#     A ResNet18 block that can be used to process input images.
#     """
#     def __init__(
#         self,
#         input_channel=3,
#         pretrained=False,
#         input_coord_conv=False,
#     ):
#         """
#         Args:
#             input_channel (int): number of input channels for input images to the network.
#                 If not equal to 3, modifies first conv layer in ResNet to handle the number
#                 of input channels.
#             pretrained (bool): if True, load pretrained weights for all ResNet layers.
#             input_coord_conv (bool): if True, use a coordinate convolution for the first layer
#                 (a convolution where input channels are modified to encode spatial pixel location)
#         """
#         super(ResNet18Conv, self).__init__()
#         net = vision_models.resnet18(pretrained=pretrained)
#
#         if input_coord_conv:
#             net.conv1 = CoordConv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         elif input_channel != 3:
#             net.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
#
#         # cut the last fc layer
#         self._input_coord_conv = input_coord_conv
#         self._input_channel = input_channel
#         self.nets = torch.nn.Sequential(*(list(net.children())[:-2]))
#
#     def output_shape(self, input_shape):
#         """
#         Function to compute output shape from inputs to this module.
#
#         Args:
#             input_shape (iterable of int): shape of input. Does not include batch dimension.
#                 Some modules may not need this argument, if their output does not depend
#                 on the size of the input, or if they assume fixed size input.
#
#         Returns:
#             out_shape ([int]): list of integers corresponding to output shape
#         """
#         assert(len(input_shape) == 3)
#         out_h = int(math.ceil(input_shape[1] / 32.))
#         out_w = int(math.ceil(input_shape[2] / 32.))
#         return [512, out_h, out_w]
#
#     def __repr__(self):
#         """Pretty print network."""
#         header = '{}'.format(str(self.__class__.__name__))
#         return header + '(input_channel={}, input_coord_conv={})'.format(self._input_channel, self._input_coord_conv)



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride!= 1 or in_channels!= out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class  ResNet18(nn.Module):
    def __init__(self, input_dim, num_classes=512):
        '''
        mapping_dim 是 坐标变换矩阵
        '''
        super(ResNet18, self).__init__()
        self.in_channels = 64
        #self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1 = CoordConv2d(input_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)
        self.maxpool2 = nn.MaxPool2d(kernel_size=4, stride=2, padding=1)
        self.fc = nn.Linear(512 * 16, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        #out = self.avgpool(out)
        out = self.maxpool2(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

