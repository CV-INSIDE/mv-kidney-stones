"""
Library dedicated to the resnet with attention creation
"""
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import init
from models.attention.cbam import CBAM

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample = None, use_cbam= False,
                 reduction_ratio=16, norm_layer = None, pool_types=['avg', 'max']) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        # Define if we want to use cbam or not
        if use_cbam:
            self.cbam = CBAM(planes, reduction_ratio=reduction_ratio, pool_types=pool_types)
        else:
            self.cbam = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        if self.cbam is not None:
            out = self.cbam(out)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample= None, groups: int = 1,
                 base_width: int = 64, dilation: int = 1, norm_layer = None, use_cbam=False, reduction_ratio=16,
                 pool_types=['avg', 'max']):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        if use_cbam:
            self.cbam = CBAM(planes * self.expansion, reduction_ratio=reduction_ratio, pool_types=pool_types)
        else:
            self.cbam = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        if self.cbam is not None:
            out = self.cbam(out)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers,  network_type, num_classes, norm_layer=None, att_type=None, reduction_ratio=16):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        self.reduction_ratio = reduction_ratio
        self.network_type = network_type

        # There is a different model between CIFAR and ImageNet
        if network_type == "ImageNet":
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.avgpool = nn.AvgPool2d(7)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64,  layers[0], att_type=att_type)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, att_type=att_type)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, att_type=att_type)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, att_type=att_type)

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Initialize weights
        init.kaiming_normal_(self.fc.weight)
        for key in self.state_dict():
            if key.split('.')[-1]=="weight":
                if "conv" in key:
                    init.kaiming_normal_(self.state_dict()[key], mode='fan_out')
                if "bn" in key:
                    if "SpatialGate" in key:
                        self.state_dict()[key][...] = 0
                    else:
                        self.state_dict()[key][...] = 1
            elif key.split(".")[-1]=='bias':
                self.state_dict()[key][...] = 0

    def _make_layer(self, block, planes, blocks, stride=1, att_type=None):
        # a
        downsample = None
        norm_layer = self._norm_layer

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride=stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_cbam=att_type=='CBAM',
                            reduction_ratio=self.reduction_ratio, norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_cbam=att_type=='CBAM',
                                reduction_ratio=self.reduction_ratio, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        if self.network_type == "ImageNet":
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.network_type == "ImageNet":
            x = self.avgpool(x)
        else:
            x = F.avg_pool2d(x, 4)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def ResNet18(network_type="ImageNet", num_classes=1000, reduction_ratio=16, att_type=None):
    return ResNet(
                BasicBlock,
                [2,2,2,2],
                network_type=network_type,
                num_classes=num_classes,
                reduction_ratio= reduction_ratio,
                att_type=att_type
                )


def ResNet34(network_type="ImageNet", num_classes=1000, reduction_ratio=16, att_type=None):
    return ResNet(
        BasicBlock,
        [3,4,6,3],
        network_type=network_type,
        num_classes=num_classes,
        reduction_ratio=reduction_ratio,
        att_type=att_type
        )

def ResNet50(network_type="ImageNet", num_classes=1000, reduction_ratio=16, att_type=None):
    return ResNet(
        Bottleneck,
        [3,4,6,3],
        network_type=network_type,
        num_classes=num_classes,
        reduction_ratio=reduction_ratio,
        att_type=att_type
        )

def ResNet101(network_type="ImageNet", num_classes=1000, reduction_ratio=16, att_type=None):
    return ResNet(
        Bottleneck,
        [3,4,23,3],
        network_type=network_type,
        num_classes=num_classes,
        reduction_ratio=reduction_ratio,
        att_type=att_type
        )

def ResNet152(network_type="ImageNet", num_classes=1000, reduction_ratio=16, att_type=None):
    return ResNet(
        Bottleneck,
        [3,8,36,3],
        network_type=network_type,
        num_classes=num_classes,
        reduction_ratio=reduction_ratio,
        att_type=att_type
        )