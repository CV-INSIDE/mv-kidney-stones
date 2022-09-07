import os
import random
import logging
import numpy as np
import pytorch_lightning as pl
from collections import OrderedDict

import torch
from torchvision import datasets
from torch.utils.data import DataLoader, SubsetRandomSampler
from models.attention.models.resnet import ResNet50Cbam

# test_dataset = datasets.CIFAR10(".", train=False, transform=None)


checkpoint = torch.load(r'C:\Users\15B38LA\Downloads\attention_resnet50 (1).ckpt')
test2 = ResNet50Cbam(num_classes=10, pretrained=True,)
test = ResNet50Cbam.load_from_checkpoint(r'C:\Users\15B38LA\Downloads\attention_resnet50 (1).ckpt', strict=True)
a = test2.state_dict()
b = checkpoint['state_dict']
for key in a.keys():
    assert torch.equal(a[key], b[key])
test2 = OrderedDict({k: v for k, v in checkpoint['state_dict'].items()})
a = 1
