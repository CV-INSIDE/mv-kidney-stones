import helpers.transferLearningBaseModel as tlm
import torch
import warnings
import torchvision.models as models
import torch.nn as nn
from collections import namedtuple
from torch import Tensor
from typing import Callable, Any, Optional, Tuple, List
from einops import reduce

__all__ = ['Inception3', 'inception_v3', 'InceptionOutputs', '_InceptionOutputs']


model_urls = {
    # Inception v3 ported from TensorFlow
    'inception_v3_google': 'https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth',
}

InceptionOutputs = namedtuple('InceptionOutputs', ['logits', 'aux_logits'])
InceptionOutputs.__annotations__ = {'logits': Tensor, 'aux_logits': Optional[Tensor]}

# Script annotations failed with _GoogleNetOutputs = namedtuple ...
# _InceptionOutputs set here for backwards compat
_InceptionOutputs = InceptionOutputs


class InceptionModel(tlm.BaseModel):
    def __init__(self, hparams={}, num_classes=6, batch_size=64, pretrained=False, seed=None):
        if "lr" not in hparams:
            hparams["lr"] = 0.001
        #LOG INFO
        hparams["num_classes"] = num_classes
        hparams["batch_size"] = batch_size
        hparams["is_pretrained"] = pretrained
        super(InceptionModel, self).__init__(hparams, seed=seed)
        self.inception = models.inception_v3(pretrained=pretrained, init_weights=True, aux_logits=False)
        #self.inception.fc = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.Linear(512, num_classes))
        self.inception.fc = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.Linear(512, 256),
                                          nn.ReLU(), nn.Linear(256, num_classes))
        # self.inception.fc2 = nn.Sequential(nn.Linear(256, num_classes))
        self.batch_size = batch_size
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.inception(x)


class InceptionModeMulti(tlm.BaseModel):
    def __init__(self, hparams={}, num_classes=6, batch_size=64, pretrained=False, seed=None):
        if "lr" not in hparams:
            hparams["lr"] = 0.001
        # LOG INFO
        hparams["num_classes"] = num_classes
        hparams["batch_size"] = batch_size
        hparams["is_pretrained"] = pretrained
        super(InceptionModeMulti, self).__init__(hparams, seed=seed)
        self.inception = models.inception_v3(pretrained=pretrained, init_weights=True, aux_logits=False)

        # Freeze Parameters
        for param1 in self.inception.parameters():
            param1.requires_grad = False

        self.inception.fc = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.Linear(512, 256),
                                          nn.ReLU(), nn.Linear(256, num_classes))

        self.batch_size = batch_size
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, y):
        x = self.inception._transform_input(x)
        x, aux = self.custom_forward(x)
        aux_defined = self.training and self.inception.aux_logits

        y = self.inception._transform_input(y)
        y, aux = self.custom_forward(y)

        xy = torch.stack([x, y])
        xy = reduce(xy, 'c b l -> 1 b l', 'max')
        xy = torch.squeeze(xy, dim=0)
        xy = self.inception.fc(xy)

        if torch.jit.is_scripting():
            if not aux_defined:
                warnings.warn("Scripted Inception3 always returns Inception3 Tuple")
            return InceptionOutputs(xy, aux)
        else:
            return self.inception.eager_outputs(xy, aux)

    def custom_forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        # N x 3 x 299 x 299
        x = self.inception.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.inception.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.inception.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.inception.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.inception.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.inception.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.inception.maxpool2(x)
        # N x 192 x 35 x 35
        x = self.inception.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.inception.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.inception.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.inception.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.inception.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.inception.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.inception.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.inception.Mixed_6e(x)
        # N x 768 x 17 x 17
        aux: Optional[Tensor] = None
        if self.inception.AuxLogits is not None:
            if self.training:
                aux = self.inception.AuxLogits(x)
        # N x 768 x 17 x 17
        x = self.inception.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.inception.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.inception.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = self.inception.avgpool(x)
        # N x 2048 x 1 x 1
        x = self.inception.dropout(x)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 2048
        return x, aux