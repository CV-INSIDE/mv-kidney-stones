import helpers.transferLearningBaseModel_original as tlm
import torch.nn as nn

from models.attention.models.resnet_utils import ResNet50, ResNet34
from helpers.isic_utilities import SigmoidFocalLoss


class ResNet50Cbam(tlm.BaseModel):
    def __init__(self, hparams={}, num_classes=10, batch_size=64, pretrained=False, seed=None):
        if "lr" not in hparams:
            hparams["lr"] = 0.001

        # LOG INFO
        hparams["num_classes"] = num_classes
        hparams["batch_size"] = batch_size
        hparams["is_pretrained"] = pretrained
        super(ResNet50Cbam, self).__init__(hparams, seed=seed)

        self.resnet = ResNet50(network_type="ImageNet", num_classes=num_classes, att_type='CBAM')
        self.resnet.fc = nn.Sequential(nn.Linear(2048, 512),
                                       nn.ReLU(),
                                       nn.Dropout(p=0.5),
                                       nn.Linear(512, 256),
                                       nn.ReLU(),
                                       nn.Dropout(p=0.5),
                                       nn.Linear(256, num_classes))
        self.batch_size = batch_size
        # self.loss_fn = nn.CrossEntropyLoss()
        self.loss_fn = SigmoidFocalLoss(reduce='mean', n_classes=num_classes)


    def forward(self, x):
        return self.resnet(x)



class ResNet34Cbam(tlm.BaseModel):
    def __init__(self, hparams={}, num_classes=6, batch_size=64, pretrained=False, seed=None):
        if "lr" not in hparams:
            hparams["lr"] = 0.001

        # LOG INFO
        hparams["num_classes"] = num_classes
        hparams["batch_size"] = batch_size
        hparams["is_pretrained"] = pretrained
        super(ResNet34Cbam, self).__init__(hparams, seed=seed)

        self.resnet = ResNet34(network_type="ImageNet", num_classes=num_classes, att_type='CBAM')
        self.batch_size = batch_size
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.resnet(x)

