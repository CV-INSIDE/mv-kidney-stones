"""
Class that contains multi-view model structure for alexnet using three different color channels
"""
import torch
from einops import reduce

import helpers.transfer_learning_basemodel_mv_custom as tlm
import torchvision.models as models
import torch.nn as nn

class AlexnetMax(tlm.BaseModel):
    def __init__(self, hparams={}, num_classes=6, batch_size=64, pretrained=False, seed=None):
        if "lr" not in hparams:
            hparams["lr"] = 0.001

        #LOG INFO
        hparams["num_classes"] = num_classes
        hparams["batch_size"] = batch_size
        hparams["is_pretrained"] = pretrained
        super(AlexnetMax, self).__init__(hparams, seed=seed)

        self.alex = models.alexnet(pretrained=pretrained)
        # freeze fully connected layers
        for param in self.alex.parameters():
            param.requires_grad = False

        # create the fully connected classifier
        self.alex.classifier = nn.Sequential(nn.Dropout(p=0.5),
                                             nn.Linear(9216, 4096),
                                             nn.ReLU(),
                                             nn.Dropout(p=0.5),
                                             nn.Linear(4096, 256),
                                             nn.ReLU(),
                                             nn.Linear(256, num_classes))

        # define batch size and loss function
        self.batch_size = batch_size
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, y, z):
        """
        Perform the forward step of multi view model. Fusion is done by concatenation of the values from different
        views. The first layer of the classifier will then be: number_views * samples
        The way it works is at follows:
            1) First, do a forward step by execution of features, avg pool and flatten. The result will be a tensor of
            size 1, 9216
            2) Concatenate the feature vectors of the different views and connect them to a fc layer
            3) Perform the classification task with RELU and dropout of 50%
            4) The last step consists on the connection to the fc last layer to a softmax activation (not here)
        :param x: batch of the first view
        :param y: batch of the second view
        :param z: batch of the third view
        :return: the features after forward propagation
        """
        x = self.alex.features(x)
        x = self.alex.avgpool(x)
        x = torch.flatten(x, 1)

        y = self.alex.features(y)
        y = self.alex.avgpool(y)
        y = torch.flatten(y, 1)

        z = self.alex.features(z)
        z = self.alex.avgpool(z)
        z = torch.flatten(z, 1)

        # xy = torch.cat((x, y), 0)
        xyz = torch.stack([x, y, z])
        xyz = reduce(xyz, 'c b l -> 1 b l', 'max')
        xyz = torch.squeeze(xyz, dim=0)
        xyz = self.alex.classifier(xyz)

        return xyz
