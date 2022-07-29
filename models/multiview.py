"""
Class that contains multi-view model structure
"""
import torch
from einops import reduce

import helpers.transferLearningBaseModel as tlm
import torchvision.models as models
import torch.nn as nn


class MultiViewConcatenate(tlm.BaseModel):
    def __init__(self, hparams={}, num_classes=6, batch_size=64, pretrained=False, seed=None):
        if "lr" not in hparams:
            hparams["lr"] = 0.001

        #LOG INFO
        hparams["num_classes"] = num_classes
        hparams["batch_size"] = batch_size
        hparams["is_pretrained"] = pretrained
        super(MultiViewConcatenate, self).__init__(hparams, seed=seed)

        self.alex = models.alexnet(pretrained=pretrained)
        # self.model2 = models.alexnet(pretrained=pretrained)
        # complete FC layer.
        for param1 in self.alex.parameters():
            param1.requires_grad = False

        self.alex.classifier = nn.Sequential(nn.Dropout(p=0.5),
                                        nn.Linear(18432, 4096),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.5),
                                        #nn.Linear(9216, 4096),
                                        #nn.ReLU(),
                                        #nn.Dropout(p=0.5),
                                        nn.Linear(4096, 256),
                                        nn.ReLU(),
                                        nn.Linear(256, num_classes))

        # Alternate FC layer to output the features of the image instead of of the probability of belonging go a class.
        # The output contains 256 features.

        #self.alex.classifier = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(9216, 4096), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(4096, 256), nn.ReLU())
        self.batch_size = batch_size
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, y):
        """
        Perform the forward step of multi view model. Fusion is done by concanetation of the values from differentes
        views. The first layer of the classifier will then be: number_views * samples
        The way it works is at follows:
            1) First, do a forward step by execution of features, avg pool and flatten. The result will be a tensor of
            size 1, 9216
            2) Concatenate the feature vectors of the different views and connec them to a fc layer
            3) Perform the classification task with RELU and dropout of 50%
            4) The last step consists on the connection to the fc last layer to a softmax activation (not here)
        :param x: batch of the first view
        :param y: batch of the second view
        :return: the features after forward propagation
        """
        x = self.alex.features(x)
        x = self.alex.avgpool(x)
        x = torch.flatten(x, 1)

        y = self.alex.features(y)
        y = self.alex.avgpool(y)
        y = torch.flatten(y, 1)

        xy = torch.cat((x, y), 1)
        xy = self.alex.classifier(xy)

        return xy


class MultiViewMaxPool(tlm.BaseModel):
    def __init__(self, hparams={}, num_classes=6, batch_size=64, pretrained=False, seed=None):
        if "lr" not in hparams:
            hparams["lr"] = 0.001

        #LOG INFO
        hparams["num_classes"] = num_classes
        hparams["batch_size"] = batch_size
        hparams["is_pretrained"] = pretrained
        super(MultiViewMaxPool, self).__init__(hparams, seed=seed)

        self.alex = models.alexnet(pretrained=pretrained)
        # self.model2 = models.alexnet(pretrained=pretrained)
        # complete FC layer.
        for param1 in self.alex.parameters():
            param1.requires_grad = False

        self.alex.classifier = nn.Sequential(nn.Dropout(p=0.5),
                                             nn.Linear(9216, 4096),
                                             nn.ReLU(),
                                             nn.Dropout(p=0.5),
                                             nn.Linear(4096, 256),
                                             nn.ReLU(),
                                             nn.Linear(256, num_classes))

        # Alternate FC layer to output the features of the image instead of of the probability of belonging go a class.
        # The output contains 256 features.

        #self.alex.classifier = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(9216, 4096), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(4096, 256), nn.ReLU())
        self.batch_size = batch_size
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, y):
        """
        Perform the forward step of multi view model. Fusion is done by concanetation of the values from differentes
        views. The first layer of the classifier will then be: number_views * samples
        The way it works is at follows:
            1) First, do a forward step by execution of features, avg pool and flatten. The result will be a tensor of
            size 1, 9216
            2) Concatenate the feature vectors of the different views and connec them to a fc layer
            3) Perform the classification task with RELU and dropout of 50%
            4) The last step consists on the connection to the fc last layer to a softmax activation (not here)
        :param x: batch of the first view
        :param y: batch of the second view
        :return: the features after forward propagation
        """
        x = self.alex.features(x)
        x = self.alex.avgpool(x)
        x = torch.flatten(x, 1)

        y = self.alex.features(y)
        y = self.alex.avgpool(y)
        y = torch.flatten(y, 1)

        # xy = torch.cat((x, y), 0)
        xy = torch.stack([x, y])
        xy = reduce(xy, 'c b l -> 1 b l', 'max')
        xy = torch.squeeze(xy, dim=0)
        # xy = reduce(xy, 'c l -> 1 l', 'max')
        xy = self.alex.classifier(xy)

        return xy


class MultiViewConcVGG16(tlm.BaseModel):
    def __init__(self, hparams={}, num_classes=6, batch_size=64, pretrained=False, seed=None):
        if "lr" not in hparams:
            hparams["lr"] = 0.001

        #LOG INFO
        hparams["num_classes"] = num_classes
        hparams["batch_size"] = batch_size
        hparams["is_pretrained"] = pretrained
        super(MultiViewConcVGG16, self).__init__(hparams, seed=seed)

        self.vgg16 = models.vgg16(pretrained=pretrained)
        # self.model2 = models.alexnet(pretrained=pretrained)
        # complete FC layer.
        for param1 in self.vgg16.parameters():
            param1.requires_grad = False

        self.vgg16.classifier = nn.Sequential(nn.Dropout(p=0.5),
                                              nn.Linear(25088*2, 25088),
                                              nn.ReLU(),
                                              nn.Dropout(p=0.5),
                                              nn.Linear(25088, 4096),
                                              nn.ReLU(),
                                              nn.Dropout(p=0.5),
                                              nn.Linear(4096, 256),
                                              nn.ReLU(),
                                              nn.Linear(256, num_classes))

        # Alternate FC layer to output the features of the image instead of of the probability of belonging go a class.
        # The output contains 256 features.

        #self.alex.classifier = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(9216, 4096), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(4096, 256), nn.ReLU())
        self.batch_size = batch_size
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, y):
        """
        Perform the forward step of multi view model. Fusion is done by concanetation of the values from differentes
        views. The first layer of the classifier will then be: number_views * samples
        The way it works is at follows:
            1) First, do a forward step by execution of features, avg pool and flatten. The result will be a tensor of
            size 1, 9216
            2) Concatenate the feature vectors of the different views and connec them to a fc layer
            3) Perform the classification task with RELU and dropout of 50%
            4) The last step consists on the connection to the fc last layer to a softmax activation (not here)
        :param x: batch of the first view
        :param y: batch of the second view
        :return: the features after forward propagation
        """
        x = self.vgg16.features(x)
        x = self.vgg16.avgpool(x)
        x = torch.flatten(x, 1)

        y = self.vgg16.features(y)
        y = self.vgg16.avgpool(y)
        y = torch.flatten(y, 1)

        xy = torch.cat((x, y), 1)
        xy = self.vgg16.classifier(xy)

        return xy


class MultiViewPoolVGG16(tlm.BaseModel):
    def __init__(self, hparams={}, num_classes=6, batch_size=64, pretrained=False, seed=None):
        if "lr" not in hparams:
            hparams["lr"] = 0.001

        #LOG INFO
        hparams["num_classes"] = num_classes
        hparams["batch_size"] = batch_size
        hparams["is_pretrained"] = pretrained
        super(MultiViewPoolVGG16, self).__init__(hparams, seed=seed)

        self.vgg16 = models.vgg16(pretrained=pretrained)
        # complete FC layer.
        for param1 in self.vgg16.parameters():
            param1.requires_grad = False

        self.vgg16.classifier = nn.Sequential(
                                              nn.Dropout(p=0.5),
                                              nn.Linear(25088, 4096),
                                              nn.ReLU(),
                                              nn.Dropout(p=0.5),
                                              nn.Linear(4096, 256),
                                              nn.ReLU(),
                                              nn.Linear(256, num_classes))

        # Alternate FC layer to output the features of the image instead of of the probability of belonging go a class.
        # The output contains 256 features.

        #self.alex.classifier = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(9216, 4096), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(4096, 256), nn.ReLU())
        self.batch_size = batch_size
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, y):
        """
        Perform the forward step of multi view model. Fusion is done by concanetation of the values from differentes
        views. The first layer of the classifier will then be: number_views * samples
        The way it works is at follows:
            1) First, do a forward step by execution of features, avg pool and flatten. The result will be a tensor of
            size 1, 9216
            2) Concatenate the feature vectors of the different views and connec them to a fc layer
            3) Perform the classification task with RELU and dropout of 50%
            4) The last step consists on the connection to the fc last layer to a softmax activation (not here)
        :param x: batch of the first view
        :param y: batch of the second view
        :return: the features after forward propagation
        """
        x = self.vgg16.features(x)
        x = self.vgg16.avgpool(x)
        x = torch.flatten(x, 1)

        y = self.vgg16.features(y)
        y = self.vgg16.avgpool(y)
        y = torch.flatten(y, 1)

        # xy = torch.cat((x, y), 0)
        xy = torch.stack([x, y])
        xy = reduce(xy, 'c b l -> 1 b l', 'max')
        xy = torch.squeeze(xy, dim=0)
        # xy = reduce(xy, 'c l -> 1 l', 'max')
        xy = self.vgg16.classifier(xy)

        return xy
