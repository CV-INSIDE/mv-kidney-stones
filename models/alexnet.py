import helpers.transferLearningBaseModel_original as tlm
import torchvision.models as models
import torch.nn as nn
import torch

class AlexnetModel(tlm.BaseModel):
    def __init__(self, hparams={}, num_classes=6, batch_size=64, pretrained=False, seed=None):
        if "lr" not in hparams:
            hparams["lr"] = 0.001

        #LOG INFO
        hparams["num_classes"] = num_classes
        hparams["batch_size"] = batch_size
        hparams["is_pretrained"] = pretrained
        super(AlexnetModel, self).__init__(hparams, seed=seed)

        self.alex = models.alexnet(pretrained=pretrained)
        # complete FC layer.
        self.alex.classifier = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(9216, 4096), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(4096, 256), nn.ReLU(), nn.Linear(256, num_classes))
        # Alternate FC layer to output the features of the image instead of of the probability of belonging go a class. The output
        # contains 256 features.

        #self.alex.classifier = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(9216, 4096), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(4096, 256), nn.ReLU())
        #self.alex.classifier_2 = nn.Sequential(nn.Linear(256, num_classes+1)) # still trying to find out why it works with +1
        self.batch_size = batch_size
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.alex(x)
