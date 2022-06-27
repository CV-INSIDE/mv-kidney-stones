import helpers.transferLearningBaseModel as tlm
import torchvision.models as models
import torch.nn as nn

class Vgg16Model(tlm.BaseModel):
  def __init__(self, hparams={}, num_classes=6, batch_size=64, pretrained=False, seed=None):
    if "lr" not in hparams:
      hparams["lr"] = 0.001

    #LOG INFO
    hparams["num_classes"] = num_classes
    hparams["batch_size"] = batch_size
    hparams["is_pretrained"] = pretrained
    super(Vgg16Model, self).__init__(hparams, seed=seed)
    
    self.vgg16 = models.vgg16(pretrained=pretrained)
    self.vgg16.classifier = nn.Sequential(nn.Linear(25088, 4096), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(4096, 256), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(256, num_classes))
    self.batch_size = batch_size
    self.loss_fn = nn.CrossEntropyLoss()

  def forward(self, x):
    return self.vgg16(x)