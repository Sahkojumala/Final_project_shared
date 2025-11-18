
#
from args import get_args
import torch
import torch.nn as nn
import torchvision.models as models

class MyModel(nn. Module) : 
    def __init__ (self, backbone='resnet18') :
        super(MyModel, self) .__init__ ()
        args = get_args()
        
        if backbone == 'resnet18':
            self.model = models.resnet18(weights='IMAGENET1K_V1')
        elif backbone == 'resnet34':
            self.model = models.resnet34(weights='IMAGENET1K_V1')
        else:
            self.model = models.resnet50(weights='IMAGENET1K_V1')

        # korvaa viimeinen kerros
        self.model.fc = nn.Linear(self.model.fc.in_features, args.num_classes)

    def forward(self, x):
        return self.model(x)


