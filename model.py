import torch
import torch.nn as nn
from torchvision import models


class Model:

    @staticmethod
    def efficientnet(num_classes):
        model = models.efficientnet_b3(
            pretrained=True
        )
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(
            in_features=in_features, out_features=num_classes)
        return model
