
import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from torchvision import models
from copy import deepcopy
from typing import List, Dict

# available torchvision models
tvmodels = [
    'alexnet',
    'densenet121', 'densenet169', 'densenet201', 'densenet161',
    'googlenet',
    'inception_v3',
    'mobilenet_v2',
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2'
    'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0',
    'squeezenet1_0', 'squeezenet1_1',
    'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19',
]

allmodels = tvmodels

def set_tvmodel_head_var(model):
    if type(model) == models.AlexNet:
        model.head_var = 'classifier'
    elif type(model) == models.DenseNet:
        model.head_var = 'classifier'
    elif type(model) == models.Inception3:
        model.head_var = 'fc'
    elif type(model) == models.ResNet:
        model.head_var = 'fc'
    elif type(model) == models.VGG:
        model.head_var = 'classifier'
    elif type(model) == models.GoogLeNet:
        model.head_var = 'fc'
    elif type(model) == models.MobileNetV2:
        model.head_var = 'classifier'
    elif type(model) == models.ShuffleNetV2:
        model.head_var = 'fc'
    elif type(model) == models.SqueezeNet:
        model.head_var = 'classifier'
    else:
        raise ModuleNotFoundError


# class ProxyModel(nn.Module):

#     def __init__(self, model: "CLNN") -> None:
#         super().__init__()
#         self.model = model
    

class CLNN(nn.Module):
    """
    Continual Learning (CL) network
    """

    def __init__(self, model: nn.Module, remove_existing_head=True):
        super(CLNN, self).__init__()
        last_layer = getattr(model, model.head_var)
        self.feature_extractor = model

        if remove_existing_head:
            if type(last_layer) == nn.Sequential:
                self.out_size = last_layer[-1].in_features
                # strips off last linear layer of classifier
                del last_layer[-1]
            elif type(last_layer) == nn.Linear:
                self.out_size = last_layer.in_features
                setattr(self.feature_extractor, model.head_var, nn.Sequential())
        else:
            self.out_size = last_layer.out_features

        self.classifiers = nn.ModuleList()
        self.task_cls = []
        self.task_offset = []

    def add_head(self, num_outputs: int):
        self.classifiers.append(nn.Linear(self.out_size, num_outputs))
        self.task_cls = torch.tensor([head.out_features for head in self.classifiers])
        self.task_offset = torch.cat([torch.LongTensor(1).zero_(), self.task_cls.cumsum(0)[:-1]])

    def freeze_all(self):
        for param in self.parameters():
            param.requires_grad = False

    def activate_all(self):
        for param in self.parameters():
            param.requires_grad = True

    def freeze_feature_extractor(self):
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def activate_feature_extractor(self):
        for param in self.feature_extractor.parameters():
            param.requires_grad = True

    def freeze_classifier(self):
        for param in self.classifiers.parameters():
            param.requires_grad = False

    def activate_classifier(self):
        for param in self.classifiers.parameters():
            param.requires_grad = True

    def freeze_bn(self):
        for m in self.feature_extractor.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, inputs, return_features=True, concat=False):
        x = self.feature_extractor(inputs)
        assert (len(self.classifiers) > 0), "no classifiers"
        y = []
        for head in self.classifiers:
            y.append(head(x))
        if concat:
            y = torch.cat(y, dim=1)
        if return_features:
            return y, x
        else:
            return y


def get_network(name: str, pretrained=True, network_type="CLNN"):

    if network_type == "CLNN":
        if name in tvmodels:
            Network = getattr(importlib.import_module(name='torchvision.models'), name)
            network = Network(pretrained=pretrained)
            set_tvmodel_head_var(network)
            return CLNN(network)
        else:
            raise NotImplementedError("CLNN")
    else:
        raise NotImplementedError("network_type")


class CNNTest(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, (5,5))
        self.conv2 = nn.Conv2d(6, 16, (5,5))
        self.fc1   = nn.Linear(256, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
