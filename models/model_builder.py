import os
import torch
from models.resnet import resnet50, resnet101, resnet152
from models.densenet import densenet121, densenet161
from models.senet import senet154, se_resnext101_32x4d, se_resnet101
from models.inception_v4 import inceptionv4
from models.xception import xception
from models.inceptionresnetv2 import inceptionresnetv2
from models.vgg import vgg16

net_dict = {
            "vgg16" : vgg16,
            "resnet50" : resnet50, 
            "resnet101" : resnet101, 
            "resnet152" : resnet152,
            "densenet121" : densenet121,
            "densenet161" : densenet161, 
            "inceptionv4" : inceptionv4,
            "senet154" : senet154, 
            "se_resnet101" : se_resnet101,
            "se_resnext101_32x4d" : se_resnext101_32x4d,
            "xception" : xception, 
            "inceptionresnetv2" : inceptionresnetv2 }

def model_builder(net_name, num_classes=100, pretrained=None, weight_path=None):
    if net_name not in net_dict:
        return None
    net = net_dict[net_name](num_classes=num_classes, pretrained=pretrained)
    if pretrained:
        load_weights(net, weight_path, net_name)
    return net

def load_weights(net, weight_path, net_name):

    if net_name == "vgg16":
        weight = torch.load(weight_path)
        net.vgg.load_state_dict(weight)
        return

    if net_name == "inceptionv4" or net_name == "inceptionresnetv2":
        weight = torch.load(weight_path)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in weight.items():
            head = k.split('.')[0]
            if head == 'last_linear':
                pass
            else:
                name = k
                new_state_dict[name] = v
        net.load_state_dict(new_state_dict, strict=False)
    else:
        weight = torch.load(weight_path)
        n = len(weight)
        count = 0
        print("load pretrained weight: ", weight_path, n)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in weight.items():
            if count < n-2:
                new_state_dict[k] = v
            else:
                pass
            count += 1
        net.load_state_dict(new_state_dict, strict=False)    