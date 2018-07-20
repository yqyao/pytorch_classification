import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import argparse
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
from utils import TrainTransform, ValTransform
from data import MyDataset
from data.config import mydataset as cfg
from models.model_builder import model_builder
import numpy as np
import time
import os 
import sys
import json
from utils import transforms


def arg_parse():

    parser = argparse.ArgumentParser(
        description='Mydataset classification')
    parser.add_argument('-v', '--version', default='resnet50',
                        help='')
    parser.add_argument('-b', '--batch_size', default=16,
                    type=int, help='Batch size for training')
    parser.add_argument('--num_workers', default=4,
                    type=int, help='Number of workers used in dataloading')
    parser.add_argument('--weights', '--weights',default="./weights/resnet50_best_model.pth", help='weights path')
    args = parser.parse_args()

    return args

def main():
    args = arg_parse()
    model_name = args.version
    weights = args.weights
    num_classes = cfg["num_classes"]
    img_hw = cfg["img_hw"]
    bgr_means = cfg["bgr_means"]
    net = model_builder(model_name, pretrained=False, num_classes=num_classes)
    net.cuda()
    state_dict = torch.load(weights)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)
    test_transform = ValTransform(img_hw, bgr_means, padding=False)
    # test_transform = transforms.Compose(
    #     [
    #     transforms.Resize((384, 384)),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    #      ])
    test_dataset = MyDataset(cfg["Testroot"], 'test', test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            num_workers=
                                            args.num_workers)
    result = test_net(test_loader, net)
    print(result)

def test_net(test_loader, net):
    net.eval()
    result = np.array(list())
    for idx, (img, target) in enumerate(test_loader):
        with torch.no_grad():
            img = img.cuda()
            output = net(img)
            _, pred = output.topk(1, 1, True, True)
            result = np.append(result, pred.cpu().numpy())
    return result

if __name__ == '__main__':
    main()
