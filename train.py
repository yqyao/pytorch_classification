import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
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
from data.config import voc as cfg
from models.model_builder import model_builder
import numpy as np
import time
import os 
import sys
import json
import shutil
from PIL import Image
from utils import AverageMeter
from utils import transforms
# import setproctitle
# setproctitle.setproctitle("python")

def arg_parse():

    parser = argparse.ArgumentParser(
        description='Mydataset classification')
    parser.add_argument('-v', '--version', default='resnet50',
                        help='')
    parser.add_argument('-b', '--batch_size', default=32,
                        type=int, help='Batch size for training')
    parser.add_argument('--num_workers', default=4,
                        type=int, help='Number of workers used in dataloading')
    parser.add_argument('--cuda', default=True,
                        type=bool, help='Use cuda to train model')
    parser.add_argument('--lr', '--learning-rate',
                        default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--ngpu', default=2, type=int, help='gpus')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument(
        '--resume_net', default=None, help='resume net for retraining')
    parser.add_argument('--resume_epoch', default=0,
                        type=int, help='resume iter for retraining')
    parser.add_argument('--weight_decay', default=5e-4,
                        type=float, help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1,
                        type=float, help='Gamma update for SGD')
    parser.add_argument('--save_folder', default='./weights/',
                        help='Location to save checkpoint models')
    return parser.parse_args()

def adjust_learning_rate(optimizer, epoch, step_epoch, gamma, epoch_size, iteration):
    """Sets the learning rate 
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    ## warmup
    if epoch < 1:
        iteration += iteration * epoch
        lr = 1e-6 + (args.lr - 1e-6) * iteration / (epoch_size * 1) 
    else:
        div = 0
        if epoch >= step_epoch[-1]:
            div = len(step_epoch) - 1
        else:
            for idx, v in enumerate(step_epoch):
                if epoch >= step_epoch[idx] and epoch < step_epoch[idx+1]:
                    div = idx 
                    break
        lr = args.lr * (gamma ** div)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def train(train_loader, net, criterion, optimizer, epoch, epoch_step):
    net.train()
    begin = time.time()
    epoch_size = len(train_loader)
    for iteration, (img, target) in enumerate(train_loader):
        lr = adjust_learning_rate(optimizer, epoch, epoch_step, args.gamma, epoch_size, iteration)
        img = img.cuda()
        target = target.cuda()
        t0 = time.time()
        out = net(img)
        t1 = time.time()
        optimizer.zero_grad()
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        t2 = time.time()
        if iteration % 10 == 0:
            print("Epoch: {} | iter {}/{} loss: {} Time: {} lr: {}".format(str(epoch), str(iteration), str(epoch_size), str(round(loss.item(), 5)), str(round(t2 - t0, 5)), str(lr)))

def save_checkpoint(net, epoch, is_best):
    file_name = os.path.join(args.save_folder, args.version + "_epoch_{}".format(str(epoch))+ '.pth')
    torch.save(net.state_dict(), file_name)
    if is_best:
        shutil.copyfile(file_name, os.path.join(args.save_folder, args.version+"_best_model.pth"))


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res_app = correct_k.mul_(100.0 / batch_size)
        res.append(res_app.cpu().numpy())
    return res

def eval_net(val_loader, net, criterion):
    prec_sum = 0
    for idx, (img, target) in enumerate(val_loader):
        with torch.no_grad():
            target = target.cuda()
            img = img.cuda()
            img_var = torch.autograd.Variable(img)
            target_var = torch.autograd.Variable(target)  
            output = net(img_var)
            loss = criterion(output, target_var)
            prec, _ = accuracy(output.data, target, topk=(1, 1))  
            prec_sum += prec        
    return prec_sum, loss

def main():
    global args 
    args = arg_parse()
    save_folder = args.save_folder
    weight_decay = args.weight_decay
    gamma = args.gamma
    momentum = args.momentum
    cuda = args.cuda
    model_name = args.version
    lr = args.lr
    bgr_means = cfg['bgr_means']
    img_hw = cfg['img_hw']
    Trainroot = cfg["Trainroot"]
    Valroot = cfg["Valroot"]
    epoch_step = cfg['epoch_step']
    start_epoch = cfg['start_epoch']
    end_epoch = cfg['end_epoch']
    pretrained_model = cfg["pretrained_dict"][model_name]
    num_classes = cfg['num_classes']
    net = model_builder(model_name, pretrained=True, weight_path=pretrained_model, num_classes=num_classes)
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    if args.cuda and torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    if args.ngpu > 1:
        net = torch.nn.DataParallel(net)
    if args.cuda:
        net.cuda()
        cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(net.parameters(), lr=lr,
                          momentum=momentum, weight_decay=weight_decay)

    train_transform = TrainTransform(img_hw, bgr_means, padding=False)
    val_transform = ValTransform(img_hw, bgr_means, padding=False)

    # train_transform = transforms.Compose(
    #     [
    #     transforms.Resize((384,384)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomCrop((345,345)),
    #     transforms.RandomRotation((-15,15)),
    #     transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    #      ])
    # val_transform = transforms.Compose(
    #     [
    #     transforms.Resize((384,384)),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    #      ])
    val_dataset = MyDataset(Valroot, 'val', val_transform)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            num_workers=
                                            args.num_workers)
    best_prec = 0
    for epoch in range(start_epoch, end_epoch):
        train_dataset = MyDataset(Trainroot, 'train', train_transform)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=args.batch_size,
                                                    shuffle=True,
                                                    num_workers=args.num_workers)
        train(train_loader, net, criterion, optimizer, epoch, epoch_step)
        prec, loss = eval_net(val_loader, net, criterion)
        prec /= len(val_loader)
        is_best = prec > best_prec
        best_prec = max(best_prec, prec)
        save_checkpoint(net, epoch, is_best)
        print("current accuracy:", prec, "best accuracy:", best_prec)   

if __name__ == '__main__':
    main()

