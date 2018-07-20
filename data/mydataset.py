# write by yqyao
# 
import os
import pickle
import os.path
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image

class MyDataset(data.Dataset):

    def __init__(self, root, phase, tarnsform=None, target_transform=None,
                 dataset_name='MyDataset'):
        self.root = root
        self.phase = phase
        self.tarnsform = tarnsform
        self.target_transform = target_transform
        self.name = dataset_name
        self.images_targets = list()
        data_list = os.path.join(root, self.phase + '.txt')
        with open(data_list ,"r") as f:
            for line in f.readlines():
                line = line.strip().split()
                if self.phase == 'test':
                    self.images_targets.append((line[0], 0))
                else:    
                    self.images_targets.append((line[0], int(line[1])))

    def __getitem__(self, index):
        img_id = self.images_targets[index]
        target = img_id[1]
        path = img_id[0]
        # img = Image.open(path).convert('RGB')
        img = cv2.imread(path)

        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.tarnsform is not None:
            img = self.tarnsform(img)
        return img, target

    def __len__(self):
        return len(self.images_targets)
     
