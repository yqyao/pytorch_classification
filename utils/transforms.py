from __future__ import absolute_import

from torchvision.transforms import *

import numpy as np
import torch
import random
import math

class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al. 
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''
    def __init__(self, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
       
    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
       
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size()[2] and h <= img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                else:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                return img

        return img

class Rotation(object):
    def __init__(self, degree):
        self.degree = degree
       
    def __call__(self, img):
        img = img.rotate(self.degree)
        return img

class Crop_img(object):
    def __init__(self, the_type, size):
        self.type = the_type
        self.size = size
       
    def __call__(self, img):
        origin_size = img.size[1]
        offset = origin_size - self.size
        half_offset = offset/2
        crop_img = img
        if self.type == 'Center':
            area = (half_offset, half_offset, origin_size-half_offset, origin_size-half_offset)
            crop_img = img.crop(area)
        elif self.type == 'LeftTop':
            area = (0, 0, self.size, self.size)
            crop_img = img.crop(area)
        elif self.type == 'RightDown':
            area = (offset, offset, origin_size, origin_size)
            crop_img = img.crop(area)
        elif self.type == 'LeftDown':
            area = (0, offset, self.size, origin_size)
            crop_img = img.crop(area)
        elif self.type == 'RightTop':
            area = (offset, 0, origin_size, self.size)
            crop_img = img.crop(area)
        return crop_img