import torch
from torchvision import transforms
import cv2
import numpy as np
import random
import math


def resize_image(image, width, height, mean, padding=False):
    interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
    # interp_method = interp_methods[random.randrange(5)]
    interp_method = interp_methods[0]
    if padding == False:
        image = cv2.resize(image, (width, height), interpolation=interp_method)
        image = image.astype(np.float32)
        image -= mean
    else:
        max_size = max(width, height)
        im_width, im_height = image.shape[1], image.shape[0]
        ratio = im_width / im_height
        if im_width > im_height:
            resize_height = int(max_size / ratio)
            image = cv2.resize(image, (max_size, resize_height), interpolation=interp_method)
            image = image.astype(np.float32)
            image -= mean
            #padding with zeros
            padding_im = np.zeros((max_size-resize_height, max_size, 3))
            image = np.concatenate((image, padding_im), axis=0)
        else:
            resize_width = int(max_size * ratio)
            image = cv2.resize(image, (resize_width, max_size), interpolation=interp_method)
            image = image.astype(np.float32)
            image -= mean
            #padding
            padding_im = np.zeros((max_size, max_size-resize_width, 3))
            image = np.concatenate((image, padding_im), axis=1)
    return image

def preproc_for_test(image, insize, mean, padding=False):
    interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
    # interp_method = interp_methods[random.randrange(5)]
    interp_method = interp_methods[0]
    width, height = insize[1], insize[0]
    # image = cv2.resize(image, (width, height), interpolation=interp_method)
    image = resize_image(image, width, height, mean, padding)
    # to rgb
    image = image[:, :, (2, 1, 0)]
    return image.transpose(2, 0, 1)


def _crop(image):
    height, width, _ = image.shape
    if random.randrange(2):
        scale = 0.9
        w = int(scale * width)
        h = int(scale * height)
        l = random.randrange(width - w)
        t = random.randrange(height - h)
        roi = np.array((l, t, l + w, t + h))
        image_t = image[roi[1]:roi[3], roi[0]:roi[2]]
        return image_t
    else:
        return image


def _distort(image):
    def _convert(image, alpha=1, beta=0):
        tmp = image.astype(float) * alpha + beta
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        image[:] = tmp

    image = image.copy()

    if random.randrange(2):
        _convert(image, beta=random.uniform(-32, 32))

    if random.randrange(2):
        _convert(image, alpha=random.uniform(0.5, 1.5))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    if random.randrange(2):
        tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
        tmp %= 180
        image[:, :, 0] = tmp

    if random.randrange(2):
        _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image

def _mirror(image):
    _, width, _ = image.shape
    if random.randrange(2):
        image = image[:, ::-1]
    return image


class TrainTransform(object):

    def __init__(self, resize, bgr_means, padding=False):
        self.means = bgr_means
        self.resize = resize
        self.padding = padding

    def __call__(self, image):

        image = _distort(image)
        image = _mirror(image)
        image = _crop(image)
        image = preproc_for_test(image, self.resize, self.means, self.padding)

        return torch.from_numpy(image)

class ValTransform(object):
    """Defines the transformations that should be applied to test PIL image
        for input into the network

    dimension -> tensorize -> color adj

    Arguments:
        resize (int): input dimension to SSD
        rgb_means ((int,int,int)): average RGB of the dataset
            (104,117,123)
        swap ((int,int,int)): final order of channels
    Returns:
        transform (transform) : callable transform to be applied to test/val
        data
    """
    def __init__(self, resize, rgb_means, padding=False):
        self.means = rgb_means
        self.resize = resize
        self.padding = padding

    # assume input is cv2 img for now
    def __call__(self, img, targets=None):
        img = preproc_for_test(img, self.resize, self.means, self.padding)
        return torch.from_numpy(img)
