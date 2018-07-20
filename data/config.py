import os


pretrained_dict = {
    "vgg16"    : "./weights/pretrained_models/vgg16_reducedfc.pth",
    "resnet50" : "./weights/pretrained_models/resnet50-19c8e357.pth",
    "resnet101" : "./weights/pretrained_models/resnet101-5d3b4d8f.pth",
    "resnet152" : "./weights/pretrained_models/resnet152-b121ed2d.pth",
    "densenet121" : "./weights/pretrained_models/densenet121-a639ec97.pth",
    "densenet161" : "./weights/pretrained_models/densenet161-17b70270.pth",
    "senet154" : "./weights/pretrained_models/senet154-c7b49a05.pth",
    "se_resnet101" : "./weights/pretrained_models/se_resnet101-7e38fcc6.pth",
    "inceptionv4" : "./weights/pretrained_models/inceptionv4-8e4777a0.pth",
    "se_resnext101_32x4d" : "./pretrained_models/weights/se_resnext101_32x4d-3b2fe3d8.pth",
    "xception" : "./weights/pretrained_models/xception-b0b7af25.pth",
    "inceptionresnetv2" : "./weights/pretrained_models/inceptionresnetv2-520b38e4.pth"
}


mydataset = {
    'Trainroot' : "./data/datasets/baidu/set/",
    'Valroot' : "./data/datasets/baidu/set/",
    'pretrained_dict' : pretrained_dict,
    'bgr_means' : (104, 117, 123),
    'img_hw' : (384, 384),
    'start_epoch' : 0,
    'end_epoch' : 15,
    'epoch_step' : [0, 9, 12],
    'save_folder' : './weights/',
    'num_classes' : 61,
    'Testroot' : "./data/datasets/baidu/set/"
}

voc = {
    'Trainroot' : "./data/datasets/voc/set/",
    'Valroot' : "./data/datasets/voc/set/",
    'pretrained_dict' : pretrained_dict,
    'bgr_means' : (104, 117, 123),
    'img_hw' : (300, 300),
    'start_epoch' : 0,
    'end_epoch' : 16,
    'epoch_step' : [0, 9, 14],
    'save_folder' : './weights/',
    'num_classes' : 20,
    'Testroot' : "./data/datasets/voc/set/"
}
