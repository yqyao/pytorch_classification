import os
import sys
import cv2
from multiprocessing.dummy import Pool as ThreadPool
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

voc_dict = {
    'aeroplane' : 0,
    'bicycle' : 1,
    'bird' : 2,
    'boat' : 3,
    'bottle' : 4,
    'bus' : 5,
    'car' : 6,
    'cat' : 7,
    'chair' : 8,
    'cow' : 9,
    'diningtable' : 10,
    'dog' : 11,
    'horse' : 12,
    'motorbike' : 13,
    'person' : 14,
    'pottedplant' : 15,
    'sheep' : 16,
    'sofa' : 17,
    'train' : 18,
    'tvmonitor' : 19
}

crop_label = list()

def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)

    return objects

def crop_img(image_id):
    img_folder = "/localSSD/yyq/VOCdevkit0712/VOC0712/JPEGImages"
    anno_folder = "/localSSD/yyq/VOCdevkit0712/VOC0712/Annotations"
    img_save_folder = "/localSSD/yyq/VOCdevkit0712/VOC0712/class/images"
    img_path = os.path.join(img_folder, image_id+".jpg")
    anno_path = os.path.join(anno_folder, image_id+".xml")
    im = cv2.imread(img_path)
    im_w, im_h = im.shape[1], im.shape[0]
    anno_bbox = parse_rec(anno_path)
    for idx, obj_bbox in enumerate(anno_bbox):
        bbox = obj_bbox['bbox']
        cls = obj_bbox['name']
        xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
        xmin, ymin = max(0, xmin), max(0, ymin)
        xmax, ymax = min(xmax, im_w), min(ymax, im_h)
        w, h = xmax - xmin, ymax - ymin
        if max(w, h) > 100:
            crop_img = im[ymin:ymax, xmin:xmax, :]
            crop_img_name = os.path.join(img_save_folder, image_id+"_{}_{}.jpg".format(str(cls), str(idx)))
            cv2.imwrite(crop_img_name, crop_img)
            crop_label.append((crop_img_name, cls))

image_id_list = list()
with open("/localSSD/yyq/VOCdevkit0712/VOC0712/ImageSets/Main/0712_trainval.txt", "r") as f:
    for i in f.readlines():
        image_id_list.append(i.strip())

pool = ThreadPool(processes=20)
pool.map(crop_img, image_id_list)
pool.close()
pool.join()    

with open("cls_train.txt", "w") as f:
    for item in crop_label:
        print(item[0], voc_dict[item[1]], file=f)   
