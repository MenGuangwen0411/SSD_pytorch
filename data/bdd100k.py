"""BDD Dataset Classes
USE THIS CODE FOR BDD100K(YOLO V3 FORMAT) TRAINING AND TESTING
Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py
Borrow from Ellis Brown, Max deGroot version.
writen by Men Guangwen
2021.2.7

"""
from .config import HOME
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import os

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

BDD_CLASSES = (  # always index 0
    'bike',
    'bus',
    'car',
    'motor',
    'person',
    'rider',
    'traffic light',
    'traffic sign',
    'train',
    'truck')

# note: if you used our download scripts, this should be right
BDD_ROOT = osp.join(HOME, "BDD100K")


class BDD100KDetection(data.Dataset):
    def __init__(self, root,
                 image_sets='train',
                 transform=None,
                 dataset_name='bdd100k'):
        if image_sets == 'train':
            list_path = 'train.txt'
        else:
            list_path = 'val.txt'
        self.root = root
        self.image_path_list = os.path.join(self.root, list_path)
        self.transform = transform
        self.name = dataset_name
        self.ids_array = np.loadtxt(self.image_path_list, str).reshape(-1, 1)  # list for image of
        # print()

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)
        # print(len(gt))

        return im, gt

    def __len__(self):
        return len(self.ids_array)

    def pull_item(self, index):
        img_id = self.ids_array[index][0]
        target = np.loadtxt(img_id.replace('images', 'labels').replace('.jpg', '.txt')).reshape(-1, 5)

        img = cv2.imread(img_id)
        height, width, channels = img.shape
        target_ = target.copy()  # x,y,w,h
        target[:, 0] = (target_[:, 1] - target_[:, 3] / 2)  # x1
        target[:, 2] = (target_[:, 1] + target_[:, 3] / 2)  # x2
        target[:, 1] = (target_[:, 2] - target_[:, 4] / 2)  # y1
        target[:, 3] = (target_[:, 2] + target_[:, 4] / 2)  # y2
        target[:, 4] = (target_[:, 0]).astype(np.int)  # 0 is background ??
        if 1:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width


if __name__ == '__main__':
    image_path_list = r'd:\datasets\BDD100K\train.txt'
    c = np.loadtxt(image_path_list, str).reshape(-1, 1)  # list for image of
    print()
