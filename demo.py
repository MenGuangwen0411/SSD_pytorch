#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
@File: demo.py
@Author:kong
@Time: 2020年01月21日09时40分
@Description:
'''
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from matplotlib import pyplot as plt
from data import VOCDetection, VOC_ROOT, VOCAnnotationTransform
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import cv2

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
# from data import VOC_CLASSES as labels
from data.bdd100k import BDD_CLASSES as labels
from ssd import build_ssd

# d:\datasets\BDD100K\bdd100k\images\100k\train\0000f77c-6257be58.jpg
# d:\datasets\BDD100K\bdd100k\images\100k\train\0000f77c-62c2a288.jpg
# d:\datasets\BDD100K\bdd100k\images\100k\train\0000f77c-cb820c98.jpg
# d:\datasets\BDD100K\bdd100k\images\100k\train\0001542f-5ce3cf52.jpg
# d:\datasets\BDD100K\bdd100k\images\100k\train\0001542f-7c670be8.jpg
# d:\datasets\BDD100K\bdd100k\images\100k\train\0001542f-ec815219.jpg
# d:\datasets\BDD100K\bdd100k\images\100k\train\0004974f-05e1c285.jpg
image_path = r'd:\datasets\BDD100K\bdd100k\images\100k\train\0000f77c-6257be58.jpg'
weight_path = 'weights/SSD512_BDD100K_0000001000.pth'
model_input = 512

net = build_ssd('test', 'VGG16',model_input, 11)  # initialize SSD
net.load_weights(weight_path)
image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # uncomment if dataset not downloaded
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
x = cv2.resize(image, (model_input, model_input)).astype(np.float32)
x -= (104.0, 117.0, 123.0)
x = x.astype(np.float32)
x = x[:, :, ::-1].copy()
x = torch.from_numpy(x).permute(2, 0, 1)

xx = Variable(x.unsqueeze(0))  # wrap tensor in Variable
if torch.cuda.is_available():
    xx = xx.cuda()
y = net(xx)

top_k = 10
detections = y.data
# scale each detection back up to the image
scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)  # 4个尺度的缩放系数
for i in range(detections.size(1)):  # 遍历num_class
    j = 0
    while detections[0, i, j, 0] >= 0.2:
        score = detections[0, i, j, 0]
        label_name = labels[i - 1]
        display_txt = '%s: %.2f' % (label_name, score)
        pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
        j += 1
        image = cv2.rectangle(image, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (255, 0, 0), 2)
        image = cv2.putText(image, display_txt, (int(pt[2]), int(pt[1])), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
cv2.imshow('detection', image)
cv2.waitKey()
# cv2.imwrite('./test/resut.jpg', image)
