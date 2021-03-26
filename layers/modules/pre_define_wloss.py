'''
@Author: Jeffery Sheng (Zhenfei Sheng)
@Time:   2019/12/15 17:23
@File:   pre_define_wloss.py
'''

import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Pre_Define_Wloss_SSD(nn.Module):
    def __init__(self):
        super(Pre_Define_Wloss_SSD, self).__init__()
    # pre define D matrix
    # Specifically suit to BDD100K dataset, got 10(forward ground) + 1(background) classes
        self.D = torch.tensor(
        [[0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
         [1., 0., .1, .1, .2, .2, .2, .2, .2, .2, .2],
         [1., .1, 0., .1, .2, .2, .2, .2, .2, .2, .2],
         [1., .1, .1, 0., .2, .2, .2, .2, .2, .2, .2],
         [1., .2, .2, .2, 0., .1, .1, .2, .2, .2, .2],
         [1., .2, .2, .2, .1, 0., .1, .2, .2, .2, .2],
         [1., .2, .2, .2, .1, .1, 0., .2, .2, .2, .2],
         [1., .2, .2, .2, .2, .2, .2, 0., .2, .2, .2],
         [1., .2, .2, .2, .2, .2, .2, .2, 0., .1, .2],
         [1., .2, .2, .2, .2, .2, .2, .2, .1, 0., .2],
         [1., .2, .2, .2, .2, .2, .2, .2, .2, .2, 0.]], dtype=torch.float32, device='cuda')
    def forward(self, predict, label):
        # trans predict to softmax format
        pred_softmax = F.softmax(predict, dim=0)
        # trans label to onehot
        label_onehot = F.one_hot(label, num_classes=11)

        # cal wloss
        D_matrix = self.D[label]  # shape:(batch_size, num_classes)
        W = pred_softmax - label_onehot
        W[W < 0] = 0.

        wloss = torch.sum(torch.matmul(D_matrix, W.t()).mean(0))

        return wloss


class Pre_Define_Wloss_YOLOV3(nn.Module):
    def __init__(self):
        super(Pre_Define_Wloss_YOLOV3, self).__init__()

        self.D = torch.tensor(
        [[0., .1, .1, .2, .2, .2, .3, .3, .3, .2],
         [.1, 0., .1, .2, .2, .2, .3, .3, .3, .2],
         [.1, .1, 0., .3, .3, .3, .3, .3, .3, .3],
         [.2, .2, .3, 0., .1, .1, .3, .3, .3, .3],
         [.2, .2, .3, .1, 0., .1, .3, .3, .3, .3],
         [.2, .2, .3, .1, .1, 0., .3, .3, .3, .3],
         [.3, .3, .3, .3, .3, .3, 0., .3, .3, .3],
         [.3, .3, .3, .3, .3, .3, .3, 0., .1, .3],
         [.3, .3, .3, .3, .3, .3, .3, .1, 0., .3],
         [.2, .2, .3, .3, .3, .3, .3, .3, .3, 0.]], dtype=torch.float32, device='cuda')



    def forward(self, predict, label):
        # trans predict to softmax format
        pred_softmax = F.softmax(predict, dim=0)
        # trans label to onehot
        label_onehot = label
        # trans onehot back
        label_ori = torch.argmax(label_onehot, dim=1)
        # cal wloss
        D_matrix = self.D[label_ori]  # shape:(batch_size, num_classes)
        W = pred_softmax - label_onehot
        W[W < 0] = 0.
        # import pdb;pdb.set_trace()
        wloss = torch.sum(torch.matmul(D_matrix, W.t()).mean(0))

        return wloss

if __name__ == '__main__':
    pass
    # out = F.softmax(a)
    # label = torch.zeros((a.size(0), a.size(1)))
    # for i in range(l.size(0)):
    #     label[i][int(l[i].item())] = 1
    # print(out)
    # print(out.size())
    # print(label)
    # print(label.size())
