# config.py
import os.path

# gets home dir cross platform
# HOME = os.path.expanduser("~")

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

MEANS = (104, 117, 123)
#
# voc = {
#     'SSD300': {
#         'num_classes': 21,
#         'lr_steps': (100, 180, 250),
#         'max_iter': 520000,
#         'feature_maps': [38, 19, 10, 5, 3, 1],
#         'min_dim': 300,
#         'steps': [8, 16, 32, 64, 100, 300],
#         'min_sizes': [30, 60, 111, 162, 213, 264],
#         'max_sizes': [60, 111, 162, 213, 264, 315],
#         'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
#         'variance': [0.1, 0.2],
#         'clip': True,
#         'name': 'VOC'
#     },
#     'SSD512': {
#         'num_classes': 21,
#         'lr_steps': (100, 200, 300),
#         'max_iter': 120000,
#         'feature_maps': [64, 32, 16, 8, 4, 2, 1],
#         'min_dim': 512,
#         'steps': [8, 16, 32, 64, 100, 300, 512],
#         'min_sizes': [30, 60, 111, 162, 213, 264, 315],
#         'max_sizes': [60, 111, 162, 213, 264, 315, 366],
#         'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2], [2]],
#         'variance': [0.1, 0.2],
#         'clip': True,
#         'name': 'VOC',
#     }
#
# }
#
# coco = {
#     'num_classes': 201,
#     'lr_steps': (280000, 360000, 400000),
#     'max_iter': 400000,
#     'feature_maps': [38, 19, 10, 5, 3, 1],
#     'min_dim': 300,
#     'steps': [8, 16, 32, 64, 100, 300],
#     'min_sizes': [21, 45, 99, 153, 207, 261],
#     'max_sizes': [45, 99, 153, 207, 261, 315],
#     'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
#     'variance': [0.1, 0.2],
#     'clip': True,
#     'name': 'COCO',
# }
# models config

backbone = ['VGG16', 'ResNet50']

# bdd512 = {
#     'num_classes': 11,
#     'lr_steps': (15, 50, 75, 100, 200),
#     'max_iter': 120000,
#     'feature_maps': [64, 32, 16, 8, 4, 2, 1],
#     'min_dim': 512,
#     'steps': [8, 16, 32, 64, 100, 300, 512],
#     'min_sizes': [30, 60, 111, 162, 213, 264, 315],
#     'max_sizes': [60, 111, 162, 213, 264, 315, 366],
#     'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2], [2]],
#     'variance': [0.1, 0.2],
#     'clip': True,
#     'name': 'bdd',
# }

SSD300 = {
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'net_size': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'SSD300'
}
SSD512 = {
    'feature_maps': [64, 32, 16, 8, 4, 2, 1],
    'net_size': 512,
    'steps': [8, 16, 32, 64, 100, 300, 512],
    'min_sizes': [30, 60, 111, 162, 213, 264, 315],
    'max_sizes': [60, 111, 162, 213, 264, 315, 366],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'SSD512'
}

RFB300 = {}
RFB512 = {}
CFE300 = {}
CFE512 = {}
M2DET300 = {}
M2DET512 = {}
cfg = {'SSD300': SSD300, 'SSD512': SSD512}
# datasets config

COCO = {'num_classes': 81,
        'root': '',
        'names': [],
        'lr_steps': (280000, 360000, 400000),
        'max_iter': 400000}
VOC = {'num_classes': 21,
       'root': '',
       'names': [],
       'lr_steps': (100, 180, 250),
       'max_iter': 520000, }
BDD100K = {'num_classes': 11,
           'root': r'd:\datasets\BDD100K',
           'names': [],
           'lr_steps': (20, 50, 75, 100, 150, 300),
           'max_iter': 520000, }
