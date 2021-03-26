from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd, build_ssd_efficientnet
import os
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
import math
import random
from utils.utily import *


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def seed_torch(seed=1024):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# seed_torch()

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='BDD100K', choices=['VOC', 'COCO', 'BDD100K'],
                    type=str)
parser.add_argument('--modelname', default='SSD512', choices=['SSD300', 'SSD512', 'RBF300'],
                    type=str)
parser.add_argument('--backbone', default='VGG16', type=str,
                    choices=['VGG16', 'ResNet50', 'EfficientNet'])
parser.add_argument('--losscname', default='ce', type=str,
                    choices=['ce', 'wloss', 'bce'],
                    help='class loss type')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_epoch', default=0, type=int,
                    help='Resume training at this epoch')
parser.add_argument('--total_epoch', default=300, type=int,
                    help='number of epochs to train')
parser.add_argument('--num_workers', default=6, type=int,
                    help='Number of workers used in loading data')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=1e-8, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--visdom', default=False, type=str2bool,
                    help='Use visdom for loss visualization')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def train():
    model_config = None
    datasets_config = None
    backbone = None
    cur_lr = args.lr

    if args.dataset == 'COCO':
        if args.dataset_root == VOC_ROOT:
            if not os.path.exists(COCO_ROOT):
                parser.error('Must specify dataset_root if specifying dataset')
            print("WARNING: Using default COCO dataset_root because " +
                  "--dataset_root was not specified.")
            args.dataset_root = COCO_ROOT
        cfg = coco
        dataset = COCODetection(root=args.dataset_root,
                                transform=SSDAugmentation(cfg['min_dim'],
                                                          MEANS))
    elif args.dataset == 'VOC':
        if args.dataset_root == VOC_ROOT:
            parser.error('Must specify dataset if specifying dataset_root')
        # cfg = voc
        dataset = VOCDetection(root=args.dataset_root,
                               transform=SSDAugmentation(args.input,
                                                         MEANS))
    elif args.dataset == 'BDD100K':
        from data.config import BDD100K as datasets_config
        from data.config import SSD512 as model_config
        from data.bdd100k import BDD100KDetection
        root = datasets_config['root']
        backbone = args.backbone
        dataset = BDD100KDetection(root, image_sets='train',
                                   transform=SSDAugmentation(model_config['net_size'],
                                                             MEANS))
    else:
        print(args.dataset, 'Not define,Return')
        return
    ssd_net = build_ssd('train', 'SSD512', 'VGG16', model_config['net_size'], datasets_config['num_classes'])
    net = ssd_net
    # for block in net.base.parameters():
    #     block.requires_grad = False
    if args.cuda:
        net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True

    if args.cuda:
        net = net.cuda()

    if args.start_epoch == 0:
        print('Initializing weights with {}'.format('vgg16_reducedfc.pth'))
        # initialize newly added layers' weights with xavier method
        vgg_weights = torch.load(r'weights/vgg16_reducedfc.pth')
        ssd_net.base.load_state_dict(vgg_weights)
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)
    else:
        path = args.save_folder + '/{}_{}_epoch_{}.pth'.format(args.modelname, args.dataset,
                                                               str(args.start_epoch).zfill(10))
        if os.path.exists(path):
            print('Initializing weights with {}'.format(str(path)))
            ssd_net.load_weights(path)
        else:
            vgg_weights = torch.load(r'weights/vgg16_reducedfc.pth')
            ssd_net.base.load_state_dict(vgg_weights)
            ssd_net.extras.apply(weights_init)
            ssd_net.loc.apply(weights_init)
            ssd_net.conf.apply(weights_init)

    optimizer = optim.AdamW(net.parameters(), lr=args.lr)
    criterion = MultiBoxLoss(datasets_config['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, args.cuda,
                             args.modelname)

    net.train()
    iteration = 1

    epoch_size = math.ceil(len(dataset) / args.batch_size)
    print(args)
    step_index = 0

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)

    for epoch in range(args.start_epoch, args.total_epoch):
        print('\n' + '-' * 70 + 'Epoch: {}'.format(epoch) + '-' * 70 + '\n')
        if epoch <= 5:
            warmup_learning_rate(args.lr, optimizer, epoch)
        else:
            if epoch in datasets_config['lr_steps']:
                step_index += 1
                adjust_learning_rate(args.lr, optimizer, args.gamma, step_index)
        for param in optimizer.param_groups:
            if 'lr' in param.keys():
                cur_lr = param['lr']

        for images, targets in data_loader:  # load train data

            if args.cuda:
                images = Variable(images.cuda())
                targets = [Variable(ann.cuda()) for ann in targets]
            else:
                images = Variable(images)
                targets = [Variable(ann) for ann in targets]
            t0 = time.time()
            out = net(images)
            optimizer.zero_grad()
            loss_l, loss_c = criterion(out, targets)
            loss = loss_l + loss_c
            loss.backward()
            optimizer.step()
            t1 = time.time()

            if iteration % 10 == 0:
                print('Epoch ' + repr(epoch) + '|| iter ' + repr(iteration % epoch_size) + '/' + repr(
                    epoch_size) + '|| Total iter ' + repr(
                    iteration) + ' || Total Loss: %.4f || Loc Loss: %.4f || Cls Loss: %.4f || LR: %f || timer: %.4f sec.\n' % (
                          loss.item(), loss_l.item(), loss_c.item(), cur_lr, (t1 - t0)), end=' ')

            if iteration != 0 and iteration % 1000 == 0:
                print('Saving state, iter:', iteration)
                torch.save(ssd_net.state_dict(),
                           args.save_folder + '{}_{}_{}_iter_{}.pth'.format(args.modelname, args.dataset,
                                                                            args.losscname,
                                                                            str(iteration).zfill(10)))
            iteration += 1
        torch.save(ssd_net.state_dict(),
                   args.save_folder + '/{}_{}_{}_epoch_{}.pth'.format(args.modelname, args.dataset, args.losscname,
                                                                      str(epoch).zfill(5)))


if __name__ == '__main__':
    train()
