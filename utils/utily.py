import torch.nn as nn
import torch.nn.init as init


def adjust_learning_rate(lr, optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr_ = lr * (gamma ** (step - 1))
    print('Now we change lr ...')
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_


def warmup_learning_rate(lr, optimizer, epoch):
    lr_ini = 0.0001
    print('lr warmup...')
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_ini + (lr - lr_ini) * epoch / 5


def xavier(param):
    init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()
