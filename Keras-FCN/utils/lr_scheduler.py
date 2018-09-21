import numpy as np


# learning rate scheduler
def lr_scheduler(epoch, mode='power_decay'):

    # original lr scheduler
    if mode is 'power_decay':
        lr = lr_base * ((1 - float(epoch)/epochs) ** lr_power)

    # exponential decay
    if mode is 'exp_decay':
        lr = (float(lr_base) ** float(lr_power)) ** float(epoch+1)

    # adam default lr
    if mode is 'adam':
        lr = 0.001

    # drops as progression proceeds, good for sgd
    if mode is 'progressive_drops':
        if epoch > 0.9 * epochs:
            lr = 0.0001
        elif epoch > 0.75 * epochs:
            lr = 0.001
        elif epoch > 0.50 * epochs:
            lr = 0.01
        else:
            lr = 0.1

    print('Learning Rate : %f' % lr)
    return lr
    