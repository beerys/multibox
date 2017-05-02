#Sara Beery
# EE 148
# HW3
# 4/21/17
# Python 2.7.13

import numpy as np
from keras import backend as K




def custom_loss(y_true, y_pred):
    epsilon = 1.0e-7
    alpha = 0.25
    num_boxes = 1419
    batch_size = 32
    priors = np.zeros((0, 4))
    for i in range(num_boxes):
        col = i % 40
        row = (i - col) / 40
        bbox = [6 * row, 6 * col, 50, 50]
        priors = np.append(priors, [bbox], axis=0)
    priors = np.reshape(priors, (1, num_boxes, 4))
    priors = np.tile(priors, (batch_size, 1, 1))

    conf = y_pred[:,:,-1]
    y_pred = y_pred[:,:,:-1]
    y_pred = y_pred + priors
    loc_loss = alpha*K.sum(K.square(y_true-y_pred),axis=2)
    conf_loss = -K.log(conf+epsilon)+K.log(1-conf+epsilon)
    univ_conf_loss = K.sum(K.log(1-conf+epsilon))
    return K.min(loc_loss+conf_loss, axis=1)-univ_conf_loss
