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

    conf = y_pred[:,:,-1]
    y_pred = y_pred[:,:,:-1]
    loc_loss = alpha*K.sum(K.square(y_true-y_pred),axis=2)
    conf_loss = -K.log(conf+epsilon)+K.log(1-conf+epsilon)
    univ_conf_loss = K.sum(K.log(conf+epsilon))
    return K.min(loc_loss+conf_loss, axis=1)-univ_conf_loss
