#Sara Beery
# EE 148
# HW3
# 4/21/17
# Python 2.7.13

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as imio
import scipy.io as sio
import skimage.io as imio
import skimage.transform as tform
import numpy as np
from get_data_info import get_data_info
from sklearn.preprocessing import normalize

import keras
from keras.models import Model
from keras.models import load_model
from multibox_loss import custom_loss
import keras.losses
keras.losses.custom_loss = custom_loss

source_dir = 'CUB_200_2011/CUB_200_2011/'
image_folder = source_dir + 'images'
test_folder = 'Test'
train_folder = 'Train'
model_name = 'Multibox_Bird_Model_2'
filepath = model_name + '.h5'

num_ims = 11788
num_classes = 200
img_rows = 299
img_cols = 299
num_boxes = 1419

size = (img_rows,img_cols)
priors = np.zeros((0, 4))
for i in range(num_boxes):
    col = i % 40
    row = (i - col) / 40
    bbox = [6 * row, 6 * col, 50, 50]
    priors = np.append(priors, [bbox], axis=0)
priors = np.reshape(priors, (1, num_boxes, 4))
# priors = np.tile(priors, (batch_size, 1, 1))

#get info on dataset
x_train_names, x_test_names, y_train, y_test, classes, bbox = get_data_info(num_ims)

#load the model
model = load_model(filepath)


choice = np.random.choice(x_train_names, 1)
pic = imio.imread(image_folder + '/' + choice[0])
(h, w, d) = pic.shape
resized_pic = tform.resize(pic, size)
resized_pic = np.reshape(resized_pic, (1,img_rows,img_cols,3))

y_pred = model.predict([resized_pic], batch_size=1, verbose=1)
conf = y_pred[:,:,-1]
conf = np.reshape(conf, (1419, 1))
print(np.amax(conf))
# if(np.amax(conf) is not 0):
#     conf = conf / np.amax(conf)
# conf = conf / np.amax(conf)

y_pred = y_pred[:,:,:-1] + priors
print(np.amax(conf))
print(y_pred.shape)
print(conf.shape)

#plot predicted boxes
boxes = []
fig,ax = plt.subplots(1)
ax.imshow(pic)
for i in range(num_boxes):
    boxes = y_pred[:,i,:]
    boxes = np.reshape(boxes, (4,1))
    boxes = [boxes[0]*w/img_rows, boxes[1]*h/img_cols, boxes[2]*w/img_rows, boxes[3]*h/img_cols]
    a = conf[i,:]
    # print(a)
    if( a > 0.2):
        ax.add_patch(patches.Rectangle((boxes[1], boxes[0]), boxes[3] - boxes[1], boxes[2] - boxes[0], alpha=float(a), facecolor='green'))
#plt.title('Bounding boxes in image ' + str(imNum))
plt.axis('off')
img_name = choice[0].split('/')[1]
print(img_name)
plt.savefig(model_name + '_predictions_' + img_name)