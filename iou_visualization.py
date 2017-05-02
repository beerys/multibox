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

import keras
from keras.models import Model
from keras.models import load_model
from multibox_loss import custom_loss

source_dir = 'CUB_200_2011/CUB_200_2011/'
image_folder = source_dir + 'images'
test_folder = 'Test'
train_folder = 'Train'
model_name = 'Multibox_Bird_Model_1'
filepath = model_name + '.h5'

num_ims = 11788
num_classes = 200
img_rows = 299
img_cols = 299
size = (img_rows,img_cols)

#get info on dataset
x_train_names, x_test_names, y_train, y_test, classes, bbox = get_data_info(num_ims)

#load the model
model = load_model(filepath)


choice = np.random.choice(x_train_names, 1)
pic = imio.imread(image_folder + '/' + choice[0])
(h, w, d) = pic.shape
resized_pic = tform.resize(pic, size)

y_pred = model.predict([resized_pic], batch_size=1, verbose=1)
conf = y_pred[:,:,-1]
y_pred = y_pred[:,:,:-1]
print(y_pred.shape)
print(conf.shape)

#plot predicted boxes
boxes = []
fig,ax = plt.subplots(1)
ax.imshow(pic)
# for line in green_lights:
#     if line[0] == str(imNum).zfill(3):
#         print(line)
#         boxes = line[1:5]
#         a = line[5]
#         ax.add_patch(patches.Rectangle((boxes[3], boxes[0]), boxes[3] - boxes[1], boxes[2] - boxes[0], alpha=float(a), facecolor='green'))
# for line in red_lights:
#     if line[0] == str(imNum).zfill(3):
#         print(line)
#         boxes = line[1:5]
#         a = line[5]
#         ax.add_patch(patches.Rectangle((boxes[3], boxes[0]), boxes[3] - boxes[1], boxes[2] - boxes[0], alpha=float(a), facecolor='red'))
#
# #plt.title('Bounding boxes in image ' + str(imNum))
# plt.axis('off')
# plt.savefig(model_name + '_img_' + str(imNum))