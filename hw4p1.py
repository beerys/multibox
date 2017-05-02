# Sara Beery
# EE 148
# HW3
# 4/21/17
# Python 2.7.13

import numpy as np
#import confusion_mat as cm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from get_data_info import get_data_info
from multibox_loss import custom_loss
import skimage.io as imio
import skimage.transform as tform

import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Input, GlobalAveragePooling2D, Conv2D, Flatten, Reshape
from keras.layers.merge import concatenate
from keras import backend as K
#from keras.models import load_model
from keras.applications.inception_v3 import InceptionV3



batch_size = 32
train_all_classes = True
create_train_test_dirs = True
layers_to_train = 1
img_rows = 299
img_cols = 299
size = (img_rows,img_cols)
epochs = 10
batches = 300

if train_all_classes:
    num_ims = 11788
    num_classes = 200
else:
    num_ims = 1115
    num_classes = 20

source_dir = 'CUB_200_2011/CUB_200_2011/'
image_folder = source_dir + 'images'
test_folder = 'Test'
train_folder = 'Train'
model_name = 'Multibox_Bird_Model_1'
filepath = model_name + '.h5'

x_train_names, x_test_names, y_train, y_test, classes, bbox = get_data_info(num_ims)


# convert class vectors to binary class matrices, subtract 1 to get correct classes
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299,299,3))

# create model
x = base_model.output

#left path
a = Conv2D(96, (1,1), activation='relu')(x)
a = Conv2D(96, (3,3), padding='SAME', activation='relu')(a)
loc_a = Conv2D(44, (1,1))(a)
loc_a = Reshape(target_shape=(704,4))(loc_a)
conf_a = Conv2D(11, (1,1), activation='sigmoid')(a)
conf_a = Reshape(target_shape=(704,1))(conf_a)
out_a = concatenate([loc_a, conf_a])

# #center path
b = Conv2D(96, (3,3), padding='SAME', activation='relu')(x)
b = Conv2D(96, (3,3), activation='relu')(b)
loc_b = Conv2D(44, (1,1))(b)
loc_b = Reshape(target_shape=(396,4))(loc_b)
conf_b = Conv2D(11, (1,1), activation='sigmoid')(b)
conf_b = Reshape(target_shape=(396,1))(conf_b)
out_b = concatenate([loc_b, conf_b])
#
# #right path
c = Conv2D(256,(3,3), padding='SAME', strides=2, activation='relu')(x)

c1 = Conv2D(128,(3,3), padding = 'SAME',activation='relu')(c)
loc_c1 = Conv2D(44, (1,1))(c1)
loc_c1 = Reshape(target_shape=(176,4))(loc_c1)
conf_c1 = Conv2D(11, (1,1),activation='sigmoid')(c1)
conf_c1 = Reshape(target_shape=(176,1))(conf_c1)
out_c1 = concatenate([loc_c1, conf_c1])
#
c2 = Conv2D(128, (1,1), activation='relu')(c)
c2 = Conv2D(96, (2,2), activation='relu')(c2)
loc_c2 = Conv2D(44, (1,1))(c2)
loc_c2 = Reshape(target_shape=(99,4))(loc_c2)
conf_c2 = Conv2D(11, (1,1), activation='sigmoid')(c2)
conf_c2 = Reshape(target_shape=(99,1))(conf_c2)
out_c2 = concatenate([loc_c2, conf_c2])
#
c3 = Conv2D(128, (1,1), activation='relu')(c)
c3 = Conv2D(96, (3,3), activation='relu')(c3)
loc_c3 = Conv2D(44, (1,1))(c3)
loc_c3 = Reshape(target_shape=(44,4))(loc_c3)
conf_c3 = Conv2D(11, (1,1), activation='sigmoid')(c3)
conf_c3 = Reshape(target_shape=(44,1))(conf_c3)
out_c3 = concatenate([loc_c3, conf_c3])
#
out = concatenate([out_a, out_b, out_c1, out_c2, out_c3], axis=1)

# predictions = (loc_a, conf_a, loc_b, conf_b, loc_c1, conf_c1, loc_c2, conf_c2, loc_c3, conf_c3)
predictions = out
# this is the model we will train
model = Model(input=base_model.input, output=predictions)
# model.summary()
# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='RMSprop', loss=custom_loss, metrics=['accuracy'])
#

for i in range(epochs):
    for j in range(batches):
        choices = np.random.choice(x_train_names,batch_size)
        x_train = np.zeros((0,img_rows,img_cols,3))
        y_true = np.zeros((0,4))
        for choice in choices:
            pic = imio.imread(image_folder + '/' + choice)
            v_ratio = pic.shape[1]/img_cols
            h_ratio = pic.shape[0]/img_rows
            pic = tform.resize(pic, size)
            # print("pic shape:")
            # print(pic.shape)
            # print("x_shape")
            # print(x_train.shape)
            while(pic.ndim != 3):
                print(choice)
                replace = np.random.choice(x_train_names, 1)
                pic = imio.imread(image_folder + '/' + replace[0])
                v_ratio = pic.shape[1] / img_cols
                h_ratio = pic.shape[0] / img_rows
                pic = tform.resize(pic, size)

            x_train = np.append(x_train, [pic], axis = 0)
            old_bbox = bbox[choice]
            new_bbox = [old_bbox[0]*h_ratio, old_bbox[1]*v_ratio, (old_bbox[0]+ old_bbox[2])*h_ratio, (old_bbox[1]+old_bbox[3])*v_ratio]
            y_true = np.append(y_true, [new_bbox], axis=0)

        y_true = np.reshape(y_true,(batch_size,1,4))
        y_true = np.tile(y_true,(1,1419,1))
        # print(y_true.shape)
        model.fit(x_train,y_true,shuffle = False)

    model.save('models/' + model_name + str(i) + '.h5')

model.save(filepath)




