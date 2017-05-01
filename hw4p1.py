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
import loss

import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Input, GlobalAveragePooling2D, Conv2D
from keras import backend as K
#from keras.models import load_model
#from sklearn.metrics import confusion_matrix
from keras.callbacks import ModelCheckpoint, CSVLogger
#from keras.callbacks import TensorBoard
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator
import pickle


batch_size = 32
train_all_classes = True
create_train_test_dirs = True
layers_to_train = 1
img_rows = 299
img_cols = 299
size = (img_rows,img_cols)
epochs = 10

if train_all_classes:
    num_ims = 11788
    num_classes = 200
else:
    num_ims = 1115
    num_classes = 20

image_folder = 'CUB_200_2011/CUB_200_2011/images'
test_folder = 'Test'
train_folder = 'Train'
filepath = "Bird_Model_1.h5"

x_train_names, x_test_names, y_train, y_test, classes = get_data_info(num_ims)

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
conf_a = Conv2D(11, (1,1))(a)

# #center path
b = Conv2D(96, (3,3), padding='SAME', activation='relu')(x)
b = Conv2D(96, (3,3), activation='relu')(b)
loc_b = Conv2D(44, (1,1))(b)
conf_b = Conv2D(11, (1,1))(b)
#
# #right path
c = Conv2D(256,(3,3), padding='SAME', strides=2, activation='relu')(x)

c1 = Conv2D(128,(3,3), padding = 'SAME',activation='relu')(c)
loc_c1 = Conv2D(44, (1,1))(c1)
conf_c1 = Conv2D(11, (1,1))(c1)
#
c2 = Conv2D(128, (1,1), activation='relu')(c)
c2 = Conv2D(96, (2,2), activation='relu')(c2)
loc_c2 = Conv2D(44, (1,1))(c2)
conf_c2 = Conv2D(11, (1,1))(c2)
#
c3 = Conv2D(128, (1,1), activation='relu')(c)
c3 = Conv2D(96, (3,3), activation='relu')(c3)
loc_c3 = Conv2D(44, (1,1))(c3)
conf_c3 = Conv2D(11, (1,1))(c3)
#
predictions = (loc_a, conf_a, loc_b, conf_b, loc_c1, conf_c1, loc_c2, conf_c2, loc_c3, conf_c3)

# this is the model we will train
model = Model(input=base_model.input, output=predictions)
model.summary()
# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

#create checkpoints (after model has been compiled)
# filepath = "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
# callbacks_list = [checkpoint]

#create Tensorboard Logs
#remember to pass this to your model while fitting!! model.fit(...inputs and parameters..., callbacks=[tbCallBack])
#tbCallBack = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)

#create callbacks
callbacks = [ModelCheckpoint('models/' + filepath + '-{epoch:02d}-{val_acc:.4f}.hdf5'),CSVLogger(filepath + '-history', separator=',', append=False)]

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
#
datagen = ImageDataGenerator()


#fit the model (should I specify classes?  How do I split the training and test data)
history = model.fit_generator(datagen.flow_from_directory(directory=train_folder, target_size=size,classes=classes),
					validation_data=datagen.flow_from_directory(directory=test_folder, target_size=size,classes=classes),
					validation_steps=len(x_test_names)/batch_size,
                    epochs=epochs,
                    steps_per_epoch=len(x_train_names)/batch_size,
                    callbacks=callbacks,
                    verbose=1)

model.save(filepath)
#
# # score = model.evaluate(x_test, y_test, verbose=0)
#
# # print('Test loss:', score[0])
# # print('Test accuracy:', score[1])
# # score = model.evaluate(x_train, y_train, verbose=0)
# # print('Training loss:', score[0])
# # print('Training accuracy:', score[1])
#
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['train','test'], loc = 'upper left')
plt.savefig(filepath + '_accuracy.png', bbox_inches='tight')
#
#
