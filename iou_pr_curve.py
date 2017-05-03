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
from sklearn.metrics import precision_recall_curve

import keras
from keras.models import Model
from keras.models import load_model
from multibox_loss import custom_loss
import keras.losses
keras.losses.custom_loss = custom_loss
import pickle

def get_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = np.amax([boxA[0], boxB[0]])
    yA = np.amax([boxA[1], boxB[1]])
    xB = np.amin([boxA[2], boxB[2]])
    yB = np.amin([boxA[3], boxB[3]])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


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

#get info on dataset
x_train_names, x_test_names, y_train, y_test, classes, bbox = get_data_info(num_ims)

#load the model
model = load_model(filepath)
iou_thresh = [0.0,0.25,0.5]
threshes = [.1*i for i in range(10)]
pr_curves = {}
new_pr = {}
for iou in iou_thresh:
    pr_curves[iou] = {}
    new_pr[iou] = []
    for thresh in threshes:
        pr_curves[iou][thresh] = []


for im_name in x_train_names[:10]:
    print(im_name)
    pic = imio.imread(image_folder + '/' + im_name)
    if pic.ndim == 3:
        (h, w, d) = pic.shape
        resized_pic = tform.resize(pic, size)
        resized_pic = np.reshape(resized_pic, (1, img_rows, img_cols, 3))
        old_bbox = bbox[im_name]
        new_bbox = [old_bbox[0] * img_rows / w, old_bbox[1] * img_cols / h, (old_bbox[0] + old_bbox[2]) * img_rows / w,
                (old_bbox[1] + old_bbox[3]) * img_cols / h]
        y_pred = model.predict([resized_pic], batch_size=1, verbose=0)
        conf = y_pred[:, :, -1]
        conf = np.reshape(conf, (1419, 1))
        # print(np.amax(conf))
        # if(np.amax(conf) is not 0):
        #     conf = conf / np.amax(conf)
        #conf = conf / np.amax(conf)

        y_pred = np.reshape(y_pred[:, :, :-1]+priors,(num_boxes,4))
        new_bbox = np.asarray(new_bbox)
    # print(new_bbox.shape)
    # print(y_pred.shape)
    # print(conf.shape)
        iou = []
        for box in range(num_boxes):
        # print(y_pred[box,:].shape)
        # print(y_pred[box,:])
        # print(new_bbox)
            iou.append(get_iou(list(y_pred[box,:]),list(new_bbox)))

        for iou_th in iou_thresh:
            for thresh in threshes:
                TP = 0
                TN = 0
                FP = 0
                FN = 0
                for box in range(num_boxes):
                    if conf[box,:] >= thresh:
                        # print('high confidence')
                        if iou[box] >= iou_th:
                            print('true positive')
                            TP +=1
                        else:
                            FP +=1
                    else:
                        if iou[box] >= iou_th:
                            FN += 1
                        else:
                            TN += 1
                pr_curves[iou_th][thresh].append([TP, TN, FP, FN])
print("Done predicting images")
pickle.dump(pr_curves,open('pr_curves2.p','wb'))
prec = 0
rec = 0
for i in iou_thresh:
    for j in threshes:
        pr_curves[i][j] = np.sum(np.asarray(pr_curves[i][j]),axis=1)
        print(i)
        print(j)
        print(pr_curves[i][j])
        prec = pr_curves[i][j][0]/(pr_curves[i][j][0]+pr_curves[i][j][3])
        rec = pr_curves[i][j][0]/(pr_curves[i][j][0]+pr_curves[i][j][2])
        # print(prec)
        # print(rec)
        new_pr[i].append([prec,rec])
    new_pr[i] = np.asarray(new_pr[i])

pickle.dump(new_pr,open('new_pr2.p','wb'))

plt.figure()
for i in iou_thresh:
    plt.plot(new_pr[i][:,0],new_pr[i][:,1])
plt.legend(['0.1','0.25', '0.5'], loc = 'upper left')
plt.title('PR curves for different IOUs')
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.savefig('PR_curves.png')
