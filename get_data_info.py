#Sara Beery
# EE 148
# HW3
# 4/21/17
# Python 2.7.13

import numpy as np


def get_data_info(num_ims):

    source_dir = 'CUB_200_2011/CUB_200_2011/'
    class_file = source_dir + 'image_class_labels.txt'
    image_file = source_dir + 'images.txt'
    split_file = source_dir + 'train_test_split.txt'
    bboxes = [line.rstrip().split() for line in open(source_dir + 'bounding_boxes.txt')]
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    classes = []
    bbox = {}
    f1 = open(class_file, 'r')
    f2 = open(image_file, 'r')
    f3 = open(split_file, 'r')
    count = 0
    while count < num_ims:
        line = f3.readline().split()
        file_name = f2.readline().split()[1]
        class_name = file_name.split('/')[0]
        class_num = float(f1.readline().split()[1])-1
        # print(bboxes[count])
        bbox[file_name] = [float(bboxes[count][1]), float(bboxes[count][2]), float(bboxes[count][3]), float(bboxes[count][4])]
        if class_name not in classes:
            classes.append(class_name)
        if float(line[1]):
            x_train.append(file_name)
            y_train.append(class_num)
        else:
            x_test.append(file_name)
            y_test.append(class_num)
        count += 1
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    return x_train, x_test, y_train, y_test, classes, bbox