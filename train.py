#!/usr/bin/env python3
# -*- coding=utf-8 -*-

import cv2 as cv
import os
import numpy as np

"""

"""

images = []
labels = []
path_of_images = "resources/train_data"
classifier = "resources/svm_led.data"


def load_data():
    files = os.listdir(path_of_images)
    count = len(files)
    sample_data = np.zeros((count, 30 * 50), dtype=np.float32)
    index = 0
    for name in files:
        file = os.path.join(path_of_images, name)
        if True is os.path.isfile(file):
            images.append(file)
            labels.append(name[:1])
            image = cv.imread(file, cv.IMREAD_GRAYSCALE)
            image = cv.resize(image, (30, 50))
            row = np.reshape(image, (-1, 30 * 50))
            sample_data[index] = row
            index += 1
    return sample_data, np.asarray(labels, np.int32)


def test_train(data):
    svm = cv.ml.SVM_load(classifier)
    result = svm.predict(data)[1]
    print("[info]: {}".format(result))


def main():
    train_data, train_labels = load_data()
    svm = cv.ml.SVM_create()
    svm.setKernel(cv.ml.SVM_LINEAR)
    svm.setType(cv.ml.SVM_C_SVC)
    svm.setC(2.67)
    svm.setGamma(5.383)
    svm.train(train_data, cv.ml.ROW_SAMPLE, train_labels)
    svm.save(classifier)
    print("[info]: 分类完成")
    test_train(train_data)


if "__main__" == __name__:
    main()
