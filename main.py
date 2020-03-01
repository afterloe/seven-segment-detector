#!/usr/bin/env python3
# -*- coding=utf-8 -*-

import cv2 as cv
import imutils
from imutils import contours
from imutils.perspective import four_point_transform
import numpy as np

"""
识别步骤

Step #1: 使用边缘检测实现获取LED液晶显示器
Step #2: 取出LED，并更具映射模型寻找一个矩形轮廓
Step #3: 提取数字区域
Step #4: 数字识别
"""

# 表的关键是七段数组。数组中的1表示给定的段处于打开状态，0表示该段处于关闭状态。
DIGITS_LOOKUP = {
    (1, 1, 1, 0, 1, 1, 1): 0,
    (0, 0, 1, 0, 0, 1, 0): 1,
    (1, 0, 1, 1, 1, 1, 0): 2,
    (1, 0, 1, 1, 0, 1, 1): 3,
    (0, 1, 1, 1, 0, 1, 0): 4,
    (1, 1, 0, 1, 0, 1, 1): 5,
    (1, 1, 0, 1, 1, 1, 1): 6,
    (1, 0, 1, 0, 0, 1, 0): 7,
    (1, 1, 1, 1, 1, 1, 1): 8,
    (1, 1, 1, 1, 0, 1, 1): 9,
}
classifier = "resources/svm_led.data"


def generator_data(image):
    cnts, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    rois = []
    for c in range(len(cnts)):
        box = cv.boundingRect(cnts[c])
        if box[3] < 10:
            continue
        rois.append(box)
    num = len(rois)
    for i in range(num):
        for j in range(i + 1, num, 1):
            x1, y1, w1, h1 = rois[i]
            x2, y2, w2, h2 = rois[j]
            if x2 < x1:
                temp = rois[j]
                rois[j] = rois[i]
                rois[i] = temp
    bgr = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    index = 0
    digit_data = np.zeros((num, 30 * 50), dtype=np.float32)
    for x, y, w, h in rois:
        # cv.putText(bgr, str(index), (x, y+10), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), 1)
        digit = image[y: y + h, x: x + w]
        img = cv.resize(digit, (30, 50))
        row = np.reshape(img, (-1, 30 * 50))
        digit_data[index] = row
        index += 1
    return digit_data, rois


def main():
    image = cv.imread("resources/example.jpg")
    image = imutils.resize(image, width=500)
    # 通过调整图像大小，将其转换为灰度，模糊并计算边缘
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (9, 9), 0)
    edged = cv.Canny(blurred, 50, 200, 255)
    # cv.imshow("edged", edged)
    # cv.waitKey(0)
    # 寻找轮廓
    cnts = cv.findContours(edged.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # 轮廓排序
    cnts = sorted(cnts, key=cv.contourArea, reverse=True)
    display_cnt = None
    # 如果近似轮廓有四个顶点，假设找到LED区域。
    for c in cnts:
        # approximate the contour
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.02 * peri, True)
        if 4 == len(approx):
            display_cnt = approx
            break
    # 获得四个顶点后，我们可以通过四点透视变换提取LCD：
    warped = four_point_transform(gray, display_cnt.reshape(4, 2))
    output = four_point_transform(image, display_cnt.reshape(4, 2))
    # _, thresh = cv.threshold(warped, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    thresh = cv.adaptiveThreshold(warped, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 45, 3)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (1, 3))
    thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)

    cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    digit_cnts = []
    for cnt in cnts:
        x, y, w, h = cv.boundingRect(cnt)
        if w >= 15 and (30 <= h <= 50):
            digit_cnts.append(cnt)
            cv.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)

    digit_cnts = contours.sort_contours(digit_cnts, method="left-to-right")[0]
    digits = []

    svm = cv.ml.SVM_load(classifier)

    # i = 0
    for cnt in digit_cnts:
        # extract the digit ROI
        (x, y, w, h) = cv.boundingRect(cnt)
        roi = thresh[y:y + h, x:x + w]
        # cv.imwrite("./resources/train_data/{}.jpeg".format(str(i)), roi, [cv.IMWRITE_JPEG_QUALITY, 100])
        # i = i + 1
        # cv.imshow("roi", roi)
        data, boxes = generator_data(roi)
        result = svm.predict(data)[1]
        digits.append(str(np.int32(result[0][0])))
        cv.waitKey(0)
        # compute the width and height of each of the 7 segments
        # we are going to examine
        # roi_h, roi_w = roi.shape
        # d_w, d_h = int(roi_w * 0.35), int(roi_h * 0.25)
        # dHC = int(roi_h * 0.15)
        # print(roi_h, roi_w)
        # print(d_w, d_h)
        # # define the set of 7 segments
        # segments = [
        #     ((0, 0), (w, d_h)),  # top
        #     ((0, 0), (d_w, h // 2)),  # top-left
        #     ((w - d_w, 0), (w, h // 2)),  # top-right
        #     ((0, (h // 2) - dHC), (w, (h // 2) + dHC)),  # center
        #     ((0, h // 2), (d_w, h)),  # bottom-left
        #     ((w - d_w, h // 2), (w, h)),  # bottom-right
        #     ((0, h - d_h), (w, h))  # bottom
        # ]
        # print(segments)
        # on = [0] * len(segments)
        # print(on)
        # for i, ((x_a, y_a), (x_b, y_b)) in enumerate(segments):
        #     seg_roi = roi[y_a: y_b, x_a: y_b]
        #     total = cv.countNonZero(seg_roi)
        #     area = (x_b - x_a) * (y_b - y_a)
        #     if 0.5 < total / float(area):
        #         on[i] = 1
        # print(tuple(on))
        # print("--------------------------------------------->>>>>>>>")
        # digit = DIGITS_LOOKUP[tuple(on)]
        # digits.append(digit)
        # cv.putText(output, str(digit), (x - 10, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    print(u"{}{}.{} \u00b0C".format(*digits))

    # cv.imshow("thresh", thresh)
    cv.imshow("output", output)
    cv.waitKey(0)


if "__main__" == __name__:
    main()
