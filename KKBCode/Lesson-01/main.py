# -*- coding:utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt
import random


def imreadModes(flag=1):
    img = cv2.imread('test.jpg', flag)
    # print(img)
    return img


def imgShow(name='test', img=None, type=0):
    if type == 0:
        cv2.imshow(name, img)
        key = cv2.waitKey()
        if key == 0:
            cv2.destroyAllWindows()
    else:
        img_new = bgr2RGB(img)
        plt.imshow(img_new)
        plt.show()

def bgr2RGB(img):
    # BGR -> RGB
    # B, G, R = cv2.split(img)
    # img_new = cv2.merge((R, G, B))

    img_new = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_new


## 一阶导
# sobel
def sobelX(img):
    kernel = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])
    sx_img = cv2.filter2D(img, -1, kernel)
    return sx_img


def sobelY(img):
    kernel = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]])
    sy_img = cv2.filter2D(img, -1, kernel)
    return sy_img


# medianblur
# 中值滤波
def medianblur(img):
    return cv2.medianBlur(img, 7)

## 二阶导
# Gaussian kernel
def gaussianKernel(img):
    g_img = cv2.blur(img, (7, 7), 4)
    kernel = cv2.getGaussianKernel(7, 4)
    return g_img, kernel


def filter2D(img, kernelx, kernely):
    f_img = cv2.filter2D(img, -1, kernelx, kernely)
    return f_img


# Laplacian
def laplacian(img):
    # 突出边缘
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]])
    lap_img = cv2.filter2D(img, -1, kernel)
    return lap_img


if __name__ == '__main__':
    img = imreadModes()

    # Gaussian
    g_img, kernel_1d = gaussianKernel(img)
    # cv2.imshow('test', img)
    # imgShow(name='g_test', img=g_img)

    f2d_img = filter2D(img, kernel_1d, kernel_1d)
    # cv2.imshow('test', img)
    # imgShow(name='g1_test', img=f2d_img)

    # Laplacian
    lap_img = laplacian(img)
    # cv2.imshow('test', img)
    # imgShow(name='l_test', img=lap_img)

    # sobel
    sx_img = sobelX(img)
    sy_img = sobelY(img)
    # cv2.imshow('sobelX', sx_img)
    # imgShow('sobelY', sy_img)

    # medianblure
    med_img = medianblur(img)
    # cv2.imshow('test', img)
    # imgShow('medianblure', med_img)








