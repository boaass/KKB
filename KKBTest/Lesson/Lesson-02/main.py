# -*- coding:utf-8 -*-


import cv2
import numpy as np
import matplotlib.pyplot as plt


def imreadModes(flag=1):
    img = cv2.imread('test.jpg', flag)
    # print(img)
    return img


def imgShow(img, type=0):
    if type == 0:
        cv2.imshow('test', img)
        key = cv2.waitKey()
        if key == 0:
            cv2.destroyAllWindows()
    else:
        img_new = bgr2RGB(img)
        plt.imshow(img_new)
        plt.show()


def customMedianBlur(scr, ksize=3):
    if ksize % 2 == 0:
        raise AssertionError("ksize % 2 == 1")
    h, w = scr.shape[:2]
    space = int((ksize-1)/2)
    new_img = scr.copy()
    for i in range(space, h - space):
        for j in range(space, w - space):
            new_img[i, j, 0] = np.median(scr[i - space:i + space + 1, j - space:j + space + 1, 0])  # B
            new_img[i, j, 1] = np.median(scr[i - space:i + space + 1, j - space:j + space + 1, 1])  # G
            new_img[i, j, 2] = np.median(scr[i - space:i + space + 1, j - space:j + space + 1, 2])  # R

    return new_img


def bgr2RGB(img):
    # BGR -> RGB
    # B, G, R = cv2.split(img)
    # img_new = cv2.merge((R, G, B))

    img_new = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_new


def addNoise(img):
    h, w = img.shape[:2]
    # new_img = img.copy()
    new_img = np.array(img)
    noisecount = 4000
    for i in range(0, noisecount):
        x = int(np.random.random() * (h-1))
        y = int(np.random.random() * (w-1))
        new_img[x, y] = (255, 255, 255)

    return new_img


if __name__ == '__main__':
    img = imreadModes()
    noise_img = addNoise(img)
    med_img = customMedianBlur(noise_img, 3)
    cv2.imshow('test1', noise_img)
    imgShow(med_img)
