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


def bgr2RGB(img):
    # BGR -> RGB
    # B, G, R = cv2.split(img)
    # img_new = cv2.merge((R, G, B))

    img_new = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_new


def customMedianBlur(scr, ksize=3):
    if ksize % 2 <= 0:
        raise AssertionError("ksize % 2 == 1")

    h, w = scr.shape[:2]
    space = int((ksize - 1) / 2)
    new_img = scr.copy()

    for i in range(0, h):
        for j in range(0, w):
            if len(scr.shape) == 3:
                new_img[i, j, 0] = np.median(arrForMedian(scr, i, j, space, 0))  # B
                new_img[i, j, 1] = np.median(arrForMedian(scr, i, j, space, 1))  # G
                new_img[i, j, 2] = np.median(arrForMedian(scr, i, j, space, 2))  # R
            elif len(scr.shape) == 2:
                new_img[i, j] = np.median(arrForMedian(scr, i, j, space))  # gray

    return new_img


def arrForMedian(scr, i, j, space, channel_id=-1):

    h, w = scr.shape[:2]
    if channel_id == -1:
        return scr[i - space if (i - space) >= 0 else 0:
               i + space + 1 if (i + space + 1) <= h else h,
           j - space if (j - space) >= 0 else 0:
           j + space + 1 if (j + space + 1) <= w else w]
    else:
        return scr[i - space if (i - space) >= 0 else 0:
               i + space + 1 if (i + space + 1) <= h else h,
           j - space if (j - space) >= 0 else 0:
           j + space + 1 if (j + space + 1) <= w else w, channel_id]


def addNoise(img):
    h, w = img.shape[:2]
    # new_img = img.copy()
    new_img = np.array(img)
    noisecount = 4000
    for i in range(0, noisecount):
        x = int(np.random.random() * (h-1))
        y = int(np.random.random() * (w-1))
        if len(img.shape) == 3:
            new_img[x, y] = (255, 255, 255)
        elif len(img.shape) == 2:
            new_img[x, y] = 255

    return new_img


def RANSAC():
    # Given:
    #     data – a set of observed data points
    #     model – a model that can be fitted to data points
    #     n – the minimum number of data values required to fit the model
    #     k – the maximum number of iterations allowed in the algorithm
    #     t – a threshold value for determining when a data point fits a model
    #     d – the number of close data values required to assert that a model fits well to data
    #
    # Return:
    #     bestfit – model parameters which best fit the data (or nul if no good model is found)
    #
    iterations = 0
    # bestfit = nul
    # besterr = something really large
    # while iterations < k {
    #     maybeinliers = n randomly selected values from data
    #     maybemodel = model parameters fitted to maybeinliers
    #     alsoinliers = empty set
    #     for every point in data not in maybeinliers {
    #         if point fits maybemodel with an error smaller than t
    #              add point to alsoinliers
    #     }
    #     if the number of elements in alsoinliers is > d {
    #         % this implies that we may have found a good model
    #         % now test how good it is
    #         bettermodel = model parameters fitted to all points in maybeinliers and alsoinliers
    #         thiserr = a measure of how well model fits these points
    #         if thiserr < besterr {
    #             bestfit = bettermodel
    #             besterr = thiserr
    #         }
    #     }
    #     increment iterations
    # }
    # return bestfit


if __name__ == '__main__':

    # 测试中值滤波
    img = imreadModes()
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # 测试灰度图
    noise_img = addNoise(img)   # 生成噪点图
    med_img = customMedianBlur(noise_img, 3)
    cv2.imshow('noise_test', noise_img)
    cv2.imshow('m_test', cv2.medianBlur(noise_img, 3))
    imgShow(med_img)

    # RANSAC 伪代码
    RANSAC()

