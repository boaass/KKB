# -*- coding:utf-8 -*-


import cv2
import numpy as np
import matplotlib.pyplot as plt
import random


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
# iterations = 0
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

def RANSAC():
    iterations = 0
    bestfit_a = 0
    bestfit_b = 0
    besterr = 0.2

    n = 2       # the minimum number of data values required to fit the model
    k = 1000    # the maximum number of iterations allowed in the algorithm
    t = 0.25    # a threshold value for determining when a data point fits a model
    d = 0       # the number of close data values required to assert that a model fits well to data

    data_nums = 50
    data_X = np.linspace(0, 10, 50)
    data_Y = [3 * i + 10 for i in data_X[:-20]]
    data_Y += [random.randint(0, int(i)) for i in data_X[-20:]]

    # 画图展示
    fig = plt.figure()
    axe = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    axe.scatter(np.array(data_X), np.array(data_Y))
    plt.show()

    plt.ion()
    fig, ax = plt.subplots()
    while iterations < k:

        # 从样本数据中随机选取两个元素，构建模型
        random_indexs = random.sample(range(data_nums), n)
        maybeinliers_x1 = data_X[random_indexs[0]]
        maybeinliers_x2 = data_X[random_indexs[1]]
        maybeinliers_y1 = data_Y[random_indexs[0]]
        maybeinliers_y2 = data_Y[random_indexs[1]]

        maybemodel_a = (maybeinliers_y2 - maybeinliers_y1)/(maybeinliers_x2 - maybeinliers_x1)
        maybemodel_b = maybeinliers_y1 - maybemodel_a * maybeinliers_x1

        # 计算 inliers
        alsoinliers = 2
        for i in range(data_nums):
            if i == random_indexs[0] or i == random_indexs[1]:
                continue

            estimate_y = data_X[i] * maybemodel_a + maybemodel_b
            if abs(estimate_y - data_Y[i]) <= t:
                alsoinliers += 1

        # 判断模型好坏
        if alsoinliers > d:
            d = alsoinliers
            bestfit_a = maybemodel_a
            bestfit_b = maybemodel_b

        # 判断是否当前模型已经符合超过一定规模的点
        if alsoinliers >= (1 - besterr) * data_nums:
            break

        # 画图展示


if __name__ == '__main__':

    # 测试中值滤波
    img = imreadModes()
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # 测试灰度图
    noise_img = addNoise(img)   # 生成噪点图
    med_img = customMedianBlur(noise_img, 3)
    # cv2.imshow('noise_test', noise_img)
    # cv2.imshow('m_test', cv2.medianBlur(noise_img, 3))
    # imgShow(med_img)

    # RANSAC 伪代码
    RANSAC()

