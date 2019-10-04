# -*- coding:utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt
import random



def imreadModes(flag=None):
    img = cv2.imread('test.jpg', flag)
    # print(img)
    return img


def imgShow(img, type=0):
    if type == 0:
        cv2.imshow('test', img)
        key = cv2.waitKey(10 * 1000)
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


def colorShift(img):
    # 颜色偏移

    new_B = []
    new_G = []
    new_R = []

    for i in range(256):
        r_num = random.randint(1, 100)
        if i + r_num > 255:
            new_B.append(255)
            new_G.append(i)
            new_R.append(i * r_num / 255.0)
            continue
        if r_num - i < 0:
            new_G.append(0)
            new_B.append(i)
            new_R.append(i * r_num / 255.0)
            continue

        new_B.append(i)
        new_G.append(i)
        new_R.append(i*r_num/255.0)

    # lut = np.arange(255, -1, -1, dtype=np.uint8)
    new_B = np.array(new_B).astype('uint8')
    new_G = np.array(new_B).astype('uint8')
    new_R = np.array(new_B).astype('uint8')

    table = cv2.merge((new_B, new_G, new_R))
    # table = np.array(table).astype('uint8')
    img_cs = cv2.LUT(img, table)
    return img_cs


def gammaAdjust(img, gamma=0.6):
    # gamma correction
    table = []
    for i in range(256):
        # 归一化
        table.append(((i/255.0)**gamma) * 255)
    table = np.array(table).astype('uint8')
    img_gamma = cv2.LUT(img, table)
    return img_gamma


def imgCrop(img):
    # 图像截取
    # print(img.shape)
    img_crop = img[0:100, 0:100]
    return img_crop


def imgHist(img):
    hist = img.flatten()
    print(hist)
    plt.hist(hist, 25, [0, 256])
    plt.show()


def similarityTransform(img, angle, scale):
    # 图像旋转
    # 相似变换
    # 旋转、平移、缩放
    # 保角性

    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((h/2, w/2), angle, scale)
    img_rotation = cv2.warpAffine(img, M, (w, h))
    return img_rotation


def affineTransform(img):
    # 仿射变换
    # 旋转、平移
    # 不保角，平行线依然平行
    h, w = img.shape[:2]
    pts1 = np.float32([[0, 0], [w - 1, 0], [0, h - 1]])
    pts2 = np.float32([[w*0.3, h*0.3], [w*0.9, h*0.1], [w*0.1, h*0.9]])
    M = cv2.getAffineTransform(pts1, pts2)
    img_affine = cv2.warpAffine(img, M, (w, h))
    return img_affine


def imgPerspective(img):
    # 透视变换
    # 直线依然是直线
    # 单应性：单应性是一个从实射影平面到射影平面的可逆变换，直线在该变换下仍映射为直线。
    h, w = img.shape[:2]
    pts1 = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
    pts2 = np.float32([[w*0.3, h*0.3], [w*0.9, h*0.1], [w*0.1, h*0.9], [w*0.6, h*0.5]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    img_perspective = cv2.warpPerspective(img, M, img.shape[:2])
    return img_perspective


# def imgAugmentation():
#     # 图像各种变换、GAN


if __name__ == '__main__':
    # 0为灰度，1为彩色三通道，BGR
    # img = imreadModes(0)
    img = imreadModes(1)
    # imgShow(img)

    # gamma correction
    img_gamma = gammaAdjust(img, 0.4)
    # imgShow(img_gamma)

    # color shift
    img_cs = colorShift(img)
    # imgShow(img_cs)

    # 图像截取
    img_crop = imgCrop(img)
    # imgShow(img_crop)

    # hist
    img_gray = imreadModes(0)
    # imgHist(img_gray)

    # 图像校准算法：统计直方图平均化
    img_eq = cv2.equalizeHist(img_gray)
    # imgHist(img_eq)


    # 相似变换
    # 旋转、平移、缩放
    # 保角性
    img_rotation = similarityTransform(img, 180, 0.5)
    # imgShow(img_rotation)

    # 仿射变换
    # 旋转、平移
    # 不保角，平行线依然平行
    img_affine = affineTransform(img)
    # imgShow(img_affine)

    # 投影(透视)变换
    # 直线依然是直线
    # 单应性：单应性是一个从实射影平面到射影平面的可逆变换，直线在该变换下仍映射为直线。
    img_perspective = imgPerspective(img)
    # imgShow(img_perspective)



