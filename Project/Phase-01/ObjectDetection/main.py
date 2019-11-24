# -*- coding:utf-8 -*-

import numpy as np
import cv2


# 获取特征点
def detection(img):
    descriptor = cv2.xfeatures2d.SIFT_create()
    (kps, features) = descriptor.detectAndCompute(img, None)
    return kps, features


# 特征点匹配
def matchKeyPoints(kpsA, kpsB, featureA, featureB, ratio, reprojThresh):
    matcher = cv2.BFMatcher_create()
    rawMatches = matcher.knnMatch(featureA, featureB, 2)

    matches = []
    for m in rawMatches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            matches.append(m[0])

    if len(matches) > 4:
        ptsA = np.float32([kpsA[m.queryIdx].pt for m in matches])
        ptsB = np.float32([kpsB[m.trainIdx].pt for m in matches])
        M, mask = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)

        return M, mask, matches

    return None


# FLANN匹配器
def matchKeyPointsWithFLANN(kpsA, kpsB, featureA, featureB, ratio, reprojThresh):
    index_params = dict(algorithm=1, tree=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    rawMatches = flann.knnMatch(featureA, featureB, 2)

    matches = []
    for m in rawMatches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            matches.append(m[0])

    if len(matches) > 4:
        ptsA = np.float32([kpsA[m.queryIdx].pt for m in matches])
        ptsB = np.float32([kpsB[m.trainIdx].pt for m in matches])
        M, mask = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)

        return M, mask, matches

    return None


def drawMatches(imgA, imgB, kpsA, kpsB, M, matches):
    hA, wA = imgA.shape[:2]
    hB, wB = imgB.shape[:2]
    vis = np.zeros((max(hA, hB), wA + wB, 3), dtype='uint8')
    vis[0:hA, 0:wA] = imgA
    # vis[0:hB, wA:] = imgB

    pts = np.float32([[0, 0], [0, hA - 1], [wA - 1, hA - 1], [wA - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)
    x_max, y_max = np.float32(dst).reshape(4, 2).max(axis=0)
    x_min, y_min = np.float32(dst).reshape(4, 2).min(axis=0)

    cv2.rectangle(imgB, (x_min, y_min), (x_max, y_max), (0, 0, 255), 5, cv2.LINE_AA)
    vis[0:hB, wA:] = imgB
    # vis[0:hB, wA:] = cv2.polylines(imgB, [np.int32(dst)], True, (0, 0, 255), 10, cv2.LINE_AA)

    matchesMask = mask.ravel().tolist()
    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=5)
    cv2.drawMatches(imgA, kpsA, imgB, kpsB, matches, outImg=vis, **draw_params)

    cv2.namedWindow('vis', 0)
    cv2.imshow('vis', vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return vis


if __name__ == '__main__':
    local_img = cv2.imread('image/local.jpeg')
    global_img = cv2.imread('image/global.jpeg')

    kps_l, feature_l = detection(local_img)
    kps_g, feature_g = detection(global_img)

    H = matchKeyPoints(kps_l, kps_g, feature_l, feature_g, ratio=0.75, reprojThresh=4.0)
    # H = matchKeyPointsWithFLANN(kps_l, kps_g, feature_l, feature_g, ratio=0.75, reprojThresh=4.0)
    if H:
        M, mask, matches = H
        drawMatches(local_img, global_img, kps_l, kps_g, M, matches)
