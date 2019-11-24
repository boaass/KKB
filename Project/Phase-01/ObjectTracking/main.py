# -*-coding:utf-8-*-


import cv2
import numpy as np
import sys


# 获取特征点信息
def detection(img):
    descriptor = cv2.xfeatures2d.SIFT_create()
    kps, features = descriptor.detectAndCompute(img, None)
    return kps, features


def matchKeyPoints(kpsA, kpsB, featureA, featureB, ratio, reprojThresh):
    matcher = cv2.BFMatcher_create()
    rawMatches = matcher.knnMatch(featureA, featureB, 2)

    matches = []
    for m, n in rawMatches:
        if m.distance < n.distance * ratio:
            matches.append(m)

    if len(matches) > 4:
        ptsA = np.float32([kpsA[m.queryIdx].pt for m in matches])
        ptsB = np.float32([kpsB[m.trainIdx].pt for m in matches])
        M, mask = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)
        return M, mask, matches


def drawMatches(imgA, imgB, M):
    hA, wA = imgA.shape[:2]
    pts = np.float32([[0, 0], [0, hA - 1], [wA - 1, hA - 1], [wA - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    x_max, y_max = np.float32(dst).reshape(4, 2).max(axis=0)
    x_min, y_min = np.float32(dst).reshape(4, 2).min(axis=0)

    cv2.rectangle(imgB, (x_min, y_min), (x_max, y_max), (0, 0, 255), 5, cv2.LINE_AA)
    return imgB


if __name__ == '__main__':
    video = cv2.VideoCapture("videos/chaplin.mp4")
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    ok, frame = video.read()
    if not ok:
        print("Could not read video file")
        sys.exit()

    bbox = cv2.selectROI('Tracking', frame, False)
    image_obj = frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]

    while True:
        ok, frame = video.read()
        if not ok:
            break

        kpsA, featureA = detection(image_obj)
        kpsB, featureB = detection(frame)

        H = matchKeyPoints(kpsA, kpsB, featureA, featureB, 0.75, 4.0)
        if H:
            M, mask, matches = H
            dst_img = drawMatches(image_obj, frame, M)
            cv2.imshow('Tracking', dst_img)

            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break