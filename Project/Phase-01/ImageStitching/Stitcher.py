import numpy as np
import cv2


def stitch(images, ratio=0.75, reprojThresh=4.0, showMatches=False):
    # imageB, imageA = images
    imageA, imageB = images
    (kpsA, featuresA) = detectAndDescribe(imageA)
    (kpsB, featuresB) = detectAndDescribe(imageB)

    M = matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)
    if M is None:
        return None

    matches, H, status = M

    shft = np.array([[1.0, 0, imageA.shape[1]], [0, 1.0, 0], [0, 0, 1.0]])
    D = np.dot(shft, H)

    result = cv2.warpPerspective(imageA, D, (imageB.shape[1] + imageA.shape[1], max(imageA.shape[0], imageB.shape[0])))
    result[0:imageB.shape[0], imageA.shape[1]:] = imageB

    if showMatches:
        vis = drawMatches(imageA, imageB, kpsA, kpsB, matches, status)
        return result, vis


def detectAndDescribe(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    descriptor = cv2.xfeatures2d.SIFT_create()
    (kps, features) = descriptor.detectAndCompute(image, None)
    kps = np.float32([kp.pt for kp in kps])
    return (kps, features)


def matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
    # matcher = cv2.DescriptorMatcher_create("BruteForce")
    matcher = cv2.BFMatcher_create()

    rawMatches = matcher.knnMatch(featuresA, featuresB, 2)

    matches = []
    for m in rawMatches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            matches.append((m[1].queryIdx, m[0].trainIdx))

    if len(matches) > 4:
        ptsA = np.float32([kpsA[i] for i, _ in matches])
        ptsB = np.float32([kpsB[i] for _, i in matches])
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)

        return matches, H, status

    return None


def drawMatches(imageA, imageB, kpsA, kpsB, matches, status):
    (hA, wA) = imageA.shape[:2]
    (hB, wB) = imageB.shape[:2]
    vis = np.zeros((max(hA, hB), wA + wB, 3), dtype='uint8')
    vis[0:hA, 0:wB] = imageB
    vis[0:hB, wB:] = imageA

    for ((queryIdx, trainIdx), s) in zip(matches, status):
        if s == 1:
            if hA > queryIdx:
                ptA = (int(kpsA[queryIdx][0] + wB), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[queryIdx][0]), int(kpsB[queryIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

    return vis


if __name__ == '__main__':
    imageA = cv2.imread('image/left_01.png')
    imageB = cv2.imread('image/right_01.png')

    # imageA = cv2.imread('image/left_02.jpg')
    # imageB = cv2.imread('image/right_02.jpg')

    (result, vis) = stitch([imageA, imageB], showMatches=True)

    cv2.namedWindow('imageA', 0)
    cv2.imshow('imageA', imageA)
    cv2.namedWindow('imageB', 0)
    cv2.imshow('imageB', imageB)
    cv2.namedWindow('vis', 0)
    cv2.imshow('vis', vis)
    cv2.namedWindow('result', 0)
    cv2.imshow('result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()