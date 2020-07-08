import numpy as np
import cv2


def KMeansSeg(img, k):
    if len(img.shape) > 2:
        vectorized = img.reshape((-1, 3))
    else:
        vectorized = img.reshape((-1, 1))
    vectorized = np.float32(vectorized)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = k
    attempts = 10
    ret, label, center = cv2.kmeans(
        vectorized, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    seg_img = res.reshape((img.shape))
    return seg_img


def segmentation(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    img = hsv[:, :, 1]
    seg_img = KMeansSeg(img, 2)
    # return seg_img
    seg_img = cv2.Canny(seg_img, 50, 100)
    cnts, hier = cv2.findContours(seg_img, cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
    area_cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    max_area_cnt = area_cnts[0]
    return max_area_cnt
    # cnt_img = cv2.drawContours(bgr, area_cnts, 0, (0, 255, 0), 1)
