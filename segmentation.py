import numpy as np
import cv2


def KMeansSeg(bgr, k):
    '''
        Carry out k-means clustering segmentation on image
        Args:
            bgr(ndarray): BGR image
            k(int): number of cluseters/segments desired
        Returns:
            seg_img(ndarray): a segmented image
    '''
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    img = hsv[:, :, 1]

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

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


def find_contour(bgr):
    '''
        Return biggest contour from BGR image
        Args:
            bgr(ndarray): BGR image
        Returns:
            max_area_cnt(List(ndarray)): Biggest contour detected
    '''
    seg_img = KMeansSeg(bgr, 2)
    seg_img = cv2.Canny(seg_img, 50, 100)
    cnts, hier = cv2.findContours(seg_img, cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
    area_cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    if len(area_cnts) > 0:
        max_area_cnt = area_cnts[0]
    else:
        max_area_cnt = None
    return max_area_cnt


def find_length(bgr):
    '''
        Return contour and maximum length of contour from a BGR image
        Args:
            bgr(ndarray): BGR image
        Returns:
            cnt(List(ndarray)): Biggest contour detected
            max(w,h): Maximum length of the contour
    '''
    cnt = find_contour(bgr)
    if cnt is not None:
        x, y, w, h = cv2.boundingRect(cnt)
        return (x, y, w, h), max(w, h)
    else:
        return None, None

    # return KMeansSeg(bgr, 3)
