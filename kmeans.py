import numpy as np
import cv2


class KMeansSeg():
    '''
        Perform k-means clustering segmentation
        Args:
        Attributes:
            criteria: stopping conditions for k-means clustering
    '''

    def __init__(self):
        self.criteria = (cv2.TERM_CRITERIA_EPS +
                         cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    def cluster(self, img, K=2):
        '''
            Perform clustering
            Args:
                img(ndarray): BGR image
                K(int, default to 2): number of segments desired
            Returns:
                seg_img(ndarray): segmented image
        '''
        vectorized = img.reshape((-1, 3))
        vectorized = np.float32(vectorized)
        attempts = 10
        ret, label, center = cv2.kmeans(
            vectorized, K, None, self.criteria, attempts, cv2.KMEANS_PP_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        seg_img = res.reshape((img.shape))
        return seg_img
