import cv2
import numpy as np


class ArUco():
    '''
        ArUco detector
        Args:
        Attributes:
            dictionary: cv2 aruco dictionary, using 4x4 50bit code dictionary
            parameters: cv2 parameter module for aruco detection
            order_corners(List(int)): order of points taken from each aruco, 
                starting from top-left corner clockwise
    '''

    def __init__(self):
        # Initialize the detector parameters using default values
        self.dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        self.parameters = cv2.aruco.DetectorParameters_create()
        self.order_corners = [1, 0, 3, 2]

    def detect(self, img):
        '''
            Return the detection rectangle coordinates
            Args:
                img(ndarray): BGR image
            Returns:
                rectangle(ndarray): contour of the rectangle
                markerIds(nparray(int)): marker ids detected
        '''
        # Detect the markers in the image
        markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(
            img, self.dictionary, parameters=self.parameters)
        rectangle = None
        if markerIds is not None:
            if len(markerIds) == 4:
                markerIds = markerIds.flatten()
                rectangle = np.full((1, 4, 2), 0)
                for markerId, markerCorner in zip(markerIds, markerCorners):
                    corners = markerCorner.reshape(-1, 2)
                    corners = corners.astype(np.int)
                    corner_idx = self.order_corners[markerId-1]
                    corner = corners[corner_idx]
                    rectangle[0][markerId-1] = corner
        return rectangle, markerIds

# 1,0,2,1
