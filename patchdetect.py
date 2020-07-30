import numpy as np
import cv2
from .aruco import ArUco
from .warping import warping
from .segmentation import find_length, KMeansSeg


class FindPatch():
    def __init__(self):
        self.aruco = ArUco()
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def detect_patch(self, frame):
        # Find red patch and return contour+length from cv2 image
        # black = np.full(frame.shape, 0.)

        rectangle, ids = self.aruco.detect(frame)
        cnt_img = frame
        warped = None
        patch = None
        length = 0
        cnt_img = cv2.putText(frame.copy(), f'{ids}', (0, 25),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
        if rectangle is not None:
            cnt_img = cv2.drawContours(cnt_img, rectangle, -1, (0, 255, 0), 3)

            warped = warping(frame, rectangle)
            if warped is not None:

                # patch = KMeansSeg(warped, 2)
                # patch = find_length(warped)
                rect, length = find_length(warped)

                if rect is not None:
                    x, y, w, h = rect
                    start = (x, y)
                    end = (x+w, y+h)
                    patch = cv2.rectangle(
                        warped.copy(), start, end, (0, 255, 0), 2)
                    cv2.putText(patch, f'{length:.0f}px', (0, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if patch is not None:
            if patch.shape[0] < patch.shape[1]:
                patch = np.rot90(patch)
            aspect_ratio = 2/7
            height = int(frame.shape[0])
            width = int(aspect_ratio*height)
            dim = (width, height)
            patch = cv2.resize(patch, dim, interpolation=cv2.INTER_AREA)
            return patch
        else:
            return frame
