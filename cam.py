import numpy as np
import cv2
from aruco import ArUco
from warping import warping
from segmentation import segmentation

cap = cv2.VideoCapture(0)
aruco = ArUco()

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    black = np.full(frame.shape, 0.)

    rectangle, ids = aruco.detect(frame)
    cnt_img = frame
    warped = None
    patch = None
    if rectangle is not None:
        cnt_img = cv2.drawContours(frame.copy(), rectangle, -1, (0, 255, 0), 3)

        warped = warping(frame, rectangle)

    if warped is not None:
        cnt = segmentation(warped)
        patch = cv2.drawContours(warped, cnt, -1, (0, 255, 0), 2)
    else:
        warped = black
        patch = black
    cv2.imshow('aruco', cnt_img)
    cv2.imshow('warped', warped)
    cv2.imshow('patch', patch)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
