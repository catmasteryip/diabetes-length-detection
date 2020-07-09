import numpy as np
import cv2
from aruco import ArUco
from warping import warping
from segmentation import find_length

cap = cv2.VideoCapture(0)
aruco = ArUco()

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    black = np.full(frame.shape, 0.)

    rectangle, ids = aruco.detect(frame)
    cnt_img = frame
    warped = black
    patch = black
    cnt_img = cv2.putText(frame.copy(), f'{ids}', (0, 25),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
    if rectangle is not None:
        cnt_img = cv2.drawContours(cnt_img, rectangle, -1, (0, 255, 0), 3)

        warped = warping(frame, rectangle)
        if warped is not None:
            cnt, length = find_length(warped)
            if cnt is not None:
                patch = cv2.drawContours(
                    warped.copy(), cnt, -1, (0, 255, 0), 2)
                cv2.putText(patch, f'{length:.1f} px', (0, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

    cv2.imshow('aruco', cnt_img)
    cv2.imshow('warped', warped)
    cv2.imshow('patch', patch)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
