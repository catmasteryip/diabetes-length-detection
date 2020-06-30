import numpy as np
import cv2
from qrdecoder import QRdecoder
from rectangle import rectangle
from kmeans import KMeansSeg

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    contour_img, warped_img = rectangle(frame)
    cv2.imshow('qrcode', warped_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
