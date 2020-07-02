import numpy as np
import cv2
from qrdecoder import QRdecoder
from rectangle import rectangle
from kmeans import KMeansSeg

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    qrdecoder = QRdecoder()
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX

    # org
    org = (50, 50)

    # fontScale
    fontScale = 1

    # Blue color in BGR
    color = (0, 255, 0)

    # Line thickness of 2 px
    thickness = 2
    qr_img, center, data = qrdecoder.detect(frame)
    image = cv2.putText(qr_img, f'{center}', org, font,
                        fontScale, color, thickness, cv2.LINE_AA)
    cv2.imshow('qrcode center', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
