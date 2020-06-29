import numpy as np
import cv2
import imutils
# from Scanner.scan import DocScanner

cap = cv2.VideoCapture(0)


def KMeansSeg(img, K=2):
    '''

    '''
    vectorized = img.reshape((-1, 3))
    vectorized = np.float32(vectorized)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    attempts = 10
    ret, label, center = cv2.kmeans(
        vectorized, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    seg_img = res.reshape((img.shape))
    return seg_img


def rectangle(img):
    '''
        Recognize a rectangle from a BGR image
        Args:
            img(ndarray): BGR image
        Returns:
            out(ndarray): BGR image of the rectangle recognized
            img(ndarray): the parameter image itself
    '''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 50, 100)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    # loop over the contours
    screenCnt = None
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # if our approximated contour has four points and is convex, then we
        # can assume that we have found our screen
        if len(approx) == 4 and cv2.isContourConvex(approx):
            screenCnt = approx
            break
    if screenCnt is not None:
        mask = cv2.drawContours(img, [screenCnt], 0, 's', -1)
        # Extract out the object and place into output image
        out = np.zeros_like(img)
        # TODO: filter only the rectangle where the test paper is in
        out[mask == 's'] = img[mask == 's']
        return out
    else:
        return img


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    frame = rectangle(frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
