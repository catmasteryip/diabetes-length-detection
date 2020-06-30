import cv2
import numpy as np


class QRdecoder():
    '''
        A class object for qr code detection
        Args:
        Attributes:
            qrDecoder (cv2.QRCodeDetector): qrcode detector instance from cv2
    '''

    def __init__(self):
        self.qrDecoder = cv2.QRCodeDetector()

    def centroidnp(self, arr):
        '''
            Return cetroid coordinates from array
            Args:
                arr (list): array of coordinates
            Returns:
                cent_x (int): coordinate of x centroid
                cent_y (int): coordinate of y centroid

        '''
        length = arr.shape[0]
        sum_x = sum(arr[:, 0])
        sum_y = sum(arr[:, 1])
        cent_x = int(sum_x/length)
        cent_y = int(sum_y/length)
        return cent_x, cent_y

    def detect(self, img):
        '''
            Return qr code information for a BGR image
            Args:
                img (ndarray): BGR image
            Returns:
                qr_img (ndarray): BGR image with bounding box for qr code
                center (int): coordinate of the center of qr code
                data (str): data of the qr code 

        '''
        # Detect and decode the qrcode
        data, bbox, _ = self.qrDecoder.detectAndDecode(img)
        if len(data) > 0:
            print("Decoded Data : {}".format(data))
            bbox = bbox.reshape(-1, 2)
            bbox = np.uint8(bbox)
            for i, pt in enumerate(bbox):
                qr_img = cv2.line(img, tuple(pt), tuple(
                    bbox[i-1]), (255, 0, 0), 5)
            center = self.centroidnp(bbox)
            qr_img = cv2.circle(qr_img, center, 1, (0, 255, 0), 5)
            return qr_img, center, data
        else:
            return img, None, None
