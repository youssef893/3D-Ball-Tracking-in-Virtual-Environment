import cvzone
from cvzone.ColorModule import ColorFinder
import cv2 as cv
import socket

from imgaug import augmenters as iaa
cap = cv.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
success, img = cap.read()
h, w, _ = img.shape
color_finder = ColorFinder(False)
hsv_values = {'hmin': 11, 'smin': 158, 'vmin': 148, 'hmax': 179, 'smax': 255, 'vmax': 255}

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_address_port = ('127.0.0.1', 5053)

while True:
    success, img = cap.read()
    img_color, mask = color_finder.update(img, hsv_values)
    img_contours, countors = cvzone.findContours(img, mask)

    if countors:
        data = countors[0]['center'][0], \
               h - countors[0]['center'][1], \
               int(countors[0]['area'])
        print(data)
        sock.sendto(str.encode(str(data)), server_address_port)
    # img_stack = cvzone.stackImages([img, img_color, mask, img_contours], 2, 0.5)
    # cv.imshow('image', img_stack)
    img_contours = cv.resize(img_contours, (0, 0), None, 0.3, 0.3)
    cv.imshow("ImageContour", img_contours)
    cv.waitKey(1)
