import cv2 as cv

def bgr_to_hsv(img):
    return cv.cvtColor(img, cv.COLOR_BGR2HSV)

def bgr_to_lab(img):
    return cv.cvtColor(img, cv.COLOR_BGR2LAB)

def split_lab(img_lab):
    return cv.split(img_lab)