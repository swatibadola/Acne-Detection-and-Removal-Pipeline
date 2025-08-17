import cv2 as cv
import numpy as np

def generate_skin_mask(img):
    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    lower_skin = np.array([0,15,60], dtype='uint8')
    upper_skin = np.array([25,255,255], dtype='uint8')

    lower_skin2 = np.array([0,30,30], dtype='uint8')
    upper_skin2 = np.array([20,255,200], dtype='uint8')

    mask1 = cv.inRange(hsv_img, lower_skin, upper_skin)
    mask2 = cv.inRange(hsv_img, lower_skin2, upper_skin2)

    # kernel = np.ones((5,5), np.uint8)
    # cleaned_mask = cv.morphologyEx(skin_mask, cv.MORPH_CLOSE, kernel)
    skin_mask = cv.bitwise_or(mask1, mask2)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
    skin_mask = cv.morphologyEx(skin_mask, cv.MORPH_CLOSE, kernel, iterations=2)

    return skin_mask