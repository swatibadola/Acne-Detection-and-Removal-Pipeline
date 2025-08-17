# for initial acne candidate detection
# Input: Skin only image, non-skin regions are blacked out
#TECHNIQUE: Convert to grayscale + apply adaptive threshold or CLAHE + Otsu
#OUTPUT: Binary mask

import cv2 as cv
import numpy as np

def extract_acne_candidates(skin_only_img):
    # redness filter form lab space
    lab = cv.cvtColor(skin_only_img, cv.COLOR_BGR2LAB)
    l,a,b = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv.merge((l,a,b))

    #normalize lighting

    # redness mask => acne tends to ahve higher 'a' value
    thresh_value = np.percentile(a[a>0], 85)
    _, red_mask = cv.threshold(lab[:,:,1], thresh_value, 255, cv.THRESH_BINARY)

    # texture mask => find small dark blobs on light skin
    gray = cv.cvtColor(skin_only_img, cv.COLOR_BGR2GRAY)
    dog = cv.GaussianBlur(gray, (0,0), 2) - cv.GaussianBlur(gray, (0,0), 5)
    _, texture_mask = cv.threshold(dog, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    acne_candidates = cv.bitwise_and(red_mask, texture_mask)

    return acne_candidates