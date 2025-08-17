import cv2 as cv

def show(title, img):
    cv.imshow(title, img)

def save_image(img, path):
    cv.imwrite(path, img)