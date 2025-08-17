import cv2 as cv
import os

def read_and_resize(image_path, size=(550,600)):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")
    
    img = cv.imread(image_path)
    if img is None:
        raise ValueError(f"Unable to read the image: {image_path}")
    
    resized = cv.resize(img, size)
    return resized