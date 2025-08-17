# take cleaned acne mask and inpaint acne pixels in original image
# INPUT - original image + final acne mask
# OUTPUT - acne free image

import cv2 as cv

def remove_acne(original_image, final_mask):
    inpainted = cv.inpaint(original_image, final_mask, 13, cv.INPAINT_TELEA)
    return inpainted