# Refine Mask : REMOVE EYES, LIPS, AND LARGE AREAS
# INPUT: raw acne mask from adaptive mask file
# TECHNIQUE: Contour area filtering + positional filtering(ROI :cheeks, forehead)
# OUTPUT: cleaned binary mask wiht only valid acne regions

import cv2 as cv
import numpy as np

def filter_acne_mask(acne_mask, original_image=None, min_area = 10, max_area=5000):
    contours, _ = cv.findContours(acne_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    refined_mask = np.zeros_like(acne_mask)

    for cnt in contours:
        area = cv.contourArea(cnt)
        # perimeter = cv.arcLength(cnt, True)
        # circularity = 4* np.pi * (area/(perimeter**2 + 1e-5))

        # if 15 < area < 800 and 0.4 < circularity < 1.2:
        #     cv.drawContours(refined_mask, [cnt], -1, 255, -1)

        if min_area < area < max_area:
            cv.drawContours(refined_mask, [cnt], -1, 255, -1)

    if original_image is not None:
        pass

    return refined_mask