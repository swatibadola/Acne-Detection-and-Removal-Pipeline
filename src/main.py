from src.utils.image_loader import read_and_resize
from src.utils.visualization import show, save_image
# from src.preprocessing.color_conversion import bgr_to_lab, split_lab
from src.utils.skin_mask_generator import generate_skin_mask
from src.mask_generator.adaptive_mask import extract_acne_candidates
from src.mask_generator.contour_filter import filter_acne_mask
from src.inpainting.inpaint import remove_acne
import cv2 as cv
import numpy as np

#load img
img = read_and_resize('assets/download.jpeg')

# skin detection
skin_mask = generate_skin_mask(img)
skin_only_img = cv.bitwise_and(img, img, mask=skin_mask)

# acne candidate detection
acne_raw = extract_acne_candidates(skin_only_img)

# contour + region refinement
acne_mask = filter_acne_mask(acne_raw)

# Rectangle exclusion
x1, y1, x2, y2 = 100, 100, 200, 200
exclusion_mask = np.zeros_like(acne_mask)
cv.rectangle(exclusion_mask, (x1, y1), (x2, y2), 255, -1)
cv.imshow('Exclusion mask', exclusion_mask)

# Circle exclusion
cx, cy, radius = 300, 300, 40
cv.circle(exclusion_mask, (cx, cy), radius, 255, -1)

acne_mask = cv.bitwise_and(acne_mask, cv.bitwise_not(exclusion_mask))

# inpainting
result = remove_acne(img, acne_mask)

# saved final result
save_image(result, "output/result.png")

# debug visualization
show('Original', img)
show('Skin mask', skin_mask)
show('Skin only', skin_only_img)
show('Acne raw ', acne_raw)
show('Acne mask', acne_mask)
show('Result img', result)

print("Acne pixels:", cv.countNonZero(acne_mask))

cv.waitKey(0)
cv.destroyAllWindows()