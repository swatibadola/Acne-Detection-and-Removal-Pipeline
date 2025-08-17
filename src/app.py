import gradio as gd
import cv2 as cv
import numpy as np
from src.utils.image_loader import read_and_resize
from src.utils.skin_mask_generator import generate_skin_mask
from src.mask_generator.adaptive_mask import extract_acne_candidates
from src.mask_generator.contour_filter import filter_acne_mask
from src.inpainting.inpaint import remove_acne

def process_image(image):
    # converting from PIL(Python image library/Pillow) to BGR img (uploaded img comes in as a PIL img in GRADIO)
    img = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)

    # Running the pipeline
    skin_mask = generate_skin_mask(img)
    skin_only_img = cv.bitwise_and(img, img, mask=skin_mask)

    acne_raw = extract_acne_candidates(skin_only_img)
    acne_mask = filter_acne_mask(acne_raw, img)

    # Rectangle exclusion
    x1, y1, x2, y2 = 100, 100, 200, 200
    exclusion_mask = np.zeros_like(acne_mask)
    cv.rectangle(exclusion_mask, (x1, y1), (x2, y2), 255, -1)

    # Circle exclusion
    cx, cy, radius = 300, 300, 40
    cv.circle(exclusion_mask, (cx, cy), radius, 255, -1)

    acne_mask = cv.bitwise_and(acne_mask, cv.bitwise_not(exclusion_mask))

    result = remove_acne(img, acne_mask)

    # converting back to RGB for GRADIO
    result_rgb = cv.cvtColor(result, cv.COLOR_BGR2RGB)
    return result_rgb


iface = gd.Interface(
    fn=process_image,
    inputs=gd.Image(type='pil', label='Upload Face Image'),
    outputs=gd.Image(type='numpy', label='Acne-Free Result'),
    title='Acne Removal With Computer Vision',
    description="Upload an image, click process, and download the result!"
)

if __name__ == '__main__':
    iface.launch()