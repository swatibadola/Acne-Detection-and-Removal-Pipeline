# Acne Detection & Removal Pipeline

An end-to-end computer vision pipeline to detect and remove acne using OpenCV, NumPy, and image inpainting—wrapped in a live Gradio demo. Deployed on Hugging Face Spaces.

[▶️ Try the live demo on Hugging Face](https://huggingface.co/spaces/swati156/acne-removal-project)  <br>
[💻 View the code on GitHub](https://github.com/swatibadola/Acne-Detection-and-Removal-Pipeline)

---

##  Project Overview

Acne removal in images often leads to over-smoothing or loss of natural skin texture. This project explores a more nuanced method:

- Detect acne-affected regions using skin segmentation, contour detection, and masking.<br>
- Use image inpainting to fill only those regions—preserving facial features and texture.<br>
- Provide an interactive interface via Gradio for real-time image testing.<br>

---

##  Tech Stack

| Component | Purpose |
|-----------|---------|
| **Python** | Core language for processing |
| **OpenCV** | Skin segmentation, mask generation |
| **NumPy** | Efficient numerical operations |
| **Image Inpainting** | Reconstructing skin texture |
| **Gradio** | Web-based interface for demo |
| **Hugging Face Spaces** | Deployment & hosting of live app |

---

## Current Limitations

1) Masks sometimes include lips or nose regions—leading to imperfect results.<br>
2) Works best on clear, front-facing images under consistent lighting.<br>
3) Skin tone diversity and complex lighting conditions still pose challenges.
