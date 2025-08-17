# import mediapipe as mp
# import cv2 as cv
# import numpy as np

# def exclude_eyes_lips(skin_img, acne_mask):
#     mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)
#     h,w,_ = skin_img.shape
#     results = mp_face_mesh.process(cv.cvtColo)