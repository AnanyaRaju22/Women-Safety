import cv2
import numpy as np

def preprocess_face(frame, face_coords):
    x, y, w, h = face_coords
    face = frame[y:y+h, x:x+w]
    face = cv2.resize(face, (64, 64))
    face = face / 255.0
    face = np.expand_dims(face, axis=0)
    return face
