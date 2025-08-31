import cv2
import numpy as np
from PIL import Image
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def get_images(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    faces, ids = [], []

    for imagePath in image_paths:
        gray_img = Image.open(imagePath).convert("L")
        img_np = np.array(gray_img, "uint8")
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces_detected = detector.detectMultiScale(img_np)

        for (x, y, w, h) in faces_detected:
            faces.append(img_np[y:y+h, x:x+w])
            ids.append(id)

    return faces, ids

faces, ids = get_images("dataset")
recognizer.train(faces, np.array(ids))
recognizer.save("trainer.yml")
