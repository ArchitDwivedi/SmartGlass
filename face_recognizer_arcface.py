import cv2
from deepface import DeepFace
import numpy as np

class ArcFaceRecognizer:
    def __init__(self):
        self.db_path = "faces"   # folder containing images
        print("ArcFace Model Loaded")

    def recognize_from_bbox(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        try:
            result = DeepFace.find(
                img_path=crop,
                db_path=self.db_path,
                model_name="ArcFace",
                enforce_detection=False
            )

            if len(result) > 0 and len(result[0]) > 0:
                identity = result[0].iloc[0]['identity']
                name = identity.split("\\")[-1].split(".")[0]
                return name

        except Exception as e:
            print("Recognition error:", e)

        return None
