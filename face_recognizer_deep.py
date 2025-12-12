import cv2
import os
import numpy as np
from deepface import DeepFace

class DeepFaceRecognizer:
    def __init__(self, faces_dir="faces"):
        print("Loading SFace embeddings... (no TensorFlow)")

        self.model_name = "SFace"    # Lightweight ONNX model
        self.known_embeddings = []
        self.known_names = []

        for file in os.listdir(faces_dir):
            if file.lower().endswith(("jpg", "jpeg", "png")):
                path = os.path.join(faces_dir, file)

                try:
                    rep = DeepFace.represent(
                        img_path=path,
                        model_name=self.model_name
                    )[0]["embedding"]
                except Exception as e:
                    print("Face not detected in:", file, "--", e)
                    continue

                self.known_embeddings.append(np.array(rep))
                self.known_names.append(file.split(".")[0])

        print("Loaded faces:", self.known_names)

    def recognize_from_bbox(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)

        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            return "Unknown"

        try:
            rep = DeepFace.represent(
                face_crop,
                model_name=self.model_name
            )[0]["embedding"]
        except:
            return "Unknown"

        rep = np.array(rep)

        best_score = -1
        best_name = "Unknown"

        for emb, name in zip(self.known_embeddings, self.known_names):
            similarity = np.dot(rep, emb) / (np.linalg.norm(rep) * np.linalg.norm(emb))

            if similarity > best_score:
                best_score = similarity
                best_name = name

        # SFace threshold around 0.35~0.40 (cosine similarity)
        return best_name if best_score > 0.38 else "Unknown"
