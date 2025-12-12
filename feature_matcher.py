import cv2
import os
import numpy as np

class FeatureMatcher:
    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=2000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.reference_features = {}

        self.load_reference_images()

    def load_reference_images(self):
        ref_dir = "custom_images"
        for filename in os.listdir(ref_dir):
            path = os.path.join(ref_dir, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            key, ext = os.path.splitext(filename)
            kp, des = self.orb.detectAndCompute(img, None)
            if des is not None:
                self.reference_features[key] = des

        print(f"[INFO] Loaded {len(self.reference_features)} custom objects.")

    def match(self, crop):
        """
        Match cropped YOLO region with custom images.
        Returns best matching label if matches > threshold.
        """
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)
        if des is None:
            return None

        best_label = None
        best_score = 0

        for label, ref_des in self.reference_features.items():
            matches = self.matcher.match(des, ref_des)
            score = len(matches)

            if score > best_score:
                best_score = score
                best_label = label

        # threshold to control accuracy
        if best_score > 30:
            return best_label

        return None
