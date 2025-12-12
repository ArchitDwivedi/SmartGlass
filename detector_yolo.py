import cv2
from ultralytics import YOLO

class YOLODetector:
    def __init__(self):
        # custom trained model
        self.custom_model = YOLO("runs/train/archit_detector5/weights/best.pt")
        # fallback COCO model
        self.base_model = YOLO("yolov8n.pt")

        self.custom_names = self.custom_model.names
        self.base_names = self.base_model.names

    def detect(self, frame):
        detections = []

        # -------- Try CUSTOM MODEL First --------
        results = self.custom_model(frame, verbose=False)

        archit_found = False
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                score = float(box.conf[0])
                cls = int(box.cls[0])

                if score < 0.35:
                    continue

                name = self.custom_names[cls]
                detections.append({"name": name, "bbox": [x1,y1,x2,y2], "confidence": score})

                if name.lower() == "archit":
                    archit_found = True

        # -------- If no Archit detected, run base YOLO --------
        if not archit_found:
            fallback = self.base_model(frame, verbose=False)
            for r in fallback:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    score = float(box.conf[0])
                    cls = int(box.cls[0])

                    if score < 0.50:
                        continue

                    detections.append({
                        "name": self.base_names[cls],
                        "bbox": [x1, y1, x2, y2],
                        "confidence": score
                    })

        return detections
