# src/distance.py

FOCAL = 250
REAL_HEIGHT = {
    "person": 170, "chair": 90, "couch": 90, "tvmonitor": 60,
    "laptop": 20, "dining table": 75, "bottle": 25
}

def estimate_distance(name, bbox):
    x1, y1, x2, y2 = bbox
    pixel_h = max(1, y2 - y1)
    real = REAL_HEIGHT.get(name, 100)
    return (real * FOCAL) / pixel_h / 100
