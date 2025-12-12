import cv2
import torch
import numpy as np
from ultralytics import YOLO
import pyttsx3
import face_recognition
import os

# -------------------- YOLO Model --------------------
model = YOLO("yolov8s.pt")   # change to best.pt later (custom training)

# -------------------- Depth Model (MiDaS) --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large").to(device).eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform

# -------------------- Text to Speech --------------------
speaker = pyttsx3.init()
speaker.setProperty('rate', 170)

def speak(text):
    print("[SPEAK]:", text)
    speaker.say(text)
    speaker.runAndWait()

# -------------------- Navigation Direction --------------------
def get_direction(frame_width, x1, x2):
    frame_center = frame_width // 2
    object_center = (x1 + x2) // 2
    threshold = 50
    if abs(frame_center - object_center) < threshold:
        return "straight"
    elif object_center < frame_center:
        return "left"
    else:
        return "right"

# -------------------- Load Known Faces --------------------
known_faces = []
known_names = []

faces_dir = "faces"   # create folder with subfolders per person
if os.path.exists(faces_dir):
    for name_folder in os.listdir(faces_dir):
        folder_path = os.path.join(faces_dir, name_folder)
        for file in os.listdir(folder_path):
            img = face_recognition.load_image_file(os.path.join(folder_path, file))
            enc = face_recognition.face_encodings(img)[0]
            known_faces.append(enc)
            known_names.append(name_folder)

print(f"[INFO] Loaded {len(known_names)} known faces.")

# -------------------- Video Capture --------------------
cap = cv2.VideoCapture(0)

last_spoken = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ---------------- Detect Faces ----------------
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb)
    face_encodings = face_recognition.face_encodings(rgb, face_locations)

    detected_names = []
    for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_faces, encoding, tolerance=0.45)
        name = "Unknown"

        if True in matches:
            idx = matches.index(True)
            name = known_names[idx]

        detected_names.append(name)
        cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
        cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    # ---------------- YOLO Detect ----------------
    results = model(frame)
    detections = results[0].boxes

    # ---------------- Depth Map ----------------
    input_batch = midas_transforms(rgb).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)
        depth_map = prediction.squeeze().cpu().numpy()
        depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # ------------ Process Each Detection ----------
    for box in detections:
        cls = int(box.cls)
        label = model.names[cls]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # If person detected and we have face recognized → replace label
        if label == "person" and len(detected_names) > 0:
            label = detected_names[0]

        # Distance calculation
        depth_crop = depth_map[y1:y2, x1:x2]
        if depth_crop.size > 0:
            distance = np.median(depth_crop) / 10  # approx meters
        else:
            distance = 0

        direction = get_direction(frame.shape[1], x1, x2)

        text = f"{label} {distance:.1f} meters on your {direction}"

        # Speak only when new sentence
        if text != last_spoken:
            speak(text)
            last_spoken = text

        # Draw bounding box
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, text, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("AI Glasses Vision", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
