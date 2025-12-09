import cv2
import time

from src.camera import get_camera
from src.detector_yolo import YOLODetector
from src.distance import estimate_distance
from src.speaker import Speaker


def main():
    cap = get_camera()
    detector = YOLODetector()
    speaker = Speaker()

    last_speak = 0.0
    delay = 2.0  # seconds between speech

    print("Starting Indoor AI Glasses... Press 'q' to quit.")
    print("Custom classes:", getattr(detector, "custom_names", None))
    print("Base classes:", getattr(detector, "base_names", None))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera not detected!")
            break

        # ---------- YOLO DETECTIONS ----------
        detections = detector.detect(frame)
        # Debug:
        # print("DETECTIONS:", detections)

        spoken_caption = ""

        # ---------- DISTANCE + DRAW BOXES ----------
        for obj in detections:
            name = obj["name"]
            x1, y1, x2, y2 = obj["bbox"]

            # distance estimation
            dist = estimate_distance(name, obj["bbox"])
            obj["distance"] = dist

            # Draw box and label with distance
            label = f"{name} {dist:.1f}m"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            # build spoken caption from FIRST detection
            if spoken_caption == "":
                spoken_caption = f"{name} {dist:.1f} meters ahead"

        # ---------- SPEAKER ----------
        if spoken_caption and (time.time() - last_speak > delay):
            print("Speaking:", spoken_caption)
            speaker.say(spoken_caption)
            last_speak = time.time()

        # ---------- DISPLAY ----------
        cv2.imshow("Indoor AI Glasses View", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # cleanup
    speaker.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
