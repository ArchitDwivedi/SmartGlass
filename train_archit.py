from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")  # Base YOLOv8 model

    model.train(
        data="datasets/archit_only/Extracted/data.yaml",  # updated path
        epochs=50,
        imgsz=640,
        name="archit_detector",
        project="runs/train"
    )

if __name__ == "__main__":
    main()
