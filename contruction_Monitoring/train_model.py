# train_model.py

from ultralytics import YOLO
from pathlib import Path

# Configuración
DATA_PATH = Path("../data.yaml")  # Asegúrate de que este archivo esté en formato válido para YOLOv11
MODEL_ARCH = "yolov8s.pt"         # Puedes usar yolov8n.pt, yolov8m.pt, yolov8l.pt o tu propio .pt preentrenado
EPOCHS = 50
IMG_SIZE = 640
BATCH_SIZE = 16
PROJECT_NAME = "casco_seguridad"

def train_model():
    model = YOLO(MODEL_ARCH)  # Cargar modelo preentrenado o personalizado
    model.train(
        data=str(DATA_PATH),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project="runs/train",
        name=PROJECT_NAME,
        verbose=True
    )

    print(f"✅ Entrenamiento completado: runs/train/{PROJECT_NAME}")

if __name__ == "__main__":
    train_model()
