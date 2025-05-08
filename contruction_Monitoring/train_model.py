import os
import sys
from pathlib import Path

# Ruta absoluta al directorio de YOLOv5
YOLO_PATH = "D:/YOLO/construction_Monitoring/yolov5"
# Ruta absoluta al archivo data.yaml
DATA_PATH = "../data.yaml"

def train_model():
    os.chdir(YOLO_PATH)

    command = (
        f"python train.py --img 640 --batch 16 --epochs 50 "
        f"--data {DATA_PATH} --weights yolov5s.pt --name casco_seguridad"
    )


    print("Entrenando modelo YOLOv5 con el siguiente comando:")
    print(command)
    os.system(command)

if __name__ == "__main__":
    train_model()
