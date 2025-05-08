import os
import cv2
import torch
import numpy as np
from tracker import ObjectTracker
from yolo_model import YOLOModel

# Configuración del modelo
MODEL_PATH = './weights/yolo11.pt'
DATA_PATH = './data.yaml'
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.4

# Inicializar modelo YOLO
yolo = YOLOModel(MODEL_PATH, DATA_PATH, CONFIDENCE_THRESHOLD, IOU_THRESHOLD)

# Inicializar tracker
tracker = ObjectTracker()

def process_frame(frame):
    """
    Procesa un frame de video para detectar implementos de seguridad y evaluar carga de trabajo.
    """
    # Detección con YOLO
    detections = yolo.detect(frame)
    # Seguimiento de objetos
    tracked_objects = tracker.update(detections)

    for obj in tracked_objects:
        # Extraer información del objeto
        x1, y1, x2, y2, obj_id, label = obj
        # Dibujar bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {obj_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame , tracked_objects

def main(video_path):
    """
    Función principal del sistema. Lee el video y procesa cada frame.
    """
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Procesar frame
        frame = process_frame(frame)
        # Mostrar frame procesado
        cv2.imshow("Detección de EPP y Carga de Trabajo", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = './dataset/sample_video.mp4'
    main(video_path)
