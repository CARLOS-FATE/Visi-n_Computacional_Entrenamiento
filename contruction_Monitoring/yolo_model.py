#uolo_model.py

from ultralytics import YOLO
import numpy as np
import yaml
import pathlib

class YoloModel:
    def __init__(self, model_path):
        # Carga el modelo YOLOv8/v11 desde la ruta especificada
        self.model = YOLO(model_path)
        self.names = self.model.names  

    def detect(self, frame, imgsz=640, conf_threshold=0.25):
        """
        Realiza detección sobre un frame usando YOLOv8/v11.

        Args:
            frame (np.ndarray): Imagen de entrada (BGR).
            imgsz (int): Tamaño de la imagen de entrada para redimensionamiento.
            conf_threshold (float): Umbral de confianza mínima.

        Returns:
            list: Lista de detecciones en formato [x1, y1, x2, y2, conf, class_id].
        """
        results = self.model.predict(source=frame, imgsz=imgsz, conf=conf_threshold, verbose=False)
        detections = []

        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)

                for box, conf, class_id in zip(boxes, confs, class_ids):
                    x1, y1, x2, y2 = box
                    detections.append([float(x1), float(y1), float(x2), float(y2), float(conf), int(class_id)])


        return detections

# === Cargar ruta del modelo desde config.yaml ===
config_path = pathlib.Path(__file__).parent / 'config.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

model_path = pathlib.Path(__file__).parent / config['model']['path']
model = YoloModel(str(model_path))
