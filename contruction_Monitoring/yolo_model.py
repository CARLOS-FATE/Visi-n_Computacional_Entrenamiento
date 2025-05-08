# yolo_model.py

"""
Módulo del Modelo YOLOv11

Autor: [Tu Nombre]
Fecha: [Fecha Actual]
Versión: 1.0
Descripción:
    - Carga el modelo YOLOv11 para detección de objetos.
    - Realiza predicciones y filtra resultados según los parámetros de confianza e IOU.
"""

import torch
import numpy as np

class YOLOModel:
    def __init__(self, model_path, data_path, conf_thres=0.5, iou_thres=0.4):
        """
        Inicializa el modelo YOLOv11.

        Args:
            model_path (str): Ruta del modelo entrenado (.pt).
            data_path (str): Ruta del archivo data.yaml.
            conf_thres (float): Umbral de confianza para las detecciones.
            iou_thres (float): Umbral IOU para el filtrado de detecciones.
        """
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, source='local')
        self.model.conf = conf_thres
        self.model.iou = iou_thres
        self.class_names = self._load_class_names(data_path)

    def _load_class_names(self, data_path):
        """
        Carga los nombres de las clases desde el archivo data.yaml.

        Args:
            data_path (str): Ruta del archivo data.yaml.

        Returns:
            list: Lista de nombres de clases.
        """
        import yaml
        from pathlib import Path

        data_path = Path(data_path).resolve()  # convierte a ruta absoluta
        with open(data_path, 'r') as file:
            data = yaml.safe_load(file)
        return data['names']


    def detect(self, frame):
        """
        Realiza la detección en un frame.

        Args:
            frame (np.ndarray): Frame del video.

        Returns:
            list: Lista de detecciones en formato (x1, y1, x2, y2, label).
        """
        results = self.model(frame)
        detections = []

        for det in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls_id = map(float, det)
            label = self.class_names[int(cls_id)]
            if conf >= self.model.conf:
                detections.append((int(x1), int(y1), int(x2), int(y2), label))

        return detections