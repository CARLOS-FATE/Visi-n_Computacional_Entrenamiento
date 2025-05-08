# tracker.py

"""
Módulo de Seguimiento de Objetos

Autor: [Tu Nombre]
Fecha: [Fecha Actual]
Versión: 1.1
Descripción:
    - Realiza el seguimiento de objetos detectados por YOLOv5.
    - Asigna ID único a cada objeto.
    - Calcula la duración del movimiento y áreas de actividad.
"""

import cv2
import numpy as np
from collections import defaultdict

class ObjectTracker:
    def __init__(self):
        """
        Inicializa el tracker de objetos usando el algoritmo CSRT de OpenCV.
        """
        self.trackers = cv2.MultiTracker_create()
        self.object_data = defaultdict(lambda: {'start_time': None, 'end_time': None, 'moved': False})

    def update(self, detections, frame):
        """
        Actualiza el estado del tracker con nuevas detecciones.

        Args:
            detections (list): Lista de detecciones en formato (x1, y1, x2, y2, label)
            frame (np.ndarray): Frame actual para el seguimiento.

        Returns:
            list: Lista de objetos rastreados con sus IDs y posiciones.
        """
        tracked_objects = []
        for detection in detections:
            x1, y1, x2, y2, label = detection
            bbox = (x1, y1, x2 - x1, y2 - y1)

            # Crear nuevo tracker si no existe
            tracker = cv2.TrackerCSRT_create()
            self.trackers.add(tracker, frame, bbox)

            # Generar ID único
            obj_id = len(self.object_data) + 1
            self.object_data[obj_id]['start_time'] = cv2.getTickCount()
            tracked_objects.append((x1, y1, x2, y2, obj_id, label))

        return tracked_objects

    def reset(self):
        """
        Reinicia todos los trackers y datos almacenados.
        """
        self.trackers = cv2.MultiTracker_create()
        self.object_data.clear()
