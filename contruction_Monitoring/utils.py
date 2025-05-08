# utils.py

"""
Módulo de Utilidades

Autor: [Tu Nombre]
Fecha: [Fecha Actual]
Versión: 1.0
Descripción:
    - Funciones auxiliares para conversión de coordenadas, generación de reportes
      y visualización de resultados.
"""

import cv2
import numpy as np
import os
import pandas as pd

def draw_bounding_boxes(frame, detections):
    """
    Dibuja los bounding boxes en el frame.

    Args:
        frame (np.ndarray): Frame del video.
        detections (list): Lista de detecciones (x1, y1, x2, y2, label).

    Returns:
        np.ndarray: Frame con los bounding boxes dibujados.
    """
    for (x1, y1, x2, y2, label) in detections:
        color = (0, 255, 0) if label in ['casco', 'chaleco', 'zapato'] else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame

def generate_report(data, output_path):
    """
    Genera un reporte en formato CSV con los resultados del análisis.

    Args:
        data (list): Lista de datos a incluir en el reporte.
        output_path (str): Ruta de salida del reporte.
    """
    df = pd.DataFrame(data, columns=['ID', 'Objeto', 'Tiempo Detectado', 'Duración (s)'])
    df.to_csv(output_path, index=False)

def convert_bbox_to_center(x1, y1, x2, y2):
    """
    Convierte las coordenadas del bounding box a formato centro-ancho-alto.

    Returns:
        tuple: (cx, cy, width, height)
    """
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1
    return cx, cy, width, height