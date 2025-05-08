# run.py

"""
Script de Integración del Sistema de Detección y Monitoreo

Autor: [Tu Nombre]
Fecha: [Fecha Actual]
Versión: 1.0
Descripción:
    - Punto de entrada del sistema.
    - Integra los módulos de detección, seguimiento y generación de reportes.
"""

import os
import yaml
import cv2
from main import process_frame
from report_generator import generate_report
from utils import draw_bounding_boxes

# Cargar configuración
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

VIDEO_PATH = config['system']['video_path']
OUTPUT_PATH = config['system']['output_path']
CONF_THRESHOLD = config['model']['conf_thres']
IOU_THRESHOLD = config['model']['iou_thres']

# Verificar rutas
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Inicializar variables
detections_data = []

def main():
    """
    Función principal del sistema. Ejecuta la detección, seguimiento y generación de reportes.
    """
    cap = cv2.VideoCapture(VIDEO_PATH)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Procesar frame
        processed_frame, frame_detections = process_frame(frame)

        # Almacenar detecciones para el reporte
        for det in frame_detections:
            x1, y1, x2, y2, obj_id, label = det
            detections_data.append((obj_id, label, cap.get(cv2.CAP_PROP_POS_MSEC), 0))  # Duración calculada posteriormente

        # Mostrar frame
        cv2.imshow("Monitoreo en Tiempo Real", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Generar reporte
    generate_report(detections_data, OUTPUT_PATH)

if __name__ == "__main__":
    main()
