import cv2
import numpy as np
import pandas as pd

# Paleta de colores por clase (coincide con data.yaml)
COLOR_MAP = {
    'zapatos_de_seguridad': (0, 255, 255),
    'casco': (0, 255, 0),
    'chaleco': (255, 165, 0),
    'guantes': (255, 0, 255),
    'trabajador': (100, 100, 255),
    'desconocido': (0, 0, 255)
}

def draw_bounding_boxes(frame, detections, names):
    """
    Dibuja los bounding boxes y etiquetas en el frame.

    Args:
        frame (np.ndarray): Frame del video.
        detections (list): Lista de detecciones (x1, y1, x2, y2, obj_id, cls_id).
        names (list|dict): Lista o diccionario de nombres de clases.

    Returns:
        np.ndarray: Frame con anotaciones visuales.
    """
    for (x1, y1, x2, y2, obj_id, cls_id) in detections:
        try:
            label = names[int(cls_id)].lower()
        except (IndexError, KeyError, TypeError):
            label = 'desconocido'

        color = COLOR_MAP.get(label, COLOR_MAP['desconocido'])

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(frame.shape[1] - 1, x2)
        y2 = min(frame.shape[0] - 1, y2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label} ID:{obj_id}"
        cv2.putText(frame, text, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame


def generate_report(data, output_path):
    """
    Genera un reporte en CSV con los resultados del análisis.

    Args:
        data (list): Lista de tuplas (ID, Objeto, Tiempo Detectado, Duración).
        output_path (str): Ruta para guardar el reporte CSV.
    """
    df = pd.DataFrame(data, columns=['ID', 'Objeto', 'Tiempo Detectado (ms)', 'Duración Estimada (s)'])
    df.to_csv(output_path, index=False)


def convert_bbox_to_center(x1, y1, x2, y2):
    """
    Convierte las coordenadas del bounding box al formato centro-ancho-alto.

    Returns:
        tuple: (cx, cy, width, height)
    """
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1
    return cx, cy, width, height


def convert_predictions_to_detections(prediction, tracker=None, conf_threshold=0.25):
    """
    Convierte las predicciones de YOLOv8 en una lista de detecciones para visualización.

    Args:
        prediction: Resultado de model.predict() (una sola imagen).
        tracker: Objeto de seguimiento con método 'update(detections)', opcional.
        conf_threshold (float): Umbral de confianza mínimo.

    Returns:
        list: Lista de detecciones [(x1, y1, x2, y2, id, cls_id), ...]
    """
    detections = []
    boxes = prediction.boxes
    if boxes is not None:
        for i in range(len(boxes)):
            conf = float(boxes.conf[i].cpu().numpy())
            if conf < conf_threshold:
                continue

            b = boxes.xyxy[i].cpu().numpy()
            cls_id = int(boxes.cls[i].cpu().numpy())
            x1, y1, x2, y2 = b
            detections.append((x1, y1, x2, y2, -1, cls_id))  # ID -1 si no hay tracking

    # Si hay tracker, aplicar seguimiento
    if tracker:
        detections = tracker.update(detections)

    return detections
