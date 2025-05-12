# detection.py

import numpy as np
from yolo_model import YoloModel
from tracker import ObjectTracker
from utils_custom import draw_bounding_boxes
import cv2

# Inicializar modelo y tracker
model = YoloModel("D:/YOLO/contruction_Monitoring/runs/detect/train8/weights/best.pt")  # O tu modelo personalizado
tracker = ObjectTracker(distance_threshold=30)

CLASS_NAMES = ['zapatos_de_seguridad', 'casco', 'chaleco', 'guantes', 'trabajador']
# Etiquetas de clase si las necesitas
LABELS = {
    0: "zapatos_de_seguridad",
    1: "casco",
    2: "chaleco",
    3: "guantes",
    4: "trabajador"
    # Agrega más si entrenaste otras clases
}


def process_frame(frame):
    """
    Procesa un frame: detección + tracking + visualización.

    :param frame: Imagen BGR
    :return: (frame con anotaciones, lista de detecciones [x1, y1, x2, y2, id, label])
    """
    
    raw_detections = model.detect(frame)

    detections = []
    for det in raw_detections:
        x1, y1, x2, y2, conf, class_id = det
        detections.append({
            "box": [x1, y1, x2, y2],
            "class_id": int(class_id),
            "confidence": float(conf)
        })

    tracked_objects = tracker.update(detections)

    output_detections = []
    for obj in tracked_objects:
        x1, y1, x2, y2 = obj["box"]
        obj_id = obj["id"]
        class_id = int(obj["class_id"])
        output_detections.append((int(x1), int(y1), int(x2), int(y2), obj_id, class_id))  # ← solo el ID

    annotated_frame = draw_bounding_boxes(frame.copy(), output_detections, names=LABELS)
    return annotated_frame, output_detections
