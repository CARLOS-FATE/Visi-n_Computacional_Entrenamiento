# monitor.py
import os
import cv2
import yaml
import datetime
from pathlib import Path

from detection import process_frame  
from report_generator import generate_report

# ==================== CONFIGURACIÓN ====================
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

VIDEO_PATH = config['system']['video_path']
OUTPUT_PATH = config['system']['output_path']
LOG_PATH = config['system']['log_path']

Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
Path(LOG_PATH).mkdir(parents=True, exist_ok=True)

# ==================== VALIDACIÓN EPP ====================
def validar_epp(detecciones):
    """
    Verifica por ID si usa los 3 EPP: casco, chaleco y zapatos.
    """
    estado_por_id = {}

    for x1, y1, x2, y2, obj_id, label in detecciones:
        if obj_id not in estado_por_id:
            estado_por_id[obj_id] = set()
        estado_por_id[obj_id].add(label)

    cumplimiento = {
    obj_id: 'Cumple' if all(epp in items for epp in ['casco', 'chaleco', 'zapatos_de_seguridad'])
    else 'No Cumple'
    for obj_id, items in estado_por_id.items()
    }


    return cumplimiento

# ==================== PROCESAMIENTO ====================
detections_log = []

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"❌ Error al abrir el video en {VIDEO_PATH}")
    exit()

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        processed_frame, detecciones = process_frame(frame)
    except Exception as e:
        print(f"⚠️ Error en frame {frame_count}: {e}")
        continue

    cumplimiento = validar_epp(detecciones)
    timestamp_s = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

    for x1, y1, x2, y2, obj_id, label in detecciones:
        estado = cumplimiento.get(obj_id, 'Desconocido')
        detections_log.append((obj_id, label, timestamp_s, estado))

        # Mostrar estado visualmente
        color = (0, 255, 0) if estado == 'Cumple' else (0, 0, 255)
        cv2.putText(processed_frame, estado, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Monitoreo - Validación EPP", processed_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()

# ==================== REPORTE FINAL ====================
if detections_log:
    try:
        reporte = generate_report(detections_log, OUTPUT_PATH)
        print(f"✅ Reporte EPP generado en: {reporte}")
    except Exception as e:
        print(f"❌ Error generando reporte: {e}")
else:
    print("⚠️ No se registraron detecciones para generar reporte.")
