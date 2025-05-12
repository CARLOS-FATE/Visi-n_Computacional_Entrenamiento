import os
import yaml
import cv2
from detection import process_frame
from report_generator import generate_report
from utils_custom import draw_bounding_boxes
import pathlib
from yolo_model import model  

# Cargar configuración
config_path = pathlib.Path(__file__).parent / 'config.yaml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

VIDEO_PATH = config['system']['video_path']
OUTPUT_PATH = config['system']['output_path']

# Verificar y crear ruta de salida si no existe
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Inicializar variables globales
detections_data = []

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video en {VIDEO_PATH}")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        try:
            processed_frame, frame_detections = process_frame(frame)

            # DIBUJAR BOUNDING BOXES AQUÍ
            frame_with_boxes = draw_bounding_boxes(processed_frame, frame_detections, model.names)

        except Exception as e:
            print(f"Error procesando el frame {frame_count}: {e}")
            continue

        # Almacenar detecciones para el reporte
        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        for det in frame_detections:
            x1, y1, x2, y2, obj_id, label = det
            detections_data.append((obj_id, label, timestamp_ms, 0))  # Duración a calcular después

        # Mostrar frame con boxes
        cv2.imshow("Monitoreo en Tiempo Real", frame_with_boxes)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Finalización por el usuario.")
            break

    cap.release()
    cv2.destroyAllWindows()

    # Verificar si se detectaron objetos
    if not detections_data:
        print("No se detectaron objetos para generar el reporte.")
        return

    # Generar reporte
    try:
        report_path = generate_report(detections_data, OUTPUT_PATH, class_names=model.names)
        print(f"Reporte generado exitosamente: {report_path}")
    except Exception as e:
        print(f"Error generando el reporte: {e}")


if __name__ == "__main__":
    main()
