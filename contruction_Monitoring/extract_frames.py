# extract_frames.py
import cv2
import os
from pathlib import Path
import argparse

def extract_frames(video_path, output_dir, rate=1):
    """
    Extrae frames de un video a una carpeta, útil para generar datasets para YOLOv11.

    Args:
        video_path (str): Ruta del video de entrada.
        output_dir (str): Carpeta donde guardar los frames.
        rate (int): Número de frames por segundo a guardar.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"❌ No se pudo abrir el video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(fps / rate))

    count = 0
    saved_count = 0

    print(f"🎥 Procesando {video_path} | Guardando 1 frame cada {frame_interval} frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_interval == 0:
            frame_path = output_dir / f"frame_{saved_count:04d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            saved_count += 1

        count += 1

    cap.release()
    print(f"✅ {saved_count} frames extraídos a {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extraer frames de un video para usar con YOLOv11.")
    parser.add_argument("--video", type=str, required=True, help="Ruta del video de entrada")
    parser.add_argument("--output", type=str, required=True, help="Carpeta donde se guardarán los frames")
    parser.add_argument("--rate", type=int, default=1, help="Frames por segundo a guardar")

    args = parser.parse_args()
    extract_frames(args.video, args.output, args.rate)
