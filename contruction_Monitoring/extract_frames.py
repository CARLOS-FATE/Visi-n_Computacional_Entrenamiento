import cv2
import os

video_path = "./dataset/images/sample_video.mp4"
output_dir = "./dataset/images/frames"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imwrite(f"{output_dir}/frame_{count:04d}.jpg", frame)
    count += 1

cap.release()
print(f"{count} frames extraídos a {output_dir}")
