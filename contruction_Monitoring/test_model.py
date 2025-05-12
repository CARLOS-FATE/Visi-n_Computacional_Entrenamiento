from ultralytics import YOLO
import cv2

model_path = "D:/YOLO/contruction_Monitoring/runs/detect/train8/weights/best.pt"
image_path = "D:/YOLO/contruction_Monitoring/test_image.jpg"

model = YOLO(model_path)
img = cv2.imread(image_path)

if img is None:
    print(f"❌ No se pudo cargar la imagen: {image_path}")
    exit()

results = model.predict(source=img, conf=0.25, save=False)

for result in results:
    boxes = result.boxes
    if boxes is not None:
        print(f"🔍 Detecciones encontradas: {len(boxes)}")
        for i in range(len(boxes)):
            class_id = int(boxes.cls[i].cpu().numpy())
            conf = float(boxes.conf[i].cpu().numpy())
            class_name = model.names[class_id] if hasattr(model, 'names') else f"Clase {class_id}"
            print(f" - Clase: {class_name}, Confianza: {conf:.2f}")
    else:
        print("⚠️ No se detectaron objetos.")

# Mostrar imagen con las detecciones
result_img = results[0].plot()
cv2.imshow("Detecciones", result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
