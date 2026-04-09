from ultralytics import YOLO
import os

# Verificar que el archivo existe
model_path = "yolo11n.pt"
if os.path.exists(model_path):
    print(f"\tModelo encontrado en: {os.path.abspath(model_path)}")
    model = YOLO(model_path)
    print("\tModelo cargado correctamente")
else:
    print(f"\tModelo no encontrado en: {os.path.abspath(model_path)}")
