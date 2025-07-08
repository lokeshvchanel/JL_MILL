from ultralytics import YOLO
import torch
import time

# Load model
model = YOLO('yolov8n.pt')  # You can also use your custom model
dummy_img = 'https://ultralytics.com/images/bus.jpg'

# CPU
start_cpu = time.time()
for _ in range(10):
    model.predict(dummy_img, device='cpu', verbose=False)
end_cpu = time.time()

# GPU
start_gpu = time.time()
for _ in range(10):
    model.predict(dummy_img, device='cuda', verbose=False)
end_gpu = time.time()

# Results
cpu_infer_time = end_cpu - start_cpu
gpu_infer_time = end_gpu - start_gpu

print(f"YOLOv8 CPU Time: {cpu_infer_time:.4f} s")
print(f"YOLOv8 GPU Time: {gpu_infer_time:.4f} s")
print(f"Speedup: {cpu_infer_time / gpu_infer_time:.2f}x")
