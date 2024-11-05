import cv2
import numpy as np
from ultralytics import YOLO

class Heatmap:
    def __init__(self, model_path, colormap=cv2.COLORMAP_JET):
        self.model = model_path
        self.colormap = colormap

    def generate_heatmap(self, frame):
        # Chạy dự đoán với YOLO
        results = self.model(frame)
        
        # Lấy heatmap từ đầu ra của YOLO
        heatmap = np.zeros_like(frame[:, :, 0], dtype=np.float32)
        for det in results[0].boxes:
            x1, y1, x2, y2 = map(int, det.xyxy[0])  # Toạ độ bounding box
            heatmap[y1:y2, x1:x2] = np.maximum(heatmap[y1:y2, x1:x2], det.conf)  # Tạo heatmap từ độ tin cậy

        # Chuẩn hoá heatmap để đảm bảo giá trị nằm trong khoảng [0, 255]
        heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = np.uint8(heatmap)  # Chuyển về kiểu dữ liệu uint8
        heatmap_colored = cv2.applyColorMap(heatmap, self.colormap)
        
        # Overlay heatmap lên frame gốc
        output_frame = cv2.addWeighted(frame, 0.5, heatmap_colored, 0.5, 0)
        return output_frame