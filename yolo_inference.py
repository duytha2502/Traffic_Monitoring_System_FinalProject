from ultralytics import YOLO 

model = YOLO('yolov8x')

results = model.predict('input_videos/demo2.mp4', save=True, stream=True, device='cuda')

print('=====================================')
for frame_results in results:
    # Xử lý từng frame_results tại đây
    print(frame_results)  # In ra kết quả của từng khung hình
    
    for box in frame_results.boxes:
        print(box)  # In ra thông tin từng hộp bao quanh (bounding box)