from ultralytics import YOLO 

model = YOLO('models/yolov8x/best_vehicle_dectect.pt')

results = model.predict('input_videos/demo2.mp4', save=True, stream=True, device='cuda')

print('=====================================')
for frame_results in results:
    # Xử lý từng frame_results tại đây
    print(frame_results)  # In ra kết quả của từng khung hình