from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from datetime import datetime
import cv2
import imutils
import numpy as np
import os
import math
import cvzone
import torch

# Load MiDaS model
def load_midas_model():

    # Load MiDaS model for depth estimation
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    
    # Get the default transform
    transform = midas_transforms.small_transform
    
    # Move the model to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas.to(device).eval()  # Set model to evaluation mode

    return midas, transform, device

# Get depth map from the model
def get_depth_map(frame, depth_model):

    midas, transform, device = depth_model

    # Transform frame to the input size expected by the MiDaS model
    input_batch = transform(frame).to(device)

    # Run the model
    with torch.no_grad():
        prediction = midas(input_batch)

        # Resize to the original frame size
        depth_map = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bilinear",
            align_corners=False,
        ).squeeze()

        # Convert the depth map to a NumPy array
        depth_map = depth_map.cpu().numpy()

    return depth_map

# Tính toán khoảng cách dựa trên bounding box của object
def get_object_depth(bbox, depth_map):
    x1, y1, x2, y2 = bbox
    object_depth = np.mean(depth_map[y1:y2, x1:x2])
    return object_depth

def draw_bounding_boxes_on_depth_map(depth_colored, tracks):
    for track in tracks:
        if not track.is_confirmed():
            continue

        bbox = track.to_ltrb()
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(depth_colored, (x1, y1), (x2, y2), (255, 0, 0), 2)

    return depth_colored

def normalize_depth_map(depth_map):
    # Normalize the depth map to range [0, 1]
    depth_map = depth_map - np.min(depth_map)
    depth_map = depth_map / np.max(depth_map) if np.max(depth_map) > 0 else depth_map
    return depth_map

# Load class names from file
def load_class_names(class_file):
    with open(class_file, 'r') as f:
        class_names = f.read().splitlines()
    return class_names



def processVideo(video_path, model, class_names):

    # Load the MiDaS model for depth estimation
    depth_model = load_midas_model()

    # Initialize for speed calculation
    prev_centers = {} 
    time_interval = 1 / 30

    # Initialize for counting object
    counter_cache = []
    detection_classes= []
    vehicle_count = {
        "car": 0,
        "motorcycle": 0,
        "truck": 0,
        "bus": 0
    }

    # Initialize for tracking speed and occupancy
    total_speed = 0
    vehicle_with_speed_count = 0

    frame_counter = 0
    update_interval = 5

    occupancy_density = 0
    update_total_occupancy_density = 0

    all_speeds = {}
    update_total_avg_speed = 0

    congestion_rate = 0

    # Initialize frame
    frames = [] 

    # Read video
    cap = cv2.VideoCapture(video_path)

    # Initialize Deepsort 
    object_tracker = DeepSort(max_age= 3, n_init= 1)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_counter += 1
        # Resize frame
        frame_resized = cv2.resize(frame, (720, 480))

        results = model.predict(frame_resized,stream=False)

        detection_classes = results[0].names

        # Draw on frame
        frame_resized = draw_line(frame_resized)
        frame_resized = draw_metric(frame_resized)

        # Get the depth map using depth model
        depth_map = get_depth_map(frame_resized, depth_model)

        for result in results:
            for data in result.boxes.data.tolist():
                id = data[5]

                drawBox(data, frame_resized, detection_classes[id]) 
        
            details = get_details(result, frame_resized)

        tracks = object_tracker.update_tracks(details, frame=frame_resized)

        total_frame_area = frame_resized.shape[0] * frame_resized.shape[1]

        # Initialize for calculating each frame
        speeds = {}
        total_occupancy_density = 0
        total_avg_speed = 0
        for track in tracks:
            if not track.is_confirmed():
                break

            track_id = track.track_id
            class_id = track.get_det_class() 
            class_name = detection_classes[class_id]

            bbox = track.to_ltrb()
            x1, y1, x2, y2 = map(int, bbox)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            w = x2 - x1
            h = y2 - y1
            bbox_area = w * h

            # Get the depth of the object
            object_depth = get_object_depth((x1, y1, x2, y2), depth_map)

            # Calculate % of each bounding box for the total frame size 
            occupancy_density = round((bbox_area/ total_frame_area) * 100, 2)
            total_occupancy_density += occupancy_density

            # Calculate speed if we have the previous center for this object
            if track_id in prev_centers:
                prev_depth = prev_centers[track_id]['depth']
                
                # Check if object moving
                if prev_depth != object_depth:
                    speed = calculate_speed_with_depth(prev_depth, object_depth, time_interval)
                else:
                    speed = 0

                print(prev_depth, object_depth, speed, track_id)
                # Every 5 frame update speed values
                if frame_counter % update_interval == 0:
                    speeds[track_id] = speed
                    all_speeds.update(speeds)
                else:
                    pass
                
                # Show speed of each track
                if track_id in all_speeds:
                    speed = all_speeds[track_id]
                    cv2.putText(frame_resized, f'{speed:.2f} km/h', (x1, y1), cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 255, 0), 2)

                vehicle_with_speed_count += 1
                total_speed += speed

            # Update the center of the object
            prev_centers[track_id] = {'depth': object_depth, 'center': (x1, y1)}

            # Calculate average speed
            if vehicle_with_speed_count > 0:
                total_avg_speed = total_speed / vehicle_with_speed_count
            else:
                total_avg_speed = 0

            # Count each kind of vehicles
            if y1 > int(frame_resized.shape[0] / 2 ) and track_id not in counter_cache: 
                counter_cache.append(track_id)          
                if class_name == "car":
                    vehicle_count["car"] += 1
                elif class_name == "truck":
                    vehicle_count["truck"] += 1
                elif class_name == "bus":
                    vehicle_count["bus"] += 1
                elif class_name == "motorbike":
                    vehicle_count["motorbike"] += 1
 
            # Every 5 frame update the values
            if frame_counter % update_interval == 0:
                update_total_occupancy_density = total_occupancy_density 
                update_total_avg_speed = total_avg_speed
                congestion_rate = calculate_congestion(update_total_avg_speed, update_total_occupancy_density)
            else:
                pass

        cv2.putText(frame_resized, f'Total Vehicles: {sum(vehicle_count.values())}', (930, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame_resized, f'Car: {vehicle_count["car"]}', (930, 90), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame_resized, f'Motorcycle: {vehicle_count["motorcycle"]}', (930, 120), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame_resized, f'Truck: {vehicle_count["truck"]}', (1140, 90), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame_resized, f'Bus: {vehicle_count["bus"]}', (1140, 120), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2) 
        cv2.putText(frame_resized, f'Average Speed: {update_total_avg_speed:.2f}', (930, 180), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame_resized, f'Occupancy: {update_total_occupancy_density:.2f} %', (930, 220), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame_resized, f'Congestion: {congestion_rate:.2f} %', (930, 270), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
       
        processed_depth_map = process_depth_map(frame_resized, depth_map, tracks)

        # Combine depth map and original frame for display
        combined_frame = np.hstack((frame_resized, processed_depth_map))

        frames.append(frame_resized)
        #show frames
        cv2.imshow('VD', combined_frame)
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    return frames

def process_depth_map(frame_resized, depth_map, tracks):

    # Normalize depth map
    normalized_depth_map = normalize_depth_map(depth_map)

    # Scale depth map to uint8
    depth_map_scaled = (normalized_depth_map * 255).astype(np.uint8)

    # Resize depth map to match the size of frame_resized
    depth_map_resized = cv2.resize(depth_map_scaled, (frame_resized.shape[1], frame_resized.shape[0]))

    # Convert depth map to color image for visualization
    depth_colored = cv2.applyColorMap(cv2.convertScaleAbs(depth_map_resized, alpha=0.03), cv2.COLORMAP_JET)

    # Draw bounding boxes on the depth map
    depth_map_with_boxes = draw_bounding_boxes_on_depth_map(depth_colored, tracks)

    return depth_map_with_boxes

# Calculate speed based on bounding box centers
def calculate_speed_with_depth(prev_depth, curr_depth, time_interval):
    
    # Tính khoảng cách di chuyển
    distance_moved = abs(curr_depth - prev_depth)
    
    # Tính tốc độ
    speed = (distance_moved / time_interval) * 3.6  # Chuyển từ m/s sang km/h

    return speed

# Calculate congestion
def calculate_congestion(avg_speed, occupancy):
    congestion_rate = 0
    
    # Duyệt qua từng phần tử trong avg_speed và occupancy_density
    # for i in range(len(avg_speed)):

    # Tính So (Giả sử tốc độ trung bình cao nhất để So bằng 0 là 60km/h)
    So = max(10 * (1 - avg_speed / 60), 0)
    
    # Tính Oc (Giả sử Oc lớn nhất khi độ occupancy bằng 80% )
    Oc = min(10 * occupancy / 80, 10)
    
    # Tính congestion rate
    rate = (2 * So * Oc / (So + Oc)) * 10 if (So + Oc) > 0 else -1
    
        # Thêm vào danh sách congestion_rate
        # congestion_rate.append(rate)
    
    return rate

# Draw bounding boxes and labels
def drawBox(data, frame, name):
    x1, y1, x2, y2, conf, id = data
    w, h = x2 - x1, y2 - y1
    cvzone.putTextRect(frame, name, (int(x1)+ 4, int(y1) - 8), thickness=1, scale=0.4, colorT=(255, 255, 255), colorR=(255, 0, 0), font=cv2.FONT_HERSHEY_DUPLEX, offset=6)
    cvzone.cornerRect(frame, (int(x1), int(y1), int(w), int(h)), l=10, t=2, rt=1, colorR=(255, 0, 0), colorC=(0, 255, 0))

    return frame

# Details
def get_details(result, frame):

    classes = result.boxes.cls.cpu().numpy()
    conf = result.boxes.conf.cpu().numpy()   
    xywh = result.boxes.xywh.cpu().numpy()

    detections = []
    for i,item in enumerate(xywh):
        sample = (item,conf[i] ,classes[i])
        detections.append(sample)

    return detections

def draw_line(image):
    # depth = int(image.shape[0] / 2 )
    p1 = (100,int(image.shape[0]/2))
    p2 = (image.shape[1]-200,int(image.shape[0]/2))

    image = cv2.line(image, p1, p2, (0, 255, 0), thickness=6)

    return image

def draw_metric(image):

    overlay = image.copy()
    
    p1 = (900,0)
    p2 = (1280,400)

    cv2.rectangle(overlay, p1, p2, (64,64,64), -1 )
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

    return image

# Save output
def save_video(frames, input_video_path, output_video_path):

    base_filename = os.path.basename(input_video_path).split('.')[0]
    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    output_file = os.path.join(f"{output_video_path}", f"{base_filename}_{timestamp}.avi")

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    w, h = 720, 480

    out = cv2.VideoWriter(output_file, fourcc, 30, (w, h))
    for frame in frames:
        out.write(frame)
    out.release()

# Main execution
if __name__ == "__main__":

    # Initialize
    model = YOLO("yolov8x.pt")
    # model = YOLO("models/yolov8x/best.pt")
    input_video_path = "input_videos/demo6.mp4"
    output_video_path = "output_videos"
    class_file = 'classes_name.txt'

    # Load class names
    class_names = load_class_names(class_file)

    # Process the video
    processed_frames = processVideo(input_video_path, model, class_names)

    # Save the processed video
    save_video(processed_frames, input_video_path, output_video_path)

