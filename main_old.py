from ultralytics import YOLO, solutions
from deep_sort_realtime.deepsort_tracker import DeepSort
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime, timedelta
from moviepy.editor import VideoFileClip
import time
import http.client
import json
import smtplib,ssl
import cv2
import imutils
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import math
import cvzone
import gridfs

def insert_heatmap_into_frame(frame, heatmap):
    # Resize heatmap nếu cần để phù hợp với kích thước của frame
    heatmap_resized = cv2.resize(heatmap, (frame.shape[1] // 4 + 80, frame.shape[0] // 4 + 40))

    # Xác định vị trí chèn heatmap (góc trên cùng bên trái)
    x_offset = 0
    y_offset = 0

    # Chèn heatmap vào frame chính
    frame[y_offset:y_offset + heatmap_resized.shape[0], x_offset:x_offset + heatmap_resized.shape[1]] = heatmap_resized
    return frame

# Load class names from file
def load_class_names(class_file):
    with open(class_file, 'r') as f:
        class_names = f.read().splitlines()
    return class_names

def processVideo(video_path, model, class_names, cora, corb, corc, cord, max_speed, scale, conges_spd, conges_ocp):

    heatmap = solutions.Heatmap(
    show=False,
    model='models/yolov8x/yolov8x.pt',
    colormap=cv2.COLORMAP_JET,
    )

    # Initialize for speed calculation
    prev_centers = {} 
    time_interval = 1 / 30

    # Initialize for counting object
    counter_cache = []
    detection_classes= []
    vehicle_count = {
        "car": 0,
        "truck": 0,
        "bus": 0,
        "motorbike": 0
    }
    update_vehicle_count = [0]
    total_tracks_inside_area = [0]

    # Initialize for tracking speed and occupancy
    frame_counter = 0
    frames_to_wait = 60 
    update_interval = 10

    occupancy_density = 0
    update_total_occupancy_density = [0]

    all_speeds = {}
    update_total_avg_speed = [0]

    congestion_rate = [0]
    congestion_log = []

    captured_tracks = set()

    temp_storage = []  # Danh sách tạm lưu hình ảnh
    vehicles_os = {}

    # Initialize frame
    frames = [] 
    heatmap_frames = []

    # Read video
    cap = cv2.VideoCapture(video_path)

    # Initialize Deepsort 
    object_tracker = DeepSort()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1

        # Resize frame
        frame_resized = cv2.resize(frame, (1280, 720))

        results = model.predict(frame_resized, stream=True, device='cuda')

        # Draw on frame
        frame_resized, pts, y_center = draw_area(frame_resized, cora, corb, corc, cord)

        # Heatmap
        heatmap_frame = heatmap.generate_heatmap(frame_resized)
        frame_with_heatmap = insert_heatmap_into_frame(frame_resized, heatmap_frame)

        area = polygon_area(pts)
        frame_resized = draw_metric(frame_resized)

        for result in results:
            for data in result.boxes.data.tolist():

                x1, y1, x2, y2, conf, id = int(data[0]), int(data[1]), int(data[2]), int(data[3]), data[4], int(data[5])

                if conf < 0.3 or class_names[id] == "person":
                    continue

                center = calculate_center(x1, y1, x2, y2)

                if is_inside_area((center[0],center[1]), pts):

                    drawBox(data, frame_resized, class_names[id])
                
            details = get_details(result, frame_resized, class_names)

        tracks = object_tracker.update_tracks(details, frame=frame_resized)

        total_frame_area = frame_resized.shape[0] * frame_resized.shape[1]

        # Initialize for calculating each frame
        total_occupancy_density = 0

        speeds = {}
        total_speed = 0
        vehicle_with_speed_count = 0
        total_avg_speed = 0
        tracks_inside_area = 0

        for track in tracks:
            if not track.is_confirmed():
                break

            track_id = track.track_id
            class_id = int(track.get_det_class()) 

            bbox = track.to_ltrb()
            x1, y1, x2, y2 = map(int, bbox)
            w = x2 - x1 
            h = y2 - y1
            bbox_area = w * h  
            cx2 = int(x1 + (w/2))
            cy2 = int(y1 + (h/2))
            cx1 = int(x1 - (w/2))
            cy1 = int(y1 - (h/2))

            if is_inside_area((x1, y1), pts):

                tracks_inside_area += 1

                # Calculate % of each bounding box for the total frame size 
                occupancy_density = round((bbox_area/ area) * 100, 2)
                total_occupancy_density += occupancy_density

                # Calculate speed if we have the previous center for this object
                if track_id in prev_centers:
                    prev_center = prev_centers[track_id]
                        
                    # Check if object moving
                    if prev_center != (x1, y1):
                        speed = calculate_speed(prev_center, (x1, y1, w, h), time_interval, scale)
                    else:
                        speed = 0
                    
                    # Every frame update speed values
                    if frame_counter % update_interval == 0:
                        if speed > max_speed:
                            if track_id not in captured_tracks:
                                vehicles = save_temp_images(temp_storage ,frame_resized, captured_tracks, track_id, speed, cx1, cy1, cx2, cy2)
                                vehicles_os = vehicles
                                captured_tracks.add(track_id)
                            else:
                                pass
                        
                        speeds[track_id] = speed
                        all_speeds.update(speeds)
                    
                    # Show speed of each track
                    if track_id in all_speeds:
                        speed = all_speeds[track_id]
                        if speed > max_speed:
                            cv2.putText(frame_resized, f'BREAK: {speed:.0f} km/h', (cx2 - 4, cy1 - 4), cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 255), 2)
                        else:
                            cv2.putText(frame_resized, f'{speed:.0f} km/h', (cx2 - 4, cy1 - 4), cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 128, 255), 2)

                    vehicle_with_speed_count += 1
                    total_speed += speed     

                # Update the center of the object
                prev_centers[track_id] = (x1, y1, w, h)

                # Calculate average speed
                if vehicle_with_speed_count > 0:
                    total_avg_speed = total_speed / vehicle_with_speed_count
                else:
                    total_avg_speed = 0

                # Count each kind of vehicles
                if y1 < y_center and track_id not in counter_cache: 
                    counter_cache.append(track_id)          
                    if class_names[class_id] in ["car", "truck", "bus", "motorbike"]:
                        vehicle_count[class_names[class_id]] += 1
            else: 
                if track_id in counter_cache:
                    tracks_inside_area -= 1
                    counter_cache.remove(track_id)  # Xoá khỏi cache để không trừ nhiều lần

                # Tiếp tục xử lý nếu cần thiết
                tracks.remove(track)
                continue
        
        # Every 10 frame update the values
        if frame_counter % update_interval == 0:
            
            update_total_occupancy_density.append(total_occupancy_density) 
            
            update_total_avg_speed.append(total_avg_speed)
            
            congestion_rate_result = calculate_congestion(update_total_avg_speed[-1], update_total_occupancy_density[-1], conges_spd, conges_ocp)
            congestion_rate.append(congestion_rate_result)

            update_vehicle_count.append(sum(vehicle_count.values()))

            total_tracks_inside_area.append(tracks_inside_area)
        else:
            pass
        
        # Check and sending notification
        congestion_log_total = check_congestion_and_notify(congestion_rate, frame_counter, frames_to_wait, congestion_log)

        cv2.putText(frame_resized, f'Total Vehicles: {sum(vehicle_count.values())}', (930, 60), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame_resized, f'Car: {vehicle_count["car"]}', (930, 100), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame_resized, f'Motorbike: {vehicle_count["motorbike"]}', (930, 130), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame_resized, f'Truck: {vehicle_count["truck"]}', (1140, 100), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame_resized, f'Bus: {vehicle_count["bus"]}', (1140, 130), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2) 
        cv2.putText(frame_resized, f'Average Speed: {update_total_avg_speed[-1]:.0f} km/h', (930, 190), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame_resized, f'Occupancy: {update_total_occupancy_density[-1]:.2f} %', (930, 230), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame_resized, f'Congestion: {congestion_rate[-1]:.2f} %', (930, 280), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)

        frames.append(frame_resized)
        heatmap_frames.append(heatmap_frame)

        #show frames
        cv2.imshow('VD', frame_resized)
        if cv2.waitKey(1) == ord("q"):
            break
        
    cap.release()
    cv2.destroyAllWindows()

    return frames, vehicles_os, vehicle_count, update_vehicle_count, total_tracks_inside_area, congestion_rate, update_total_avg_speed, update_total_occupancy_density, congestion_log_total

# Calculate speed based on bounding box centers
def calculate_speed(prev_center, curr_center, time_interval, scale):

    dx = curr_center[0] - prev_center[0]
    dy = curr_center[1] - prev_center[1]

    distance = math.sqrt(math.pow(dx, 2) + math.pow(dy, 2))
    distance_km = distance * scale  
    speed_kms = distance_km / time_interval
    speed_kmh = speed_kms * 3600

    return speed_kmh

def save_temp_images(temp_storage, frame, captured_tracks, track_id, speed, cx1, cy1, cx2, cy2):

    object_img = frame[cy1:cy2, cx1:cx2]

    # Mã hóa ảnh thành byte dữ liệu
    _, buffer = cv2.imencode('.png', object_img)
    img_bytes = buffer.tobytes()
    
    temp_storage.append({
        "track_id": track_id,
        "speed": speed,
        "image": img_bytes  # Bạn có thể chuyển đổi hình ảnh sang định dạng mong muốn
    })

    return temp_storage

def capture_speeding_object(video_name, temp_storage, fs):

    for record in temp_storage:

        # Tạo tên file và metadata để lưu vào MongoDB
        filename = f"vehicle_{record["track_id"]}_{record["speed"]:.2f}.png"
        metadata = {
            "video_name": video_name,
            "vehicle_id": record["track_id"],
            "speed": f"{record["speed"]:.2f}"
        }

        # Lưu hình ảnh vào GridFS
        file_id = fs.put(record["image"], filename=filename, metadata=metadata)

    temp_storage.clear() 

# Calculate congestion
def calculate_congestion(avg_speed, occupancy, conges_spd, conges_ocp):

    # Tính So (Giả sử tốc độ trung bình cao nhất để So bằng 0 là  = conges_spd)
    So = max(10 * (1 - avg_speed / conges_spd), 0)
    
    # Tính Oc (Giả sử Oc lớn nhất khi độ occupancy = conges_ocp )
    Oc = min(10 * occupancy / conges_ocp, 10)
    
    # Tính congestion rate
    rate = (2 * So * Oc / (So + Oc)) * 10 if (So + Oc) > 0 else -1
    
    return rate

# Draw bounding boxes and labels
def drawBox(data, frame, name):
    x1, y1, x2, y2, conf, id = data
    w, h = x2 - x1, y2 - y1
    cvzone.putTextRect(frame, name, (int(x1)+ 4, int(y1) - 8), thickness=1, scale=0.4, colorT=(255, 255, 255), colorR=(255, 0, 0), font=cv2.FONT_HERSHEY_DUPLEX, offset=6)
    cvzone.cornerRect(frame, (int(x1), int(y1), int(w), int(h)), l=10, t=2, rt=1, colorR=(255, 0, 0), colorC=(0, 255, 0))

    return frame

# Details
def get_details(result, frame, class_names):

    classes = result.boxes.cls.cpu().numpy()
    conf = result.boxes.conf.cpu().numpy()   
    xywh = result.boxes.xywh.cpu().numpy()
    detections = []
    for i,item in enumerate(xywh):
        # Lấy tên lớp dựa trên chỉ số của lớp
        class_name = class_names[int(classes[i])]

        # Bỏ qua các đối tượng thuộc lớp "person"
        if class_name == "person":
            continue

        # Thêm các đối tượng không phải "person" vào danh sách
        sample = (item, conf[i], classes[i])
        detections.append(sample)

    return detections

def draw_area(image, cora, corb, corc, cord):

    pts = np.array([[cora], [corb], 
                    [corc], [cord]],
                    np.int32)

    # Define a polygon area of demo2.mp4
    # pts = np.array([[650, 300], [950, 300], 
    #                 [850, 450], [10, 450]],
    #                 np.int32)

    # Define a polygon area of demo.mp4
    # pts = np.array([[600, 200], [850, 200], 
    #                 [700, 450], [10, 450]],
    #                 np.int32)
  
    pts = pts.reshape((-1, 1, 2))

    image = cv2.polylines(image, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

    y_values = pts[:, 0, 1]
    y_center = int(np.mean(y_values))

    return image, pts, y_center

def polygon_area(pts):

    pts = pts.reshape(-1, 2) 
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def calculate_center(x1, y1, x2, y2):
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    return cx, cy

def is_inside_area(center, pts):
    # Sử dụng cv2.pointPolygonTest để kiểm tra
    return cv2.pointPolygonTest(pts, center, False) >= 0 

def draw_metric(image):

    overlay = image.copy()
    
    p1 = (900,0)
    p2 = (1280,300)

    cv2.rectangle(overlay, p1, p2, (64,64,64), -1 )
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

    return image

# Send email alert
def send_email_alert(congestion_rate):
    smtp_server = "smtp.gmail.com"
    port = 587
    sender_email = "admin@gmail.com"
    receiver_email = "duythai2502@gmail.com"
    password = "wtjv ityr qyjc vbob"

    subject = "Congestion Alert!"
    body = f"The congestion rate has reached {congestion_rate[-1]:.2f} %. Immediate action required."

    # Set up the email message
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    # context = ssl.create_default_context()
    with smtplib.SMTP(smtp_server, port) as server:
        server.ehlo()
        server.starttls()
        server.login(receiver_email, password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
    print("Email sent successfully.")


# Check congestion rate
def check_congestion_and_notify(congestion_rate, frame_counter, frames_to_wait, congestion_log):

    if frame_counter % frames_to_wait == 0:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if congestion_rate[-1] >= 90:
            status = "Alert Sent"
            send_email_alert(congestion_rate)
        else:
            status = "No Alert Sent"
        
        congestion_log.append({
            "time": current_time,
            "congestion_rate": congestion_rate[-1],
            "status": status
        })

    return congestion_log

# # Line chart
# def plot_congestion_line_chart(congestion_rate, output_name, output_dir):
    
#     timestamps = np.arange(len(congestion_rate)) * 0.333

#     # Vẽ biểu đồ Line cho congestion rate
#     plt.figure(figsize=(10, 6))
#     plt.plot(timestamps, congestion_rate, color='red', label='Congestion Rate (%)')
#     plt.title('Congestion Rate over Time')
#     plt.xlabel('Time (seconds)')
#     plt.ylabel('Congestion Rate (%)')
#     plt.legend()
#     plt.grid(True)

#     # Tiêu đề cho biểu đồ
#     plt.title("Congestion rate Line Chart")

#     chart_path = os.path.join(f"{output_dir}", f"{output_name}")

#     # Tạo thư mục nếu chưa tồn tại
#     if not os.path.exists(chart_path):
#         os.makedirs(chart_path)

#     # Lưu biểu đồ vào thư mục với tên file phù hợp
#     chart_file = os.path.join(f"{chart_path}", f"{output_name}_line.png")
#     plt.savefig(chart_file)

#     # Hiển thị biểu đồ
#     # plt.show()
#     return chart_file

# # Average speed and occupancy chart
# def plot_asp_ocp_chart(average_speed, occupancy, output_name, output_dir):
    
#     timestamps = np.arange(len(average_speed)) * 0.333

#     fig, ax1 = plt.subplots(figsize=(10, 6))

#     # Biểu đồ Line cho tốc độ trung bình
#     ax1.plot(timestamps, average_speed, color='b', label='Average Speed (km/h)', linewidth=2)
#     ax1.set_xlabel('Time (seconds)')
#     ax1.set_ylabel('Average Speed (km/h)', color='b')
#     ax1.tick_params(axis='y', labelcolor='b')

#     # Biểu đồ Bar cho occupancy
#     ax2 = ax1.twinx()  # Tạo trục Y kép
#     ax2.plot(timestamps, occupancy, color='r', label='Occupancy (%)', linewidth=2)
#     ax2.set_ylabel('Occupancy (%)', color='r')
#     ax2.tick_params(axis='y', labelcolor='r')

#     # Tiêu đề và hiển thị biểu đồ
#     plt.title('Average Speed and Occupancy over Time')
#     fig.tight_layout()

#     chart_path = os.path.join(f"{output_dir}", f"{output_name}")

#     # Tạo thư mục nếu chưa tồn tại
#     if not os.path.exists(chart_path):
#         os.makedirs(chart_path)

#     # Lưu biểu đồ vào thư mục với tên file phù hợp
#     chart_file = os.path.join(f"{chart_path}", f"{output_name}_hybrid.png")
#     plt.savefig(chart_file)

#     # Hiển thị biểu đồ
#     # plt.show()
#     return chart_file

# Save to CSV
def save_to_csv(average_speed, occupancy, congestion_rate, file_name, output_dir):

    # Kiểm tra độ dài của mảng
    if not (len(average_speed) == len(occupancy) == len(congestion_rate)):
        raise ValueError("Các mảng phải có cùng độ dài")
    
    # Tạo mảng timestamp với bước nhảy 1/3 giây
    timestamps = [round(i * 1/3, 2) for i in range(len(average_speed))]

    # Tạo dataframe từ dữ liệu
    df = pd.DataFrame({
        'Timestamp':timestamps,
        'Average Speed (km/h)': [round(speed, 2) for speed in average_speed],
        'Occupancy (%)': [round(occur, 2) for occur in occupancy],
        'Congestion Rate (%)': [round(rate, 2) for rate in congestion_rate],
    })
    
    csv_file_name = os.path.join(output_dir, f"{file_name}_data.csv")
    
    # Ghi dataframe vào file CSV
    df.to_csv(csv_file_name, index=False)

    return csv_file_name, df

# Convert to MP4
def convert_to_mp4(input_file, output_file):
    # Load video file bằng MoviePy
    clip = VideoFileClip(input_file)
    
    # Lưu file với codec 'libx264' để chuyển sang MP4
    clip.write_videofile(output_file, codec="libx264")

# Save output
def save_video(frames, input_video_path, output_video_path):

    base_filename = os.path.basename(input_video_path).split('.')[0]
    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    output_name = f"{base_filename}_{timestamp}"
    output_file_xvid = os.path.join(output_video_path,  f"{output_name}.avi")
    output_file_mp4 = os.path.join(output_video_path, f"{output_name}.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    w, h = 1280, 720

    out = cv2.VideoWriter(output_file_xvid, fourcc, 30, (w, h))
    for frame in frames:
        out.write(frame)
    out.release()

    convert_to_mp4(output_file_xvid, output_file_mp4)

    if os.path.exists(output_file_xvid):
        os.remove(output_file_xvid)

    return output_file_mp4, output_name

# Save output
def save_video_heatmap(frames, input_video_path, output_video_path):

    base_filename = os.path.basename(input_video_path).split('.')[0]
    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    output_heatmap_name = f"{base_filename}_{timestamp}"
    output_heatmap_xvid = os.path.join(output_video_path,  f"{output_heatmap_name}_heatmap.avi")
    output_heatmap_mp4 = os.path.join(output_video_path, f"{output_heatmap_name}_heatmap.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    w, h = 1280, 720

    out = cv2.VideoWriter(output_heatmap_xvid, fourcc, 30, (w, h))
    for frame in frames:
        out.write(frame)
    out.release()

    convert_to_mp4(output_heatmap_xvid, output_heatmap_mp4)

    if os.path.exists(output_heatmap_xvid):
        os.remove(output_heatmap_xvid)

    return output_heatmap_mp4, output_heatmap_name

# Main execution
# if __name__ == "__main__":

#     # Initialize
#     model = YOLO("models/yolov8x/yolov8x.pt")
#     # model = YOLO("models/yolov8x/best.pt")
#     input_video_path = "input_videos/demo6.mp4"
#     output_video_path = "output_videos"
#     output_chart_path = "charts"
#     output_csv_path = "csv"
#     class_file = 'classes_name.txt'

#     # Load class names
#     class_names = load_class_names(class_file)

#     # Process the video
#     processed_frames, vehicle_count, congestion_rate, average_speed, occupancy = processVideo(input_video_path, model, class_names)

#     # Save the processed video
#     saved_video = save_video(processed_frames, input_video_path, output_video_path)
    
    # # Save charts
    # plot_vehicle_pie_chart(vehicle_count, saved_video, output_chart_path)
    # plot_congestion_line_chart(congestion_rate, saved_video, output_chart_path)
    # plot_asp_ocp_chart(average_speed, occupancy, saved_video, output_chart_path)
    
    # # Save data to CSV
    # save_to_csv(average_speed, occupancy, congestion_rate, vehicle_count,os.path.join(output_csv_path, f"{saved_video}_data.csv"))

