from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from twilio.rest import Client
from datetime import datetime
import http.client
import json
import smtplib,ssl
import cv2
import imutils
import numpy as np
import os
import math
import cvzone

# Load class names from file
def load_class_names(class_file):
    with open(class_file, 'r') as f:
        class_names = f.read().splitlines()
    return class_names

def processVideo(video_path, model, class_names):

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
    update_interval = 10

    occupancy_density = 0
    update_total_occupancy_density = 0

    all_speeds = {}
    update_total_avg_speed = 0

    congestion_rate = 91

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
        frame_resized = cv2.resize(frame, (1280, 720))

        results = model.predict(frame_resized, stream=True, device='cuda')

        # Draw on frame
        frame_resized, pts = draw_area(frame_resized)
        area = polygon_area(pts)

        frame_resized = draw_metric(frame_resized)

        # detection_classes = results.names

        for result in results:
            for data in result.boxes.data.tolist():

                x1, y1, x2, y2 = int(data[0]), int(data[1]), int(data[2]), int(data[3])
                id = int(data[5])
                center = calculate_center(x1, y1, x2, y2)

                if is_inside_area((center[0],center[1]), pts):
                    drawBox(data, frame_resized, class_names[id]) 
                
            details = get_details(result, frame_resized)

        tracks = object_tracker.update_tracks(details, frame=frame_resized)

        total_frame_area = frame_resized.shape[0] * frame_resized.shape[1]

        # Initialize for calculating each frame
        speeds = {}
        total_occupancy_density = 0
        total_avg_speed = 0

        for track in tracks[:]:
            if not track.is_confirmed():
                break

            track_id = track.track_id
            class_id = int(track.get_det_class()) 

            bbox = track.to_ltrb()
            x1, y1, x2, y2 = map(int, bbox)
            w = x2 - x1 
            h = y2 - y1
            bbox_area = w * h  

            if not is_inside_area((x1, y1), pts):
            # Nếu track nằm ngoài vùng, xoá track này
                tracks.remove(track)
                continue

            # Calculate % of each bounding box for the total frame size 
            occupancy_density = round((bbox_area/ area) * 100, 2)
            total_occupancy_density += occupancy_density
 
            # Calculate speed if we have the previous center for this object
            if track_id in prev_centers:
                prev_center = prev_centers[track_id]
                
                # Check if object moving
                if prev_center != (x1, y1):
                    speed = calculate_speed(prev_center, (x1, y1), time_interval)
                else:
                    speed = 0

                # Every 5 frame update speed values
                if frame_counter % update_interval == 0:
                    speeds[track_id] = speed
                    all_speeds.update(speeds)
                else:
                    pass
                
                # Show speed of each track
                if track_id in all_speeds:
                    speed = all_speeds[track_id]
                    cv2.putText(frame_resized, f'{speed:.0f} km/h', (x1, y1), cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 255, 0), 2)

                vehicle_with_speed_count += 1
                total_speed += speed
            
            # Update the center of the object
            prev_centers[track_id] = (x1, y1)

            # Calculate average speed
            if vehicle_with_speed_count > 0:
                total_avg_speed = total_speed / vehicle_with_speed_count
            else:
                total_avg_speed = 0

            # Count each kind of vehicles
            if y1 > int(pts[-1][0][1] - 10) and track_id not in counter_cache: 
                counter_cache.append(track_id)          
                if class_names[class_id] == "car":
                    vehicle_count["car"] += 1
                elif class_names[class_id] == "truck":
                    vehicle_count["truck"] += 1
                elif class_names[class_id] == "bus":
                    vehicle_count["bus"] += 1
                elif class_names[class_id] == "motorbike":
                    vehicle_count["motorbike"] += 1

            # Every 5 frame update the values
            if frame_counter % update_interval == 0:
                update_total_occupancy_density = total_occupancy_density 
                update_total_avg_speed = total_avg_speed
                congestion_rate = calculate_congestion(update_total_avg_speed, update_total_occupancy_density)
            else:
                pass
        
        # Check and sending notification
        check_congestion_and_notify(congestion_rate)
        cv2.putText(frame_resized, f'Total Vehicles: {sum(vehicle_count.values())}', (930, 60), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame_resized, f'Car: {vehicle_count["car"]}', (930, 100), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame_resized, f'Motorcycle: {vehicle_count["motorcycle"]}', (930, 130), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame_resized, f'Truck: {vehicle_count["truck"]}', (1140, 100), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame_resized, f'Bus: {vehicle_count["bus"]}', (1140, 130), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2) 
        cv2.putText(frame_resized, f'Average Speed: {update_total_avg_speed:.0f} km/h', (930, 190), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame_resized, f'Occupancy: {update_total_occupancy_density:.2f} %', (930, 230), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame_resized, f'Congestion: {congestion_rate:.2f} %', (930, 280), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
        
        frames.append(frame_resized)

        #show frames
        cv2.imshow('VD', frame_resized)
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    return frames

# Calculate speed based on bounding box centers
def calculate_speed(prev_center, curr_center, time_interval):

    # Example: 1 pixel = 0,0001 km
    scale_factor = 0.0001

    dx = curr_center[0] - prev_center[0]
    dy = curr_center[1] - prev_center[1]
    distance_km = math.sqrt(math.pow(dx, 2) + math.pow(dy, 2)) * scale_factor
    speed_km = (distance_km / time_interval) * 3600

    return speed_km

# Calculate congestion
def calculate_congestion(avg_speed, occupancy):

    # Tính So (Giả sử tốc độ trung bình cao nhất để So bằng 0 là 60km/h)
    So = max(10 * (1 - avg_speed / 60), 0)
    
    # Tính Oc (Giả sử Oc lớn nhất khi độ occupancy bằng 90% )
    Oc = min(10 * occupancy / 90, 10)
    
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
def get_details(result, frame):

    classes = result.boxes.cls.cpu().numpy()
    conf = result.boxes.conf.cpu().numpy()   
    xywh = result.boxes.xywh.cpu().numpy()
    detections = []
    for i,item in enumerate(xywh):
        sample = (item,conf[i] ,classes[i])
        detections.append(sample)

    return detections

def draw_area(image):

    pts = np.array([[590, 350], [810, 350], 
                    [910, 650], [210, 650]],
                    np.int32)
    
    pts = pts.reshape((-1, 1, 2))

    image = cv2.polylines(image, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

    return image, pts

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
    p2 = (1280,400)

    cv2.rectangle(overlay, p1, p2, (64,64,64), -1 )
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

    return image

# Send SMS notification
def send_sms_alert(congestion_rate):

    account_sid = 'AC88457892b9e22f910fdeeaf0fe6c16d3'
    auth_token = '41627fe7f6045a8b462aa840476aa085'
    twilio_number = '+18039982438'
    recipient_number = '+84788024737'

    # Create Twilio client
    client = Client(account_sid, auth_token)

    # Send SMS
    # in body part you have to write your message
    message = client.messages.create(
        body= f"The congestion rate has reached {congestion_rate:.2f} %. Immediate action required.",
        from_=twilio_number,
        to=recipient_number
    )
    print("SMS sent successfully.")

# Send email alert
def send_email_alert(congestion_rate):
    smtp_server = "smtp.gmail.com"
    port = 587
    sender_email = "admin@gmail.com"
    receiver_email = "duythai2502@gmail.com"
    password = "wtjv ityr qyjc vbob"

    subject = "Congestion Alert!"
    body = f"The congestion rate has reached {congestion_rate:.2f} %. Immediate action required."

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
def check_congestion_and_notify(congestion_rate):
    if congestion_rate >= 90:
        print("Congestion rate has reached 90%! Sending alert...")
        # send_email_alert(congestion_rate)
        send_sms_alert(congestion_rate)
    else:
        print(f"Congestion rate is {congestion_rate}%. No need to send alert yet.")

# Save output
def save_video(frames, input_video_path, output_video_path):

    base_filename = os.path.basename(input_video_path).split('.')[0]
    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    output_file = os.path.join(f"{output_video_path}", f"{base_filename}_{timestamp}.avi")

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    w, h = 1280, 720

    out = cv2.VideoWriter(output_file, fourcc, 30, (w, h))
    for frame in frames:
        out.write(frame)
    out.release()

# Main execution
if __name__ == "__main__":

    # Initialize
    model = YOLO("models/yolov8x/yolov8x.pt")
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

