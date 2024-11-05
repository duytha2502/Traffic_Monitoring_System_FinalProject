from ultralytics import YOLO
import streamlit as st
import cv2
from PIL import Image
import numpy as np
import tempfile
import pymongo
import gridfs
import os
import pandas as pd
from main_old import processVideo, load_class_names, save_video, save_video_heatmap, save_to_csv, plot_vehicle_pie_chart, plot_congestion_line_chart, plot_asp_ocp_chart  # Import hàm và model bạn đã định nghĩa

# Đường dẫn lưu trữ
PROCESSED_FOLDER = 'static/tracked_videos/'
HEATMAP_FOLDER = 'static/heatmaps/'
CHART_FOLDER = 'charts/'
CSV_FOLDER = 'csv/'

model = YOLO("models/yolov8x/yolov8x.pt")
class_file = 'classes_name.txt'

# Load class names
class_names = load_class_names(class_file)

# Kết nối tới MongoDB local
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["processed_videos"]
fs = gridfs.GridFS(db)

# Hàm lưu video vào MongoDB
def save_video_and_csv_to_mongodb(video_path, filename):
    # Lưu video vào MongoDB
    with open(video_path, "rb") as video_file:
        video_id = fs.put(video_file, filename=filename, metadata={'uploaded_by': st.session_state.email})

    # # Tạo DataFrame từ dữ liệu CSV
    # if not (len(average_speed) == len(occupancy) == len(congestion_rate)):
    #     raise ValueError("Các mảng phải có cùng độ dài")
    
    # timestamps = [round(i * 1/3, 2) for i in range(len(average_speed))]

    # df = pd.DataFrame({
    #     'Timestamp': timestamps,
    #     'Average Speed (km/h)': [round(speed, 2) for speed in average_speed],
    #     'Occupancy (%)': [round(occur, 2) for occur in occupancy],
    #     'Congestion Rate (%)': [round(rate, 2) for rate in congestion_rate],
    # })

    # # Lưu DataFrame vào tệp CSV tạm thời
    # csv_file_path = f"{filename}_data.csv"
    # df.to_csv(csv_file_path, index=False)

    # # Lưu tệp CSV vào MongoDB
    # with open(csv_file_path, "rb") as csv_file:
    #     csv_id = fs.put(csv_file, filename=csv_file_path)

    # # Xóa tệp CSV tạm thời nếu không cần thiết nữa
    # os.remove(csv_file_path)

    # return video_id, csv_id

def save_heatmap_to_mongodb(video_path, filename):
    # Lưu heatmap vào MongoDB
    with open(video_path, "rb") as video_file:
        video_id = fs.put(video_file, filename=filename, metadata={'uploaded_by': st.session_state.email})

st.markdown(
    """
    <style>
    .css-d1b1ld.edgvbvh6 
    {
        visibility: hidden;
    }
    .css-1v8iw7l.eknhn3m4
    {
        visibility: hidden;
    }
    .metric_label 
    {
        text-align: center;
    }
    .metric_number
    {
        text-align: center;
        color: red;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Main
st.title('📹 Traffic Monitoring')

# Sidebar
st.sidebar.title(" :red[Setting] ")

st.sidebar.header("Initialize Tracking Area")

col1, col2 = st.sidebar.columns(2)

with col1:
    corA_text = st.text_input("Coordinate A", value=None)
    corC_text = st.text_input("Coordinate C", value=None)
    
with col2:
    corB_text = st.text_input("Coordinate B", value=None)
    corD_text = st.text_input("Coordinate D", value=None)
# Chuyển đổi text thành tuple của số nguyên
try:
    corA = tuple(map(int, corA_text.split(','))) if corA_text else (0, 0)
    corB = tuple(map(int, corB_text.split(','))) if corB_text else (0, 0)
    corC = tuple(map(int, corC_text.split(','))) if corC_text else (0, 0)
    corD = tuple(map(int, corD_text.split(','))) if corD_text else (0, 0)
except ValueError:
    # Nếu có lỗi chuyển đổi, mặc định là (0, 0)
    corA, corB, corC, corD = (0, 0), (0, 0), (0, 0), (0, 0)

with col1:
    st.header("Initialize Max Speed")
    max_speed_num = st.number_input("Max speed (km/h)", value=None, step=1, min_value= 1)

with col2:
    st.header("Initialize Scaling")
    scale = st.number_input("Scaling (pixel->km)", value=None, format="%0.4f")

st.sidebar.header("Initialize Congestion Configuration")
cgs_spd = st.sidebar.number_input("Speed (km/h)", value=None, step=1, min_value= 1)
cgs_ocp = st.sidebar.slider("Occupancy (%)", 70, 100, 80, 1)   

st.sidebar.markdown('---')
stframe = st.empty()

# Upload video
uploaded_video = st.sidebar.file_uploader("Upload video", type= ["mp4", "mov", "avi", "asf", "m4v"])
tfile = tempfile.NamedTemporaryFile(suffix='.mp4', delete= False)

if uploaded_video is not None:
        tfile.write(uploaded_video.read())
        dem_vid = open(tfile.name, 'rb')
        demo_bytes = dem_vid.read()
        st.sidebar.video(demo_bytes)

        if st.sidebar.button("Start tracking", use_container_width=True):
            with st.spinner("Processing video, please wait..."):

                # Thực hiện phân tích video
                processed_video, processHeatmap, vehicle_count_each, vehicle_counts, congestion_rates, avg_speeds, occupancy_densities = processVideo(tfile.name, model, class_names, corA, corB, corC, corD, max_speed_num, scale, cgs_spd, cgs_ocp)

                # Lưu video đã xử lí
                saved_video, video_name = save_video(processed_video, uploaded_video.name, PROCESSED_FOLDER)

                 # Lưu heatmap
                save_video_heatmap, heatmap_name = save_video(processHeatmap, uploaded_video.name, HEATMAP_FOLDER)

                # Lưu kết quả vào MongoDB
                save_video_and_csv_to_mongodb(saved_video, video_name)
                save_heatmap_to_mongodb(save_video_heatmap, heatmap_name)

                # Lưu dữ liệu vào CSV
                save_to_csv(avg_speeds, occupancy_densities, congestion_rates, video_name, CSV_FOLDER)

                pie_chart =  plot_vehicle_pie_chart(vehicle_count_each, video_name, CHART_FOLDER)
                line_char = plot_congestion_line_chart(congestion_rates, video_name, CHART_FOLDER)
                hybrid_char = plot_asp_ocp_chart(avg_speeds, occupancy_densities, video_name, CHART_FOLDER)

            # Hiển thị video đã xử lý
            st.subheader("Processed Video")
            st.video(saved_video)

            # Hiển thị kết quả
            avg_speed = avg_speeds[-1] if avg_speeds else 0
            occupancy_density = occupancy_densities[-1] if occupancy_densities else 0
            congestion_rate = congestion_rates[-1] if congestion_rates else 0
            
            col1st, col2st, col3st, col4st = st.columns(4)
            
            # Hiển thị các thông số phân tích
            with col1st:
                st.write("<h5 class='metric_label'>Total Vehicles</h5>", unsafe_allow_html=True)
                st.write(f"<h3 class='metric_number'>{vehicle_counts[-1]}</h3>", unsafe_allow_html=True)
            with col2st:
                st.write("<h5 class='metric_label'>Average Speed</h5>", unsafe_allow_html=True)
                st.write(f"<h3 class='metric_number'>{avg_speed:.0f} km/h</h3>", unsafe_allow_html=True)
            with col3st:
                st.write("<h5 class='metric_label'>Occupancy</h5>", unsafe_allow_html=True)
                st.write(f"<h3 class='metric_number'>{occupancy_density:.2f} %</h3>", unsafe_allow_html=True)
            with col4st:
                st.write("<h5 class='metric_label'>Congestion</h5>", unsafe_allow_html=True)
                st.write(f"<h3 class='metric_number'>{congestion_rate:.2f} %</h3>", unsafe_allow_html=True)
            
            # Hiển thị biểu đồ
            st.subheader("Statistics")
            st.image(pie_chart, caption="Vehicle Radio", use_column_width=True)
            st.image(hybrid_char, caption="Average Speed and Occupancy over Time", use_column_width=True)
            st.image(line_char, caption="Congestion Rate", use_column_width=True)

else:
    st.sidebar.button("Start tracking",use_container_width=True)
