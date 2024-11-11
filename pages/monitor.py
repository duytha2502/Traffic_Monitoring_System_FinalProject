from ultralytics import YOLO
import streamlit as st
from PIL import Image
import cv2
import io
import numpy as np
import tempfile
import pymongo
import gridfs
import os
import pandas as pd
from main_old import processVideo, load_class_names, save_video, plot_vehicle_pie_chart, capture_speeding_object  # Import hàm và model bạn đã định nghĩa

# Đường dẫn lưu trữ
PROCESSED_FOLDER = 'static/tracked_videos/'
# HEATMAP_FOLDER = 'static/heatmaps/'
# CHART_FOLDER = 'charts/'
# CSV_FOLDER = 'csv/'

model = YOLO("models/yolov8x/yolov8x.pt")
class_file = 'classes_name.txt'

# Load class names
class_names = load_class_names(class_file)

# Kết nối tới MongoDB local
client = pymongo.MongoClient("mongodb://localhost:27017/")
pv = client["processed_videos"]
data = pv["data"]
fs = gridfs.GridFS(pv)

vo = client["vehicles_overspeed"]
fsvo = gridfs.GridFS(vo)

# Hàm lưu video vào MongoDB
def save_video_to_mongodb(video_path, filename):
    # Lưu video vào MongoDB
    with open(video_path, "rb") as video_file:
        video_id = fs.put(video_file, filename=filename, metadata={'uploaded_by': st.session_state.email})

def save_data_to_mongodb(vehicle_count_each, vehicle_count, vehicle_inside_area, average_speed, occupancy, congestion_rate, video_name):
 
    # Kiểm tra độ dài của mảng
    if not (len(average_speed) == len(occupancy) == len(congestion_rate)):
        raise ValueError("Các mảng phải có cùng độ dài")
    
    # Tạo mảng timestamp với bước nhảy 1/3 giây
    timestamps = [round(i * 1/3, 2) for i in range(len(average_speed))]

    document = {
    "user_gmail": st.session_state.email,
    "video_name": video_name, 
    'timestamp': timestamps,
    "average_speed": average_speed,  
    "occupancy": occupancy,  
    "congestion_rate": congestion_rate,
    'vehicle_count_each': vehicle_count_each,
    'vehicle_count': vehicle_counts,
    'vehicle_inside_area': vehicle_inside_area
    }

    data.insert_one(document)


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

col1sb, col2sb = st.sidebar.columns(2)

with col1sb:
    corA_text = st.text_input("Coordinate A", value=None)
    corC_text = st.text_input("Coordinate C", value=None)
    
with col2sb:
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

with col1sb:
    st.header("Initialize Max Speed")
    max_speed_num = st.number_input("Max speed (km/h)", value=None, step=1, min_value= 1)

with col2sb:
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

st.markdown('---')
st.subheader('Initializing Settings')
col1, col2 = st.columns(2)
with col1:
    st.write("Coordinate A:", corA)
    st.write("Coordinate B:", corB)
    st.write("Coordinate C:", corC)
    st.write("Coordinate D:", corD)
with col2:
    st.write("Max Speed: ", max_speed_num, " km/h")
    st.write("Scaling: ", scale, " pixel/km")
    st.write("Congestion Speed: ", cgs_spd, " km/h")
    st.write("Occupancy: ", cgs_ocp, "%")

if uploaded_video is not None:
        tfile.write(uploaded_video.read())
        dem_vid = open(tfile.name, 'rb')
        demo_bytes = dem_vid.read()
        st.sidebar.video(demo_bytes)

        if st.sidebar.button("Start tracking", use_container_width=True):
            with st.spinner("Processing video, please wait..."):

                # Thực hiện phân tích video
                processed_video, vehicles_os, vehicle_count_each, vehicle_counts, vehicle_inside_area, congestion_rates, avg_speeds, occupancy_densities = processVideo(tfile.name, model, class_names, corA, corB, corC, corD, max_speed_num, scale, cgs_spd, cgs_ocp)

                # Lưu video đã xử lí
                saved_video, video_name = save_video(processed_video, uploaded_video.name, PROCESSED_FOLDER)

                # Lưu heatmap
                # save_video_heatmap, heatmap_name = save_video(processHeatmap, uploaded_video.name, HEATMAP_FOLDER)

                # Lưu video vào MongoDB
                save_video_to_mongodb(saved_video, video_name)
                capture_speeding_object(video_name, vehicles_os, fsvo)

                # Lưu kết quả vào CSV
                save_data_to_mongodb( vehicle_count_each, vehicle_counts, vehicle_inside_area, avg_speeds, occupancy_densities, congestion_rates, video_name)

                # pie_chart =  plot_vehicle_pie_chart(vehicle_count_each, video_name, CHART_FOLDER)
                # line_char = plot_congestion_line_chart(congestion_rates, video_name, CHART_FOLDER)
                # hybrid_char = plot_asp_ocp_chart(avg_speeds, occupancy_densities, video_name, CHART_FOLDER)

            st.subheader("Processed Video")

            # Tạo mảng timestamp với bước nhảy 1/3 giây
            timestamps = [round(i * 1/3, 2) for i in range(len(avg_speeds))]

            df = pd.DataFrame({
                'Timestamp': timestamps,
                'Average Speed (km/h)': [round(speed, 2) for speed in avg_speeds],
                'Occupancy (%)': [round(occur, 2) for occur in occupancy_densities],
                'Congestion Rate (%)': [round(rate, 2) for rate in congestion_rates],
                'Vehicle Count': [round(count, 0) for count in vehicle_counts],
            })

            # Hiển thị video đã xử lý
            st.video(saved_video)

            # Hiển thị kết quả          
            col1st, col2st, col3st, col4st = st.columns(4)
            with col1st:
                st.write("<h5 class='metric_label'>Total Vehicles</h5>", unsafe_allow_html=True)
                st.write(f"<h3 class='metric_number'>{vehicle_counts[-1]}</h3>", unsafe_allow_html=True)
            with col2st:
                st.write("<h5 class='metric_label'>Average Speed</h5>", unsafe_allow_html=True)
                st.write(f"<h3 class='metric_number'>{sum(avg_speeds)/len(avg_speeds):.0f} km/h</h3>", unsafe_allow_html=True)
            with col3st:
                st.write("<h5 class='metric_label'>Occupancy</h5>", unsafe_allow_html=True)
                st.write(f"<h3 class='metric_number'>{sum(occupancy_densities)/len(occupancy_densities):.2f} %</h3>", unsafe_allow_html=True)
            with col4st:
                st.write("<h5 class='metric_label'>Congestion</h5>", unsafe_allow_html=True)
                st.write(f"<h3 class='metric_number'>{sum(congestion_rates)/len(congestion_rates):.2f} %</h3>", unsafe_allow_html=True)
            
            chart_vehicle_inside_area = pd.DataFrame({
                'Timestamp (s)': timestamps,
                'Vehicles In Area': vehicle_inside_area
            })
            st.line_chart(chart_vehicle_inside_area, x="Timestamp (s)", y="Vehicles In Area")

            st.subheader("Statistic")
            st.dataframe(df, hide_index=True, use_container_width=True)

            # Hiển thị biểu đồ
            # st.subheader("Statistics")
            # st.image(pie_chart, caption="Vehicle Radio", use_column_width=True)
            # st.image(hybrid_char, caption="Average Speed and Occupancy over Time", use_column_width=True)
            # st.image(line_char, caption="Congestion Rate", use_column_width=True)

            st.subheader("Line Chart")
            chart_spd_ocu_cgs = pd.DataFrame({
                'Timestamp (s)': timestamps,
                'Average Speed (km/h)': avg_speeds,
                'Occupancy (%)': occupancy_densities,
                'Congestion Rate (%)': congestion_rates,
            })
            st.line_chart(chart_spd_ocu_cgs, x="Timestamp (s)", y=["Average Speed (km/h)", "Occupancy (%)", "Congestion Rate (%)"], color=["#0057f9", "#fff207", "#f90004"])

            col1bc, col2bc = st.columns(2)
            with col1bc:
                st.subheader("Bar Chart")
                chart_vehicle = pd.DataFrame({
                    'Vehicle Type': list(vehicle_count_each.keys()),
                    'Count': list(vehicle_count_each.values())
                })
                st.bar_chart(chart_vehicle, x='Vehicle Type', y='Count', height=500)

            with col2bc:
                st.subheader("Vehicle Ratio Chart")
                pie_chart = plot_vehicle_pie_chart(vehicle_count_each)
                st.pyplot(pie_chart)
  

else:
    st.sidebar.button("Start tracking",use_container_width=True)
