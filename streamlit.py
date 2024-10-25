from ultralytics import YOLO
import streamlit as st
import cv2
from PIL import Image
import numpy as np
import tempfile
from main_old import processVideo, load_class_names,  save_video  # Import hàm và model bạn đã định nghĩa

# Đường dẫn lưu trữ
PROCESSED_FOLDER = 'static/tracked_videos/'
CHART_FOLDER = 'charts/'
CSV_FOLDER = 'csv/'

model = YOLO("models/yolov8x/yolov8x.pt")
class_file = 'classes_name.txt'

# Load class names
class_names = load_class_names(class_file)

def main():
    st.title('Traffic Monitoring')

    st.sidebar.title(" :red[Setting] ")

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{width: 250px;}
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{width: 400px; margin-left: -480px}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.sidebar.header("Initialize tracking area")
    corA_text = st.sidebar.text_input("Coordinate A")
    corB_text = st.sidebar.text_input("Coordinate B")
    corC_text = st.sidebar.text_input("Coordinate C")
    corD_text = st.sidebar.text_input("Coordinate D")

    # Chuyển đổi text thành tuple của số nguyên
    corA = tuple(map(int, corA_text.split(',')))
    corB = tuple(map(int, corB_text.split(',')))
    corC = tuple(map(int, corC_text.split(',')))
    corD = tuple(map(int, corD_text.split(',')))

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

        with st.spinner("Processing video, please wait..."):

            # Thực hiện phân tích video
            processed_video, vehicle_counts, congestion_rates, avg_speeds, occupancy_densities = processVideo(tfile.name, model, class_names, corA, corB, corC, corD)

            # Lưu video đã xử lí
            saved_video = save_video(processed_video, uploaded_video.name, PROCESSED_FOLDER)

        # Hiển thị video đã xử lý
        st.subheader("Processed Video")
        st.video(saved_video)

        # Hiển thị kết quả
        st.subheader("Vehicle Count and Metrics")
        total_vehicles = sum(vehicle_counts)
        avg_speed = avg_speeds[-1] if avg_speeds else 0
        occupancy_density = occupancy_densities[-1] if occupancy_densities else 0
        congestion_rate = congestion_rates[-1] if congestion_rates else 0
        
        # Hiển thị các thông số phân tích
        st.write(f"**Total Vehicles**: {total_vehicles}")
        st.write(f"**Average Speed**: {avg_speed:.2f} km/h")
        st.write(f"**Occupancy Density**: {occupancy_density:.2f}%")
        st.write(f"**Congestion Rate**: {congestion_rate:.2f}%")

    else:
        st.sidebar.write("Upload a video file to start processing")

if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass