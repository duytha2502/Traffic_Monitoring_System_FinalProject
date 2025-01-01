from ultralytics import YOLO
from PIL import Image
from datetime import datetime
from streamlit_folium import st_folium
from connectDB import *
from plotly.subplots import make_subplots
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import folium
import random
import cv2
import io
import numpy as np
import tempfile
import pymongo
import gridfs
import os
import time
import pandas as pd
from main_old import processVideo, load_class_names, save_video, capture_speeding_object  # Import hàm và model bạn đã định nghĩa

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
db, fs = connect_to_mongo("processed_videos")
data = db["data"]
log = db["logs"]   

# Kết nối tới database "vehicles_overspeed"
vo_db, fsvo = connect_to_mongo("vehicles_overspeed")

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

    .ea3mdgi5 {
        width: 1500px;
        margin-top: -50px;
    }

    .e115fcil1 {
    margin-left: 15%;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# Tạo giao diện bản đồ với Folium
def create_map(lattitude, longtitude):
    map_center = [lattitude, longtitude]
    map_zoom = 18

    # Tạo bản đồ với Google Maps
    my_map = folium.Map(location=map_center, zoom_start=map_zoom)

    # Đánh dấu một địa điểm
    folium.Marker(location=map_center, popup="Camera").add_to(my_map)

    st_folium(my_map, use_container_width=True, height=300, returned_objects=[])

def plot_bottom_left(vehicle_inside_area, placeholder):

    for i in range(1, len(vehicle_inside_area) + 1):
        # Tạo DataFrame từ mảng
        data = pd.DataFrame({
            "Index": [0.3333 * j for j in range(i)],  # Tạo cột Index (thời gian hoặc chuỗi số liệu)
            "Vehicle Inside Area": vehicle_inside_area[:i]  # Dữ liệu từ mảng
        })

        # Tạo figure với trục phụ
        fig = make_subplots()

        fig.add_trace(
            go.Scatter(
                x=data["Index"],
                y=data["Vehicle Inside Area"],
                name="Vehicles",
                mode='lines+markers',
                line=dict(color='blue'),
                hovertemplate="Vehicles: %{y:.1f}<br>Time: %{x}<extra></extra>"
            )
        )

        fig.update_layout(
            title="Vehicles Inside Area Over Time",
            xaxis=dict(title="Time"),
            yaxis=dict(
                title="Vehicles Quantity",
                range=[0, max(vehicle_inside_area) * 1.2]
            ),
            hovermode="x unified",
            height=450
        )

        placeholder.plotly_chart(fig, use_container_width=True, key=f'{i+10}')

        time.sleep(0.333333)

def plot_bottom_right(speed, occupancy, congestion_rate):

    # Tạo DataFrame từ mảng
    data = pd.DataFrame({
        "Index": range(len(speed)), 
        "Average Speed": speed,
        "Occupancy": occupancy,
        "Congestion Rate": congestion_rate
    })

    # Tạo figure với trục phụ
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=data["Index"],
            y=data["Average Speed"],
            name="Avg Speed",
            mode='lines+markers',
            line=dict(color='darkblue'),
            hovertemplate="Speed: %{y:.1f} km/h<br>Time: %{x}<extra></extra>"
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=data["Index"],
            y=data["Occupancy"],
            name="Occupancy",
            mode='lines+markers',
            line=dict(color='green'),
            hovertemplate="Occupancy: %{y:.1f}%<br>Time: %{x}<extra></extra>"
        ),
        secondary_y=True
    )
    
    fig.add_trace(
        go.Scatter(
            x=data["Index"],
            y=data["Congestion Rate"],
            name="Congestion",
            mode='lines+markers',
            line=dict(color='orange'),
            hovertemplate="Congestion: %{y:.1f}%<br>Time: %{x}<extra></extra>"
        ),
        secondary_y=True
    )

    fig.add_trace(
        go.Scatter(
            x=data["Index"],
            y=[90] * len(data["Index"]),  
            mode="lines",
            line=dict(color="red", dash="dash"),
            name="Threshold (90%)",
            hoverinfo="skip"  
        ),
        secondary_y=True
    )

    fig.update_layout(
        title="Traffic Metrics Over Time",
        xaxis=dict(title="Time"),
        yaxis=dict(
            title="Speed (km/h)",
            range=[0, max(speed) * 1.2]
        ),
        yaxis2=dict(
            title="Percentage (%)",
            range=[0, max(max(occupancy), max(congestion_rate)) * 1.2],
            overlaying="y",
            side="right"
        ),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=380
    )
    return fig

def plot_bottom_left2(vehicle_counts_each):

    data = {
        "Vehicle Type": list(vehicle_counts_each.keys()),
        "Count": list(vehicle_counts_each.values())
    }
    
    # Tạo biểu đồ tròn
    fig = px.pie(
        data,
        names="Vehicle Type",
        values="Count",
        title="Vehicle Type Distribution",
        labels={"Vehicle Type": "Type of Vehicle", "Count": "Count"},
        hole=0.3  # Tạo dạng donut chart, 0.0 nếu muốn là pie chart thuần túy
    )
    
    # Thêm hiển thị giá trị % vào biểu đồ
    fig.update_traces(textinfo="percent+label")
    
    return fig

def plot_gauge(
    indicator_number, indicator_color, indicator_suffix, indicator_title, max_bound):
    fig = go.Figure(
        go.Indicator(
            value=indicator_number,
            mode="gauge+number",
            domain={"x": [0, 1], "y": [0, 1]},
            number={
                "suffix": indicator_suffix,
                "font.size": 24,
                "font.color": indicator_color,
                "font.weight": "bold"   
            },
            gauge={
                "axis": {"range": [0, max_bound], "tickwidth": 1, 'tickfont': {'size': 16}},
                "bar": {"color": indicator_color},
            },
            title={
                "text": indicator_title,
                "font": {"size": 24, "color": "#31333f", "weight": "bold"},
            },
        )
    )
    fig.update_layout(
        height=175,
        margin=dict(l=10, r=10, t=70, b=10, pad=8),
    )
    st.plotly_chart(fig, use_container_width=True)


def log_action(action, user_email, details=None):

    log_entry = {
        "action": action,
        "user_email": user_email,
        "details": details or {},
        "timestamp": datetime.now()
    }
    log.insert_one(log_entry)

# Hàm lưu video vào MongoDB
def save_video_to_mongodb(video_path, filename):
    # Lưu video vào MongoDB
    with open(video_path, "rb") as video_file:
        video_id = fs.put(video_file, filename=filename, metadata={'uploaded_by': st.session_state.email})

    return video_id

def display_video_from_mongodb(video_id):
    video_data = fs.get(video_id)
    video_bytes = video_data.read()
    st.video(video_bytes, autoplay=True)

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
    st.header("Max Speed")
    max_speed_num = st.number_input("Max speed (km/h)", value=None, step=1, min_value= 1)

with col2sb:
    st.header("Scaling")
    scale = st.number_input("Scaling (pixel->km)", value=None, format="%0.4f")

st.sidebar.header("Initialize Congestion Configuration")
cgs_spd = st.sidebar.number_input("Speed (km/h)", value=None, step=1, min_value= 1)
cgs_ocp = st.sidebar.slider("Occupancy (%)", 70, 100, 80, 1)   

camera_option = st.sidebar.selectbox(
    "Choose camera",
    ("Cổng trường Nguyễn Huệ", "Cổng sau Bệnh viện C", "PTZ Trang Phục biểu diễn Phương Trần")
)

if camera_option == "Cổng trường Nguyễn Huệ":
    lattitude = 16.0739639
    longtitude = 108.2152701
elif camera_option == "Cổng sau Bệnh viện C":
    lattitude = 16.0744001
    longtitude = 108.2170112
elif camera_option == "PTZ Trang Phục biểu diễn Phương Trần":
    lattitude = 16.0741637
    longtitude = 108.2165022

st.sidebar.markdown('---')
stframe = st.empty()

st.markdown('---')
st.subheader('Initializing Settings')
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.write("Coordinate A:", corA)
    st.write("Coordinate B:", corB)
with col2:
    st.write("Coordinate C:", corC)
    st.write("Coordinate D:", corD)
with col3:
    st.write("Max Speed: ", max_speed_num, " km/h")
    st.write("Scaling: ", scale, " pixel/km")
with col4:
    st.write("Congestion Speed: ", cgs_spd, " km/h")
    st.write("Occupancy: ", cgs_ocp, "%")

# Upload video
uploaded_video = st.sidebar.file_uploader("Upload video", type= ["mp4", "mov", "avi", "asf", "m4v"])
tfile = tempfile.NamedTemporaryFile(suffix='.mp4', delete= False)

if uploaded_video is not None:
    tfile.write(uploaded_video.read())
    dem_vid = open(tfile.name, 'rb')
    demo_bytes = dem_vid.read()
    # st.sidebar.video(demo_bytes)

    if st.sidebar.button("Start tracking", use_container_width=True):
        with st.spinner("Processing video, please wait..."):

            # Thực hiện phân tích video
            processed_video, vehicles_os, vehicle_count_each, vehicle_counts, vehicle_inside_area, congestion_rates, avg_speeds, occupancy_densities, congestion_log = processVideo(tfile.name, model, class_names, corA, corB, corC, corD, max_speed_num, scale, cgs_spd, cgs_ocp)

            # Lưu video đã xử lí
            saved_video, video_name = save_video(processed_video, uploaded_video.name, PROCESSED_FOLDER)

            # Lưu video vào MongoDB
            video_id = save_video_to_mongodb(saved_video, video_name)

            capture_speeding_object(video_name, vehicles_os, fsvo)

            # Lưu kết quả vào CSV
            save_data_to_mongodb( vehicle_count_each, vehicle_counts, vehicle_inside_area, avg_speeds, occupancy_densities, congestion_rates, video_name)

            # Lưu logs
            log_action("Upload", st.session_state["email"], f"User {st.session_state["email"]} uploaded video {video_id} to start tracking")

            st.toast("Video Processed Successfully!")
            
        st.subheader("Traffic Overview")
        # Hiển thị video đã xử lý
        with st.container(border=True, key='1'):
            col1db1, col2db1, col3db1, col4db1 = st.columns(4)
            with col1db1:
                plot_gauge(
                    indicator_number=vehicle_counts[-1],
                    indicator_color="blue",
                    indicator_suffix="",
                    indicator_title="Total Vehicles",
                    max_bound=100,  
                )
            with col2db1:
                plot_gauge(
                    indicator_number=round(sum(congestion_rates)/len(congestion_rates), 2),
                    indicator_color="red",
                    indicator_suffix="%",
                    indicator_title="Congestion Rate",
                    max_bound=100,  
                )
            with col3db1:
                plot_gauge(
                    indicator_number=round(sum(avg_speeds)/len(avg_speeds), 2),
                    indicator_color="green",
                    indicator_suffix="km/h",
                    indicator_title="Speed",
                    max_bound=100,  
                )
            with col4db1:
                plot_gauge(
                    indicator_number=round(sum(occupancy_densities)/len(occupancy_densities), 2),
                    indicator_color="green",
                    indicator_suffix="%",
                    indicator_title="Occupancy",
                    max_bound=100,  
                )
        with st.container(border=True, key='2'):
            col1db2, col2db2 = st.columns(2, gap='large')
            with col1db2:
                st.subheader("Processed Video")
                display_video_from_mongodb(video_id) 
            with col2db2:
                placeholder = st.empty()
            
            col1db3, col2db3 = st.columns(2, gap='large')
            with col1db3:
                create_map(lattitude, longtitude)   
            with col2db3:
                fig2 = plot_bottom_right(avg_speeds, occupancy_densities, congestion_rates)
                st.plotly_chart(fig2, use_container_width=True, key="data")

        with st.container(border=True, key='3'):    
            col1db4, col2db4 = st.columns([6,4], gap='large')
            with col1db4:

                fleet_data = [
                    {"icon": "🚗", "count": vehicle_count_each["car"], "type": "Car"},
                    {"icon": "🏍️", "count": vehicle_count_each["motorbike"], "type": "Motorcycle"},
                    {"icon": "🚛", "count": vehicle_count_each["truck"], "type": "Truck"},
                    {"icon": "🚌", "count": vehicle_count_each["bus"], "type": "Bus"},
                ]
                # Tạo layout card với Streamlit columns
                cols = st.columns(len(fleet_data))

                for col, data in zip(cols, fleet_data):
                    with col:
                        # Tạo card
                        st.markdown(f"""
                        <div style="
                            color: #31333f; 
                            padding: 15px; 
                            border-radius: 8px; 
                            text-align: left; 
                            display: flex; 
                            align-items: center;
                            justify-content: center; 
                            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.3);">
                            <div style="font-size: 42px; margin-right: 15px; text-color: #01364d">{data['icon']}</div>
                            <div>
                                <div style="font-size: 26px; font-weight: bold;">{data['count']}</div>
                                <div style="font-size: 16px; font-weight: bold;">{data['type']}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                fig3 = plot_bottom_left2(vehicle_count_each)
                st.plotly_chart(fig3, use_container_width=True, key="vehicle_counts_each")
            with col2db4:
                log_df = pd.DataFrame(congestion_log)
                st.subheader("Congestion Log")
                st.dataframe(log_df, use_container_width=True)          
            
            plot_bottom_left(vehicle_inside_area, placeholder)

    else:
        st.subheader("Traffic Overview")
        with st.container(border=True, key='4'):
            col1db1, col2db1, col3db1, col4db1 = st.columns(4)
            with col1db1:
                plot_gauge(
                    indicator_number=0,
                    indicator_color="blue",
                    indicator_suffix="",
                    indicator_title="Total Vehicles",
                    max_bound=100,  
                )
            with col2db1:
                plot_gauge(
                    indicator_number=0,
                    indicator_color="red",
                    indicator_suffix="%",
                    indicator_title="Congestion Rate",
                    max_bound=100,  
                )
            with col3db1:
                plot_gauge(
                    indicator_number=0,
                    indicator_color="green",
                    indicator_suffix="km/h",
                    indicator_title="Speed",
                    max_bound=100,  
                )
            with col4db1:
                plot_gauge(
                    indicator_number=0,
                    indicator_color="green",
                    indicator_suffix="%",
                    indicator_title="Occupancy",
                    max_bound=100,  
                )
        with st.container(border=True, key='2'):
            col1db2, col2db2 = st.columns(2, gap='large')
            with col1db2:
                st.subheader("Video")
                
                if uploaded_video is not None:
                    tfile.write(uploaded_video.read())
                    dem_vid = open(tfile.name, 'rb')
                    demo_bytes = dem_vid.read()
                    st.video(demo_bytes)
                else:
                    st.info("No video uploaded yet. Upload a video to preview it.")
                # display_video_from_mongodb(video_id) 
            with col2db2:
                placeholder = st.empty()
            
            col1db3, col2db3 = st.columns(2, gap='large')
            with col1db3:
                st.subheader("Camera Coordinate")
                st.info("Process video to show map")   
            with col2db3:
                fig2 = plot_bottom_right([0], [0], [0])
                st.plotly_chart(fig2, use_container_width=True, key="data")

        with st.container(border=True, key='3'):    
            col1db4, col2db4 = st.columns([6,4], gap='large')
            with col1db4:

                fleet_data = [
                    {"icon": "🚗", "count": 0, "type": "Car"},
                    {"icon": "🏍️", "count": 0, "type": "Motorcycle"},
                    {"icon": "🚛", "count": 0, "type": "Truck"},
                    {"icon": "🚌", "count": 0, "type": "Bus"},
                ]
                # Tạo layout card với Streamlit columns
                cols = st.columns(len(fleet_data))

                for col, data in zip(cols, fleet_data):
                    with col:
                        # Tạo card
                        st.markdown(f"""
                        <div style="
                            color: #31333f; 
                            padding: 15px; 
                            border-radius: 8px; 
                            text-align: left; 
                            display: flex; 
                            align-items: center;
                            justify-content: center; 
                            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.3);">
                            <div style="font-size: 42px; margin-right: 15px; text-color: #01364d">{data['icon']}</div>
                            <div>
                                <div style="font-size: 26px; font-weight: bold;">{data['count']}</div>
                                <div style="font-size: 16px; font-weight: bold;">{data['type']}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                fig3 = plot_bottom_left2({'car':0})
                st.plotly_chart(fig3, use_container_width=True, key="vehicle_counts_each")
            with col2db4:
                log_df = pd.DataFrame([0])
                st.subheader("Congestion Log")
                st.dataframe(log_df, use_container_width=True)          
            
            plot_bottom_left([0], placeholder)
else:
    st.sidebar.button("Start tracking",use_container_width=True)
    st.subheader("Traffic Overview")
    with st.container(border=True, key='4'):
        col1db1, col2db1, col3db1, col4db1 = st.columns(4)
        with col1db1:
            plot_gauge(
                indicator_number=0,
                indicator_color="blue",
                indicator_suffix="",
                indicator_title="Total Vehicles",
                max_bound=100,  
            )
        with col2db1:
            plot_gauge(
                indicator_number=0,
                indicator_color="red",
                indicator_suffix="%",
                indicator_title="Congestion Rate",
                max_bound=100,  
            )
        with col3db1:
            plot_gauge(
                indicator_number=0,
                indicator_color="green",
                indicator_suffix="km/h",
                indicator_title="Speed",
                max_bound=150,  
            )
        with col4db1:
            plot_gauge(
                indicator_number=0,
                indicator_color="green",
                indicator_suffix="%",
                indicator_title="Occupancy",
                max_bound=100,  
            )
    with st.container(border=True, key='2'):
        col1db2, col2db2 = st.columns(2, gap='large')
        with col1db2:
            st.subheader("Video")
            st.info("No video uploaded yet. Upload a video to preview it.")
            st.image('img/video_placeholder.png', width=420)
        with col2db2:
            placeholder = st.empty()
        
        col1db3, col2db3 = st.columns(2, gap='large')
        with col1db3:
            st.subheader("Camera Map")
            st.info("Process video to show map")   
        with col2db3:
            fig2 = plot_bottom_right([0], [0], [0])
            st.plotly_chart(fig2, use_container_width=True, key="data")

    with st.container(border=True, key='3'):    
        col1db4, col2db4 = st.columns([6,4], gap='large')
        with col1db4:

            fleet_data = [
                {"icon": "🚗", "count": 0, "type": "Car"},
                {"icon": "🏍️", "count": 0, "type": "Motorcycle"},
                {"icon": "🚛", "count": 0, "type": "Truck"},
                {"icon": "🚌", "count": 0, "type": "Bus"},
            ]
            # Tạo layout card với Streamlit columns
            cols = st.columns(len(fleet_data))

            for col, data in zip(cols, fleet_data):
                with col:
                    # Tạo card
                    st.markdown(f"""
                    <div style="
                        color: #31333f; 
                        padding: 15px; 
                        border-radius: 8px; 
                        text-align: left; 
                        display: flex; 
                        align-items: center;
                        justify-content: center; 
                        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.3);">
                        <div style="font-size: 42px; margin-right: 15px; text-color: #01364d">{data['icon']}</div>
                        <div>
                            <div style="font-size: 26px; font-weight: bold;">{data['count']}</div>
                            <div style="font-size: 16px; font-weight: bold;">{data['type']}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            fig3 = plot_bottom_left2({'car':0})
            st.plotly_chart(fig3, use_container_width=True, key="vehicle_counts_each")
        with col2db4:
            log_df = pd.DataFrame([0])
            st.subheader("Congestion Log")
            st.dataframe(log_df, use_container_width=True)          
        
        plot_bottom_left([0], placeholder)

