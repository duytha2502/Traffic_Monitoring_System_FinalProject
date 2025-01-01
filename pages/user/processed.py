from PIL import Image
from datetime import datetime
from streamlit_folium import st_folium
from connectDB import *
import folium
import datetime
import streamlit as st
import time
import numpy as np
import pandas as pd
import pymongo
import gridfs
import io
import base64
import zipfile
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Kết nối tới database "processed_videos"
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

    .e1nzilvr5 {
        overflow: auto;
        max-height: 550px;
    }

    .dataframe {
        width: 100%;
    }
    .dataframe tr, th {
        text-align: center;
    }

    .ea3mdgi5 {
        width: 1300px;
        margin-top: -50px
    }

    </style>

    """,
    unsafe_allow_html=True
)


def log_action(action, user_email, details=None):

    log_entry = {
        "action": action,
        "user_email": user_email,
        "details": details or {},
        "timestamp": datetime.datetime.now()
    }
    log.insert_one(log_entry)

# Hàm lấy danh sách tên video từ MongoDB
def get_videos():

    user_email = st.session_state["email"]
    
    # Lấy danh sách video từ collection `data` của người dùng hiện tại
    video_datas = data.find({"user_gmail": user_email})

    video_list = []
    for entry in video_datas:
        video_name = entry.get("video_name")
        
        # Kiểm tra xem `filename` có tồn tại trong `GridFS`
        matching_file = fs.find_one({"filename": video_name})
        if matching_file:
            video_list.append((matching_file._id, matching_file.filename, matching_file.uploadDate))

    return video_list

def get_videos_by_date(videos, date):
    video_by_date_list = []
    # Duyệt qua danh sách videos và so sánh ngày
    for _id, filename, uploadDate in videos:
        # Chuyển đổi chuỗi sang datetime object
        upload_date_only = uploadDate.date()  # Trích xuất phần ngày
        if upload_date_only == date:
            # Thêm filename và uploadDate vào danh sách
            video_by_date_list.append({filename: _id})

    return video_by_date_list

def get_video_comparison(video_options, selected_video):
    # Lọc danh sách video để loại trừ video đang được chọn
    comparison_options = [video for video in video_options.keys() if video != selected_video]
    
    # Tạo selectbox cho video cần so sánh
    if comparison_options:
        comparison_video = st.sidebar.selectbox(
            "Select a video to compare",
            comparison_options,
            key="comparison_selectbox",
            index=None
        )
        return comparison_video
    else:
        st.sidebar.warning("No other videos available to compare!")
        return None

# Hàm lấy danh sách video và thông tin từ data collection
def get_data(video_name):
    # Tìm kiếm trong GridFS để xem có file nào có filename khớp với video_name không
    matching_file = fs.find_one({"filename": video_name})
    
    if matching_file:
  
        # Lấy dữ liệu liên quan từ collection `data`
        related_data = data.find_one({"video_name": video_name})

        return related_data
    else:
        # Nếu không tìm thấy file khớp với video_name, trả về None hoặc thông báo không tìm thấy
        return None

def get_vehicle(video_name):
    # Tìm kiếm trong GridFS để xem có file nào có filename khớp với video_name không
    matching_file = fs.find_one({"filename": video_name})

    matching_files_in_fsvo = []
    
    if matching_file:
        
        # Tìm các tệp từ fsvo có metadata "video_name" trùng với video_name
        for file in fsvo.find():

            if video_name in file.metadata.values():
                # Lưu dữ liệu khớp vào danh sách hoặc thực hiện hiển thị
                matching_files_in_fsvo.append(file)

    else:
        # Nếu không tìm thấy file khớp với video_name, trả về thông báo không tìm thấy
        st.write(f"No matching file found in GridFS with filename: {video_name}")

    return matching_files_in_fsvo

# Hàm lấy video từ MongoDB và hiển thị trên Streamlit
def display_video_from_mongodb(video_id):
    video_data = fs.get(video_id)
    video_bytes = video_data.read()
    st.video(video_bytes, autoplay=True)

def image_to_base64(image):
    """Chuyển hình ảnh PIL sang base64."""
    buffered = io.BytesIO()
    new_img = image.resize((500, 500))  # Bạn có thể thay đổi kích thước này theo ý muốn
    new_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f'<img src="data:image/png;base64,{img_str}" style="width:100px;height:auto;">'

def display_image_from_mongodb(vehicle_get):

    vehicle_images = []
    for file in vehicle_get:
        if file.filename.endswith(".png"):
            # Đọc dữ liệu hình ảnh từ GridFS
            image_data = file.read()

            # Chuyển đổi dữ liệu thành hình ảnh bằng PIL
            image = Image.open(io.BytesIO(image_data))

            # Chuyển hình ảnh sang base64
            img_base64 = image_to_base64(image)
            
            vehicle_images.append({"Vehicle ID": file.metadata.get('vehicle_id'), "Image Name": file.filename, "Image Data": image, "Vehicle Image": img_base64, "Speed (km/h)": file.metadata.get('speed')})

    return vehicle_images

def dowload_video_from_mongodb(video_id):
    video_file = fs.get(video_id)
    return video_file.read()

# Hàm xóa dữ liệu từ MongoDB
def delete_data_from_mongodb(video, image):

    video_delete = fs.find({"filename": video})
    # Xóa video
    for video_item in video_delete:
        fs.delete(video_item._id)

    for image_item in image:
        fsvo.delete(image_item._id)

    data_delete = data.find_one({"video_name": video})
    data.delete_many(data_delete)

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
            height=400
        )

        placeholder.plotly_chart(fig, use_container_width=True, key=f'{i+10}')

        time.sleep(0.333333)

# Pie chart
def plot_vehicle_pie_chart(vehicle_counts_each):

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

# Hiển thị danh sách video và cho phép chọn video
st.sidebar.header("Your Video List")
videos = get_videos()

# Filter by date
st.sidebar.subheader("Filter by Date")
date = st.sidebar.date_input(
    "Select your date for fillter",
    value=None,
    max_value= datetime.datetime.now(),
    format="MM/DD/YYYY",
)

# Query videos for the selected date
videos_on_date = get_videos_by_date(videos, date)

if videos_on_date:
    # Lấy danh sách video từ videos_on_date
    video_options = {filename: _id for video in videos_on_date for filename, _id in video.items()}
    selected_video = st.sidebar.selectbox("Select a video", list(video_options.keys()))
else:
    # Không có video nào trùng ngày
    if date:
        st.sidebar.warning(f"No videos available in {date}.")
        selected_video = []
    else:
        # Hiển thị tất cả video nếu không có bộ lọc
        video_options = {filename: _id for _id, filename, uploadDate in videos}
        selected_video = st.sidebar.selectbox("Select a video", list(video_options.keys()))

st.header(":desktop_computer: Processed Video")
st.markdown("---")
if selected_video:

    # Selected video data
    vehicle_get = get_vehicle(selected_video)
    data_get = get_data(selected_video)
    timestamp = data_get.get('timestamp')
    average_speed = data_get.get('average_speed')
    occupancy = data_get.get('occupancy')
    congestion_rate = data_get.get('congestion_rate')
    vehicle_count = data_get.get('vehicle_count')
    vehicle_count_each = data_get.get('vehicle_count_each')
    vehicle_inside_area = data_get.get('vehicle_inside_area')

    comparison_video = get_video_comparison(video_options, selected_video)
    if comparison_video:

        st.subheader("Displaying Comparison Video")
        
        # Selected comparison video data
        data_compare_get = get_data(comparison_video)
        timestamp_compare = data_compare_get.get('timestamp')
        average_speed_compare = data_compare_get.get('average_speed')
        occupancy_compare = data_compare_get.get('occupancy')
        congestion_rate_compare = data_compare_get.get('congestion_rate')
        vehicle_count_compare = data_compare_get.get('vehicle_count')
        vehicle_count_each_compare = data_compare_get.get('vehicle_count_each')


        col1, col2 = st.columns(2)
        with col1:
            st.write(f"{selected_video}")
            video_id = video_options[selected_video]
            display_video_from_mongodb(video_id)

        with col2:
            st.write(f"{comparison_video}")
            video_compare_id = video_options[comparison_video]
            display_video_from_mongodb(video_compare_id)

        metrics = {
            "Total Vehicles": {
                "selected": vehicle_count[-1],
                "comparison": vehicle_count_compare[-1],
            },
            "Average Speed (km/h)": {
                "selected": round(sum(average_speed)/len(average_speed), 2),
                "comparison": round(sum(average_speed_compare)/len(average_speed_compare), 2),
            },
            "Occupancy (%)": {
                "selected": round(sum(occupancy)/len(occupancy), 2),
                "comparison": round(sum(occupancy_compare)/len(occupancy_compare), 2),
            },
            "Congestion Rate (%)": {
                "selected": round(sum(congestion_rate)/len(congestion_rate), 2),
                "comparison": round(sum(congestion_rate_compare)/len(congestion_rate_compare), 2)
            },
        }

        # Hiển thị các số liệu bằng st.metric
        for metric_name, values in metrics.items(): 
            delta = ((values["comparison"] - values["selected"]) / values["selected"]) * 100
            with col1:
                st.metric(label=f"{metric_name}", value=values["selected"])
            with col2:
                st.metric(label=f"{metric_name}", value=values["comparison"], delta=round(delta, 2))
       
        st.subheader("Metric Per time")
        with st.container():
            col1cl, col2cl = st.columns(2)
            with col1cl:
                chart_metric = pd.DataFrame({
                    'Timestamp (s)': timestamp,
                    'Average Speed (km/h)': average_speed,
                    'Occupancy (%)': occupancy,
                    'Congestion Rate (%)': congestion_rate,
                })
                st.line_chart(chart_metric, x="Timestamp (s)")
            with col2cl:
                chart_metric_comparte = pd.DataFrame({
                    'Timestamp (s)': timestamp_compare,
                    'Average Speed Compare (km/h)': average_speed_compare,
                    'Occupancy Compare (%)': occupancy_compare,
                    'Congestion Rate Compare (%)': congestion_rate_compare,
                })
                st.line_chart(chart_metric_comparte, x="Timestamp (s)")

        metrics_chart = {
            "Avg speed (km/h)": {
                "selected": round(sum(average_speed)/len(average_speed), 2),
                "comparison": round(sum(average_speed_compare)/len(average_speed_compare), 2),
            },
            "Occupancy (%)": {
                "selected": round(sum(occupancy)/len(occupancy), 2),
                "comparison": round(sum(occupancy_compare)/len(occupancy_compare), 2),
            },
            "Congestion (%)": {
                "selected": round(sum(congestion_rate)/len(congestion_rate), 2),
                "comparison": round(sum(congestion_rate_compare)/len(congestion_rate_compare), 2)
            },
        }

        st.subheader("Comparison Chart")
        # Hiển thị phần chọn multiselect
        selected_metrics = st.multiselect(
            "Choose a metric you want to display",
            options=["Selected Video", "Comparison Video"],
            default=["Selected Video", "Comparison Video"]
        )

        col1cc, col2cc = st.columns(2)
        with col1cc:
            chart_compare = pd.DataFrame({
                    "Metric": list(metrics_chart.keys()),
                    "Selected Video": [v["selected"] for v in metrics_chart.values()],
                    "Comparison Video": [v["comparison"] for v in metrics_chart.values()],
                })

            # Hiển thị biểu đồ đường với các số liệu được chọn
            if selected_metrics:
                st.bar_chart(chart_compare, x="Metric", y=selected_metrics, height=500, horizontal=True)
            else:
                st.warning("Choose at least one metric to display")

        with col2cc:
            vehicle_type_metrics_chart = {
                vehicle: {
                    "selected": vehicle_count_each[vehicle],
                    "comparison": vehicle_count_each_compare[vehicle],
                }
                for vehicle in vehicle_count_each.keys()
            }

            chart_compare_type_vehicle = pd.DataFrame({
                    "Metric": list(vehicle_type_metrics_chart.keys()),
                    "Selected Video": [v["selected"] for v in vehicle_type_metrics_chart.values()],
                    "Comparison Video": [v["comparison"] for v in vehicle_type_metrics_chart.values()],
                })
            # Hiển thị biểu đồ đường với các số liệu được chọn
            if selected_metrics:
                st.bar_chart(chart_compare_type_vehicle, x="Metric", y=selected_metrics, height=500)
            else:
                st.warning("Choose at least one metric to display")

    else:
        
        df = pd.DataFrame({
            'Timestamp': timestamp,
            'Average Speed (km/h)': [round(speed, 2) for speed in average_speed],
            'Occupancy (%)': [round(occur, 2) for occur in occupancy],
            'Congestion Rate (%)': [round(rate, 2) for rate in congestion_rate],
            'Vehicle Count': [round(count, 0) for count in vehicle_count],
        })
        csv = df.to_csv(index=False).encode('utf-8')

        col1, col2 = st.columns([0.8,0.2])
        with col1:
            st.subheader(f"Displaying: {selected_video}")
        with col2:
            with st.popover("Video Options"):
                col1bt, col2bt = st.columns(2)
                with col1bt:
                    video_id = video_options[selected_video]
                    video_download = dowload_video_from_mongodb(video_id)
                    if st.download_button(
                        label="Download",
                        data=video_download,
                        file_name=f"{selected_video}.mp4",
                        mime="video/mp4",
                        icon="⬇️"):
                        log_action("Download", st.session_state["email"], f"User {st.session_state["email"]} dowloaded video {video_options[selected_video]}")
                        st.rerun()
                with col2bt:
                    if st.button("Delete", type="primary", use_container_width=True, icon="⚠️"):
                        delete_data_from_mongodb(selected_video, vehicle_get)
                        log_action("Delete", st.session_state["email"], f"User {st.session_state["email"]} deleted video {video_options[selected_video]}")
                        st.rerun()
                        
        video_id = video_options[selected_video]
        col1vd, col2vd = st.columns(2)
        with col1vd:
            display_video_from_mongodb(video_id)
        with col2vd:
            placeholder = st.empty()

        # Hiển thị kết quả          
        col1st, col2st, col3st, col4st = st.columns(4)
        with col1st:
            st.write("<h5 class='metric_label'>Total Vehicles</h5>", unsafe_allow_html=True)
            st.write(f"<h3 class='metric_number'>{vehicle_count[-1]}</h3>", unsafe_allow_html=True)
        with col2st:
            st.write("<h5 class='metric_label'>Average Speed</h5>", unsafe_allow_html=True)
            st.write(f"<h3 class='metric_number'>{sum(average_speed)/len(average_speed):.0f} km/h</h3>", unsafe_allow_html=True)
        with col3st:
            st.write("<h5 class='metric_label'>Occupancy</h5>", unsafe_allow_html=True)
            st.write(f"<h3 class='metric_number'>{sum(occupancy)/len(occupancy):.2f} %</h3>", unsafe_allow_html=True)
        with col4st:
            st.write("<h5 class='metric_label'>Congestion</h5>", unsafe_allow_html=True)
            st.write(f"<h3 class='metric_number'>{sum(congestion_rate)/len(congestion_rate):.2f} %</h3>", unsafe_allow_html=True)

        st.subheader("Statistic")
        st.dataframe(df, hide_index=True, use_container_width=True)

        st.subheader("Line Chart")
        # Hiển thị phần chọn multiselect
        selected_metrics = st.multiselect(
            "Choose a metric you want to display",
            options=["Average Speed (km/h)", "Occupancy (%)", "Congestion Rate (%)"],
            default=["Average Speed (km/h)", "Occupancy (%)", "Congestion Rate (%)"]
        )
        
        chart_spd_ocu_cgs = pd.DataFrame({
            'Timestamp (s)': timestamp,
            'Average Speed (km/h)': average_speed,
            'Occupancy (%)': occupancy,
            'Congestion Rate (%)': congestion_rate,
        })

        # Hiển thị biểu đồ đường với các số liệu được chọn
        if selected_metrics:
            st.line_chart(chart_spd_ocu_cgs, x="Timestamp (s)", y=selected_metrics)
        else:
            st.warning("Choose at least one metric to display")
        
        col1bc, col2bc = st.columns(2)
        with col1bc:
            st.subheader("Vehicle Type Quantity")
            chart_vehicle = pd.DataFrame({
                'Vehicle Type': list(vehicle_count_each.keys()),
                'Count': list(vehicle_count_each.values())
            })
            st.bar_chart(chart_vehicle, x='Vehicle Type', y='Count', height=500)

        with col2bc:
            fig = plot_vehicle_pie_chart(vehicle_count_each)
            st.plotly_chart(fig, use_container_width=True, key="vehicle_counts_each")
        
        st.subheader("Vehicle Overspeed Capture")
        vehicle_images = display_image_from_mongodb(vehicle_get)

        # Tạo DataFrame
        if not vehicle_images:
            st.info("No vehicle overspeed images found for this video!")
        else:
            df = pd.DataFrame(vehicle_images)   
            df_display = df.drop(columns=["Image Data", "Image Name"])
            st.write(df_display.to_html(escape=False), unsafe_allow_html=True)
            
            # Tạo file ZIP chứa cả CSV và ảnh
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                # Lưu CSV vào file ZIP
                csv_buffer = io.StringIO()
                df.drop(columns=["Vehicle Image"]).to_csv(csv_buffer, index=False)
                zip_file.writestr("vehicle_overspeed.csv", csv_buffer.getvalue())

                # Lưu từng hình ảnh vào file ZIP
                for _, row in df.iterrows():
                    image_name = row["Image Name"]
                    image_data = row["Image Data"]
                    img_byte_arr = io.BytesIO()
                    image_data.save(img_byte_arr, format='PNG')
                    zip_file.writestr(f"images/{image_name}", img_byte_arr.getvalue())
            st.markdown("---")
            # Nút tải file ZIP
            st.download_button(
                label="Download ZIP folder",
                data=zip_buffer.getvalue(),
                file_name="vehicle_overspeed.zip",
                mime="application/zip"
            )

        plot_bottom_left(vehicle_inside_area, placeholder)
