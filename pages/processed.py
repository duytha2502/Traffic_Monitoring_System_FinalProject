import streamlit as st
import time
import numpy as np
import pandas as pd
import pymongo
import gridfs
import io
import base64
import matplotlib.pyplot as plt
from main_old import plot_vehicle_pie_chart
from PIL import Image

# Kết nối tới MongoDB local
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["processed_videos"]
data = db["data"]
fs = gridfs.GridFS(db)

vo = client["vehicles_overspeed"]
fsvo = gridfs.GridFS(vo)

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
        max-height: 500px;
    }

    .dataframe {
        width: 100%;
    }
    .dataframe tr, th {
        text-align: center;
    }
    </style>

    """,
    unsafe_allow_html=True
)

# Biến trạng thái để lưu video cần xác nhận xóa
if "confirm_delete" not in st.session_state:
    st.session_state.confirm_delete = None

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
            video_list.append((matching_file._id, matching_file.filename))

    return video_list

# Hàm lấy danh sách video và thông tin từ data collection
def get_data(video_name):
    # Tìm kiếm trong GridFS để xem có file nào có filename khớp với video_name không
    matching_file = fs.find_one({"filename": video_name})
    
    if matching_file:
        # Nếu tìm thấy file khớp với video_name, lấy _id của nó
        video_id = matching_file._id

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
    st.video(video_bytes)

def image_to_base64(image):
    """Chuyển hình ảnh PIL sang base64."""
    buffered = io.BytesIO()
    new_img = image.resize((500, 500))  # Bạn có thể thay đổi kích thước này theo ý muốn
    new_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f'<img src="data:image/png;base64,{img_str}" style="width:150px;height:auto;">'

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
            
            vehicle_images.append({"Vehicle ID": file.metadata.get('vehicle_id'), "Image Name": file.filename, "Vehicle Image": img_base64, "Speed (km/h)": file.metadata.get('speed')})
    
    return vehicle_images
    
# Hàm xóa dữ liệu từ MongoDB
def delete_data_from_mongodb(query):
    video_delete = fs.find({"filename": query})
    data_delete = data.find_one({"video_name": query})
    result = data.delete_many(data_delete)
    # Xóa từng tệp
    for video in video_delete:
        fs.delete(video._id)
    st.session_state.confirm_delete = None
    st.rerun()

# Hiển thị danh sách video và cho phép chọn video
st.sidebar.header("Video List")
videos = get_videos()

video_options = {filename: _id for _id, filename in videos}
selected_video = st.sidebar.selectbox("Select a video", list(video_options.keys()))

if selected_video:
    vehicle_get = get_vehicle(selected_video)

    data_get = get_data(selected_video)
    timestamp = data_get.get('timestamp')
    average_speed = data_get.get('average_speed')
    occupancy = data_get.get('occupancy')
    congestion_rate = data_get.get('congestion_rate')
    vehicle_count = data_get.get('vehicle_count')
    vehicle_count_each = data_get.get('vehicle_count_each')
    vehicle_inside_area = data_get.get('vehicle_inside_area')

    df = pd.DataFrame({
        'Timestamp': timestamp,
        'Average Speed (km/h)': [round(speed, 2) for speed in average_speed],
        'Occupancy (%)': [round(occur, 2) for occur in occupancy],
        'Congestion Rate (%)': [round(rate, 2) for rate in congestion_rate],
        'Vehicle Count': [round(count, 0) for count in vehicle_count],
    })
    csv = df.to_csv(index=False).encode('utf-8')

    col1, col2 = st.columns([0.75,0.25])
    with col1:
        st.subheader(f"Displaying: {selected_video}")
    with col2:
        with st.expander("Delete Video", icon="⚠️"):
            st.session_state.confirm_delete = data_get.get('video_name')
        # if st.button("Delete Video", type="primary", icon="⚠️", use_container_width=True):
            if st.session_state.confirm_delete:
                    if st.button("Confirm", type="primary", use_container_width=True):
                        delete_data_from_mongodb(data_get.get('video_name'))
                        st.toast('Delete Successful', icon="✅")

    video_id = video_options[selected_video]
    display_video_from_mongodb(video_id)

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
    
    chart_vehicle_inside_area = pd.DataFrame({
        'Timestamp (s)': timestamp,
        'Vehicles In Area': vehicle_inside_area
    })
    st.line_chart(chart_vehicle_inside_area, x="Timestamp (s)", y="Vehicles In Area")

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
        st.subheader("Line Chart")
        st.line_chart(chart_spd_ocu_cgs, x="Timestamp (s)", y=selected_metrics)
    else:
        st.warning("Choose at least one metric to display")
    
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
    
    st.subheader("Vehicle Overspeed")
    vodf = pd.DataFrame({
        'Timestamp': timestamp,
        'Average Speed (km/h)': [round(speed, 2) for speed in average_speed],
        'Occupancy (%)': [round(occur, 2) for occur in occupancy],
        'Congestion Rate (%)': [round(rate, 2) for rate in congestion_rate],
        'Vehicle Count': [round(count, 0) for count in vehicle_count],
    })

    vehicle_images = display_image_from_mongodb(vehicle_get)

    # Tạo DataFrame
    df = pd.DataFrame(vehicle_images)
    st.write(df.to_html(escape=False), unsafe_allow_html=True)
