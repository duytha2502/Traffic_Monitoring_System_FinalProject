import pymongo
import gridfs
import streamlit as st

# Kết nối tới MongoDB local
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["processed_videos"]
fs = gridfs.GridFS(db)

# Hàm lưu video vào MongoDB
def save_video_to_mongodb(video_path, filename):
    with open(video_path, "rb") as video_file:
        video_id = fs.put(video_file, filename=filename)
    return video_id

# Hàm lấy danh sách tên video từ MongoDB
def get_all_videos():
    return [(file._id, file.filename) for file in fs.find()]

# Hàm lấy video từ MongoDB và hiển thị trên Streamlit
def display_video_from_mongodb(video_id):
    video_data = fs.get(video_id)
    video_bytes = video_data.read()
    st.video(video_bytes)

# Hiển thị danh sách video và cho phép chọn video
st.sidebar.header("Video List")
videos = get_all_videos()
video_options = {filename: _id for _id, filename in videos}
selected_video = st.sidebar.selectbox("Select a video", list(video_options.keys()))

if selected_video:
    st.subheader(f"Displaying: {selected_video}")
    video_id = video_options[selected_video]
    display_video_from_mongodb(video_id)
