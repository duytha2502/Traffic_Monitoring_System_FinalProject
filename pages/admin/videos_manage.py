import streamlit as st
import pymongo
import pandas as pd
import gridfs
import datetime
from connectDB import *

# Kết nối đến MongoDB
db, fs = connect_to_mongo("processed_videos")
data = db["data"]
log = db["logs"]  

def get_videos_df():
    videos = []
    for file in fs.find():
        videos.append({
            "File ID": str(file._id), 
            "Filename": file.filename, 
            "Uploaded By": file.metadata.get('uploaded_by', 'Unknown'), 
            "Upload Date": file.uploadDate
        })
    return videos

def get_videos():
    videos = [(file._id, file.filename) for file in fs.find()]

    return videos

# Hàm lấy video từ MongoDB và hiển thị trên Streamlit
def display_video_from_mongodb(video_id):
    video_data = fs.get(video_id)
    video_bytes = video_data.read()
    st.video(video_bytes)

st.title("This is video manage page")

# Hiển thị video Dataframe\
videos_df = get_videos_df()
df = pd.DataFrame(videos_df)
st.dataframe(df, use_container_width=True)

# Hiển thị danh sách video và cho phép chọn video
st.sidebar.header("Video List")
videos = get_videos()
video_options = {filename: _id for _id, filename in videos}

col1, col2, col3 = st.columns([2, 5, 1])
with col1:
    st.date_input(
        "Select your date for fillter",
        value=None,
        max_value= datetime.datetime.now(),
        format="MM/DD/YYYY",
    )
with col2:
    selected_video = st.selectbox("Select a video", list(video_options.keys()))
with col3:
    with st.popover("Delete"):
        st.write("Are you sure want to delete this video?")
        if st.button("Delete", type="primary", use_container_width=True, icon="⚠️"):
                            delete_data_from_mongodb(selected_video, vehicle_get)
                            log_action("Delete", st.session_state["email"], f"User {st.session_state["email"]} deleted video {video_options[selected_video]}")
                            st.rerun()
if selected_video:
    video_id = video_options[selected_video]
    display_video_from_mongodb(video_id)



