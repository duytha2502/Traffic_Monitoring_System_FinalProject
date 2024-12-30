from datetime import datetime
from connectDB import *
import streamlit as st
import pymongo
import pandas as pd
import gridfs
import plotly.express as px
import plotly.graph_objects as go
import random

# Kết nối đến MongoDB
db, fs = connect_to_mongo("processed_videos")
data = db["data"]
log = db["logs"]  

# Đếm số lượng user và video
user_count = db.users.count_documents({})
video_count = db.fs.files.count_documents({}) 

def plot_metric(label, value, prefix="", suffix="", show_graph=False, color_graph=""):
    fig = go.Figure()

    fig.add_trace(
        go.Indicator(
            value=value,
            gauge={"axis": {"visible": False}},
            number={
                "prefix": prefix,
                "suffix": suffix,
                "font.size": 24,
            },
            title={
                "text": label,
                "font": {"size": 24},
            },
        )
    )

    if show_graph:
        fig.add_trace(
            go.Scatter(
                y=random.sample(range(0, 101), 30),
                hoverinfo="skip",
                fill="tozeroy",
                fillcolor=color_graph,
                line={
                    "color": color_graph,
                },
            )
        )

    fig.update_xaxes(visible=False, fixedrange=True)
    fig.update_yaxes(visible=False, fixedrange=True)
    fig.update_layout(
        # paper_bgcolor="rgba(255, 255, 255, 0.5)",  # Nền mờ với màu trắng
        margin=dict(t=20, b=0),
        showlegend=False,
        height=100,
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_gauge(
    indicator_number, indicator_color, indicator_suffix, indicator_title, max_bound):
    fig = go.Figure(
        go.Indicator(
            value=indicator_number,
            mode="gauge+number",
            domain={"x": [0, 1], "y": [0, 1]},
            number={
                "suffix": indicator_suffix,
                "font.size": 26,
            },
            gauge={
                "axis": {"range": [0, max_bound], "tickwidth": 1},
                "bar": {"color": indicator_color},
            },
            title={
                "text": indicator_title,
                "font": {"size": 28},
            },
        )
    )
    fig.update_layout(
        height=200,
        margin=dict(l=10, r=10, t=70, b=10, pad=8),
    )
    st.plotly_chart(fig, use_container_width=True)

def display_logs():
    
    logs = list(log.find().sort("timestamp", -1))  # Lấy log mới nhất trước
    logs_df = pd.DataFrame(logs)
    logs_df["timestamp"] = logs_df["timestamp"].apply(lambda x: x.strftime("%H:%M:%S %d-%m-%Y"))
    
    col1, col2 = st.columns([8, 2])
    with col1:
        st.write("### Action Logs")
    with col2:
        st.download_button(
            "Download CSV",
            data=logs_df.to_csv(index=False),
            file_name="logs.csv",
            mime="text/csv"
        )
    st.dataframe(logs_df[["timestamp", "user_email", "action", "details"]])

st.title("This is checking log page")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Users")
    plot_metric(
        label="Total Users",
        value=user_count,
        suffix="users",
        show_graph=True,
        color_graph="#99ceff",
    )
    plot_gauge(
        indicator_number=user_count,
        indicator_color="blue",
        indicator_suffix=" users",
        indicator_title="Users Registered",
        max_bound=20,  # Ví dụ đặt mức tối đa là 1000 user
    )

with col2:
    # Hiển thị số lượng Video
    st.subheader("Videos")
    plot_metric(
        label="Total Videos",
        value=video_count,
        suffix=" videos",
        show_graph=True,
        color_graph="#ffbb80",
    )
    # Hiển thị thêm Gauge
    plot_gauge(
        indicator_number=video_count,
        indicator_color="orange",
        indicator_suffix=" videos",
        indicator_title="Videos Stored",
        max_bound=20,  # Ví dụ đặt mức tối đa là 500 video
    )
st.write("---")
display_logs()
