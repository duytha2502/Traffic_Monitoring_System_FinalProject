import streamlit as st
import pymongo
import os
import pandas as pd
from datetime import datetime
from connectDB import *
from uuid import uuid4

# Kết nối đến MongoDB
db, fs = connect_to_mongo("processed_videos")
data = db["data"]
user_collection = db["users"]
log = db["logs"]  

def log_action(action, user_email, details=None):

    log_entry = {
        "action": action,
        "user_email": user_email,
        "details": details or {},
        "timestamp": datetime.now()
    }
    log.insert_one(log_entry)

def fetch_users():
    # Lấy dữ liệu từ MongoDB
    users = list(user_collection.find({}, {"_id": 0}))  # Không hiển thị `_id`

    return users

def fetch_users_edit():
    # Lấy dữ liệu từ MongoDB
    users = list(user_collection.find({"role": {"$ne": 0}}, {"_id": 0}))
    
    return users

def get_user(user_email):

    matching_file = user_collection.find_one({"email": user_email})

    if matching_file:

        return matching_file
    else:
        return None


def save_uploaded_file(uploaded_file, folder="img"):
    """Lưu file được upload và trả về đường dẫn."""
    # Tạo thư mục nếu chưa tồn tại
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Tạo tên file duy nhất
    file_extension = uploaded_file.name.split(".")[-1]
    file_name = f"{uuid4()}.{file_extension}"
    file_path = os.path.join(folder, file_name)
    
    # Ghi nội dung file vào server
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path

def update_avatar(user_email, uploaded_file):

    file_path = save_uploaded_file(uploaded_file)
    
    # Lưu đường dẫn vào MongoDB
    user_collection.update_one(
        {"email": user_email},
        {"$set": {"avatar": file_path, "updated_time": datetime.now()}}
    )

def display_and_edit_users():
    users = fetch_users()
    users_edit = fetch_users_edit()
    if not users:
        st.write("No users found.")
        return

    # Chuyển dữ liệu thành DataFrame
    df = pd.DataFrame(users)
    dfedit = pd.DataFrame(users_edit)
    # Hiển thị DataFrame có thể chỉnh sửa
    edited_df = st.data_editor(
        df, 
        column_config={
        "password": st.column_config.Column(
            width="small",
            required=True,
        )
        },
        num_rows="dynamic", 
        use_container_width=True, 
        disabled=True)

    col1, col2 = st.columns([3,7], gap="large")
    with col1:
        user = st.selectbox("Choose user to edit", dfedit['email'], index=None, placeholder="Find user...")
    with col2:
        if user:
            user_data = get_user(f"{user}")
            col1av, col2av = st.columns([6,4])
            with col1av:
                avatar = st.image(user_data["avatar"])
                uploaded_file = st.file_uploader("Choose an image to change avatar", type=["png", "jpg", "jpeg"], label_visibility="collapsed" )
                   
            with col2av:
                email = st.text_input("Email", user_data["email"])
                fn = st.text_input("First Name", user_data["first_name"])
                ln = st.text_input("Last Name", user_data["last_name"])

                if f"{user_data["status"]}" == "active":
                    on = st.toggle(f"{user_data["status"]}", value=True)
                elif f"{user_data["status"]}" == "inactive":
                    on = st.toggle(f"{user_data["status"]}")

                if on:
                    st.session_state["status"] = "active"
                else:
                    st.session_state["status"] = "inactive"

            col1btn, col2btn = st.columns([0.5,0.5], gap='large')
            with col1btn:
                if st.button("Save Changes"):
                    new_data = {
                        "status": st.session_state["status"],
                        "email": email,
                        "first_name": fn,
                        "last_name": ln,
                        "updated_time": datetime.now()
                    }
                    update_avatar(user_data["email"], uploaded_file)
                    user_collection.update_many({"email": email}, {"$set": new_data})
                    st.toast(f"User {email} updated successfully.")
                    log_action("Update", st.session_state["email"], f"Update information for user {email}")

            with col2btn:
                with st.popover("Delete User", icon="⚠️", use_container_width=True):
                    st.write("Are you sure want to delete this user?")
                    if st.button("Delete User", type="primary", use_container_width=True):
                        user_collection.delete_many({"email": email})
                        st.toast(f"User {email} deleted successfully.")
                        log_action("Delete", st.session_state["email"], f"Delete user {email}")

st.title("This is manage user page")
display_and_edit_users()




