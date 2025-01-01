import streamlit as st
import os
import time
from connectDB import *
from uuid import uuid4
from PIL import Image, ImageDraw
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

# Kết nối tới database "processed_videos"
db, fs = connect_to_mongo("processed_videos")
user = db['users']

# Hàm lấy danh sách tên video từ MongoDB
def get_user():
    user_email = st.session_state["email"]
    curr_user = user.find_one({"email": user_email})
    return curr_user

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
    
     # Xử lý hình ảnh thành hình tròn
    img = Image.open(file_path).convert("RGBA")  # Đảm bảo ảnh có kênh alpha
    size = min(img.size)  # Lấy kích thước nhỏ nhất để tạo hình tròn
    mask = Image.new("L", (size, size), 0)  # Mặt nạ (mask) hình tròn
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, size, size), fill=255)  # Vẽ hình tròn trên mặt nạ
    
    # Cắt ảnh thành hình vuông và áp dụng mặt nạ
    img_cropped = img.crop((0, 0, size, size))
    img_cropped.putalpha(mask)  # Áp dụng mặt nạ
    
    # Lưu ảnh tròn
    round_file_name = f"round_{uuid4()}.{file_extension}"
    round_file_path = os.path.join(folder, round_file_name)
    img_cropped.save(round_file_path, format="PNG")  # Lưu ảnh dưới dạng PNG để giữ kênh alpha
    
    # Xóa ảnh gốc nếu không cần
    os.remove(file_path)
    
    return round_file_path
    
def update_avatar(user_email, uploaded_file):

    file_path = save_uploaded_file(uploaded_file)
    
    # Lưu đường dẫn vào MongoDB
    user.update_one(
        {"email": user_email},
        {"$set": {"avatar": file_path, "updated_time": datetime.now()}}
    )

def change_password(user_gmail, old_password, new_password):

    curr_user = get_user()

    if not curr_user:
        return {"success": False, "message": "User not found."}

    if not check_password_hash(curr_user["password"], old_password):
        return {"success": False, "message": "Old password is incorrect."}

    # Hash mật khẩu mới
    new_password_hash = generate_password_hash(new_password)

    # Cập nhật mật khẩu trong cơ sở dữ liệu
    update_result = user.update_one(
        {"email": user_gmail},
        {
            "$set": {
                "password": new_password_hash,
                "updated_time": datetime.now()  # Cập nhật thời gian thay đổi
            }
        }
    )

@st.dialog("Change Your Password")
def change_password_dialog(user_email):

    st.write("Insert your new password")
    old_password = st.text_input("Old Password", type='password')
    new_password = st.text_input("New Password", type='password')
    conf_new_password = st.text_input("Confirm New Password", type='password')
    if st.button("Submit"):
        if new_password != conf_new_password:
            st.error("New password and confirmation do not match!")
        else:
            change_password(user_email, old_password, new_password)
        st.rerun()

st.title(":newspaper: Profile Users")
st.markdown("---")
user_profile = get_user()

# Tạo container để hiển thị giao diện
with st.container():
    # Avatar và thông tin trên đầu
    col1, col2, col3 = st.columns([2, 5, 3])
    with col1:
        st.image(user_profile["avatar"])
    with col2:
        st.subheader(user_profile["first_name"] + " " + user_profile["last_name"])
        st.text(user_profile["email"])
    with col3:
        if st.button("Change Password"):    
            change_password_dialog(st.session_state['email'])

    # Form cập nhật thông tin
    with st.form("profile_form"):
        email = st.text_input("Email", value=user_profile["email"])
        col1n, col2n = st.columns(2)
        with col1n:
            first_name = st.text_input("First Name", value=user_profile["first_name"], key='1')
        with col2n:
            last_name = st.text_input("Last Name", value=user_profile["last_name"])
        col1st, col2st = st.columns([1,9.5])
        with col1st:
            st.markdown('Status')
        with col2st:
            if user_profile["status"] == "active":
                st.write(":green[**Active**]")
            else:   
                st.write("Inactive")
        uploaded_file = st.file_uploader("Choose an image to change avatar", type=["png", "jpg", "jpeg"]) 
        # Nút Save Change
        submitted = st.form_submit_button("Save Change", use_container_width=True)
        if submitted:
            if uploaded_file is not None:
                update_avatar(st.session_state['email'], uploaded_file)
            new_data = {
                "email": email,
                "first_name": first_name,
                "last_name": last_name,
                "updated_time": datetime.now()
            }
            user.update_many({"email": email}, {"$set": new_data})
            st.toast("Your profile has been updated!")
            time.sleep(1)
            st.rerun()

