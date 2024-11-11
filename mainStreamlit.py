import streamlit as st
import pymongo
from werkzeug.security import generate_password_hash, check_password_hash

# Kết nối đến MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["processed_videos"]
users_collection = db["users"]

# Hàm tạo tài khoản (lưu mật khẩu đã mã hóa)
def create_user(email, password):
    password_hash = generate_password_hash(password)
    users_collection.insert_one({"email": email, "password": password_hash})

# Hàm xác thực người dùng
def authenticate_user(email, password):
    user = users_collection.find_one({"email": email})
    if user and check_password_hash(user["password"], password):
        return True
    return False

def register():
    st.header("Register")
    email = st.text_input("Email Address")
    new_username_firstname = st.text_input("First Name")
    new_username_lastname = st.text_input("Last Name")
    new_password = st.text_input("New Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")
    if st.button("Register", use_container_width=True):
        if email and new_password:
            create_user(email, new_password)
            st.success("Account created successfully. Please log in.")
            st.session_state["is_register"] = False  # Quay lại trang đăng nhập sau khi đăng ký
            st.rerun()
        else:
            st.error("Please enter a username and password")
    
      # Nút chuyển sang đăng nhập
    if st.button("Switch to Login", use_container_width=True):
        st.session_state["is_register"] = False
        st.rerun()

def login():
    col1 , col2 = st.columns(2)
    with col1:
        st.image('img/banner.png', width=250)
    with col2:
        st.header("Login")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Log in", use_container_width=True):
            if authenticate_user(email, password):
                st.session_state.logged_in = True
                st.session_state["show_success_message"] = True
                st.session_state["email"] = email
                st.rerun()
            else:
                st.error("Invalid email or password")
    
        # Nút chuyển sang đăng ký
        if st.button("Switch to Register", use_container_width=True):
            st.session_state["is_register"] = True
            st.rerun()

def logout():
    st.subheader("Are you sure you want to log out?")
    if st.button("Log out"):
        st.session_state.logged_in = False
        del st.session_state.email
        st.rerun()

# Kiểm tra trạng thái và hiển thị giao diện phù hợp
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "show_success_message" in st.session_state:
    st.toast(f"Welcome {st.session_state['email']} !")
    del st.session_state["show_success_message"]

if "is_register" not in st.session_state:
    st.session_state["is_register"] = False

if st.session_state["is_register"]:
    register_page = st.Page(register, title="Register", icon=":material/camera:")
else:
    login_page = st.Page(login, title="Login", icon=":material/camera:")

logout_page = st.Page(logout, title="Log out", icon=":material/logout:")

about = st.Page(
    "pages/about.py", title="How to use", icon=":material/help:", default=True)
monitor = st.Page(
    "pages/monitor.py", title="Monitoring", icon=":material/psychology:")
processed = st.Page(
    "pages/processed.py", title="Processed Video", icon=":material/database:")

if st.session_state.logged_in:
    pg = st.navigation(
        {
            "Account": [logout_page],
            "Start Tracking": [about, monitor, processed],
        }
    )
else:
    if st.session_state.is_register:
        pg = st.navigation([register_page])
    else:
        pg = st.navigation([login_page])

pg.run()

