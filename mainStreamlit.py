import streamlit as st
import pymongo
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

# Kết nối đến MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["processed_videos"]
users_collection = db["users"]

# Hàm tạo tài khoản (lưu mật khẩu đã mã hóa)
def create_user(email, new_username_firstname ,new_username_lastname, password):
    password_hash = generate_password_hash(password)
    users_collection.insert_one({"role": 1, "email": email, "first_name": new_username_firstname, "last_name": new_username_lastname ,"password": password_hash, "created_time": datetime.now(), "updated_time": datetime.now()})

# Hàm xác thực người dùng
def authenticate_user(email, password):
    user = users_collection.find_one({"email": email})
    if user and check_password_hash(user["password"], password):
        return user
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
            if new_password != confirm_password:
                st.toast("Passwords do not match!", icon="⚠️")
            else:
                existing_user = db.users.find_one({"email": email})
                if existing_user:
                    st.toast("This email is already registered. Please use a different email !", icon="⚠️")
                else:
                    create_user(email, new_username_firstname, new_username_lastname, new_password)
                    st.session_state["is_register"] = False  
                    st.session_state["register_message"] = "Registration successful !"
                    st.rerun()

        else:
            st.error("Please enter a username and password")

    # Nút chuyển sang đăng nhập
    if st.button("Switch to Login", use_container_width=True):
        st.session_state["is_register"] = False
        st.rerun()

if "register_message" in st.session_state:
    st.toast(st.session_state["register_message"])
    del st.session_state["register_message"] 

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
                st.session_state["role"] = authenticate_user(email, password).get("role")
                st.session_state["email"] = email
                st.session_state["show_success_message"] = True
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.toast("Invalid email or password !")
    
        # Nút chuyển sang đăng ký
        if st.button("Switch to Register", use_container_width=True):
            st.session_state["is_register"] = True
            st.rerun()

def logout():
    st.subheader("Are you sure you want to log out?")
    if st.button("Log out"):
        st.session_state.logged_in = False
        del st.session_state.email
        del st.session_state.role
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

# Admin
user = st.Page(
    "pages/admin/users_manage.py", title="Users", icon=":material/person:", default=True)
video = st.Page(
    "pages/admin/videos_manage.py", title="Videos", icon=":material/movie:")

# --- PAGE SETUP ---
# User
about = st.Page(
    page="pages/user/about.py", title="How to use", icon=":material/help:", default=True)
monitor = st.Page(
    page="pages/user/monitor.py", title="Monitoring", icon=":material/psychology:")
processed = st.Page(
    page="pages/user/processed.py", title="Processed Video", icon=":material/database:")

if st.session_state.logged_in:
    if st.session_state["role"] == 1:
        pg = st.navigation(
            {
                "Account": [logout_page],
                "Start Tracking": [about, monitor, processed],
            }
        )
    else:
        pg = st.navigation(
            {
                "Account": [logout_page],
                "Management": [user, video],
            }
        )
else:
    
    if st.session_state.is_register:
        pg = st.navigation([register_page])
    else:
        pg = st.navigation([login_page])

pg.run()

