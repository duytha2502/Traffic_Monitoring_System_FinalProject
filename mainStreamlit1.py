import streamlit as st
import pymongo
from google_auth_oauthlib import get_user_credentials
from google.auth.transport import requests
from google.oauth2 import id_token
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

# Kết nối đến MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["processed_videos"]
users_collection = db["users"]

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

    .ea3mdgi5 {
        width: 1000px;
    }

    .e115fcil2 {
        width: 200px
    }

    #traffic-tracking-system {
        margin-top: -36px;
    }

    </style>

    """,
    unsafe_allow_html=True
)


# Hàm tạo tài khoản (lưu mật khẩu đã mã hóa)
def create_user(email, new_username_firstname ,new_username_lastname, password):
    password_hash = generate_password_hash(password)
    new_user = users_collection.insert_one({"role": 1, "status": "inactive", "email": email, "first_name": new_username_firstname, "last_name": new_username_lastname ,"password": password_hash, "avatar": "img/avatar.png", "created_time": datetime.now(), "updated_time": datetime.now()})
    get_new_user = users_collection.find_one({"_id": new_user.inserted_id})
    return get_new_user

# Hàm xác thực người dùng
def authenticate_user(email, password):
    user = users_collection.find_one({"email": email})
    if user and check_password_hash(user["password"], password):
        return user
    return False

def master():
    col1lg, col2lg = st.columns([1,9], vertical_alignment="center")
    with col1lg:
        st.image('http://emojivandals.com/emojis/emojivandals-cctv.gif', width=80)
    with col2lg:
        st.title(":red[Traffic Tracking System]")
    col1 , col2 = st.columns(2)        
    with col1:
        st.image("img/logo.png")
    with col2:
        tab1, tab2= st.tabs(["Sign In", "Sign Up"])
        with tab1:
            st.header("Login")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")

            if st.button("Log in", use_container_width=True, type="primary"):
                user_authenticate = authenticate_user(email, password)
                if user_authenticate:
                    if user_authenticate["status"] == "active":
                        st.session_state["role"] = authenticate_user(email, password).get("role")
                        st.session_state["email"] = email
                        st.session_state["show_success_message"] = True
                        st.session_state["logged_in"] = True
                        st.rerun()
                    else: 
                        st.toast("Your account is currently inactive. Please contact the admin!")
                else:
                    st.toast("Invalid email or password !")
        
        with tab2:
            st.header("Register")
            email = st.text_input("Email Address")
            col1n, col2n = st.columns(2)
            with col1n:
                new_username_firstname = st.text_input("First Name")
            with col2n:
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
                            new_user = create_user(email, new_username_firstname, new_username_lastname, new_password) 
                            st.session_state["role"] = new_user.get("role")
                            st.session_state["email"] = email
                            # st.session_state["logged_in"] = True
                            st.session_state["register_message"] = True
                            st.rerun()

                else:
                    st.error("Please enter a username and password")
        
# Kiểm tra trạng thái và hiển thị giao diện phù hợp
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "register_message" in st.session_state:
    st.toast("Registration successful")
    del st.session_state["register_message"] 

if "show_success_message" in st.session_state:
    st.toast(f"Welcome {st.session_state['email']} !")
    del st.session_state["show_success_message"]

def logout():
    st.subheader("Are you sure you want to log out?")
    if st.button("Log out"):
        st.session_state.logged_in = False
        del st.session_state.email
        del st.session_state.role
        st.rerun()

# --- PAGE SETUP ---
master_page = st.Page(master, title="Welcome", icon=":material/login:")

logout_page = st.Page(logout, title="Log out", icon=":material/logout:")

# Admin
log = st.Page(
    "pages/admin/check_logs.py", title="Logs", icon=":material/feed:", default=True)
user = st.Page(
    "pages/admin/users_manage.py", title="Users", icon=":material/person:")
video = st.Page(
    "pages/admin/videos_manage.py", title="Videos", icon=":material/movie:")

# User
profile = st.Page(
    page="pages/user/profile.py", title="User Profile", icon=":material/person:")
about = st.Page(
    page="pages/user/about.py", title="How To Use", icon=":material/help:", default=True)
monitor = st.Page(
    page="pages/user/monitor.py", title="Monitoring", icon=":material/psychology:")
processed = st.Page(
    page="pages/user/processed.py", title="Processed Video", icon=":material/database:")

if st.session_state.logged_in:
    if st.session_state["role"] == 1:
        pg = st.navigation(
            {
                "Account": [logout_page, profile],
                "Start Tracking": [about, monitor, processed],
            }
        )
    else:
        pg = st.navigation(
            {
                "Account": [logout_page],
                "Management": [log, user, video],
            }
        )
else:
    pg = st.navigation([master_page])

pg.run()

