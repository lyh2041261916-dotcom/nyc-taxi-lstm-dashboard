# app.py (Main Entry)
import streamlit as st
import time
from auth import init_login_state, user_login, user_logout

# Global page config (ONLY set here, do NOT repeat in pages)
st.set_page_config(
    page_title="Traffic Prediction & Visualization Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize login status
init_login_state()

# ---------------------- Login Page (Not Logged In) ----------------------
if not st.session_state.logged_in:
    st.title("🔐 Traffic Prediction & Visualization Platform - Login")
    st.divider()

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # 使用 st.form 包裹，完美支持键盘回车登录，并防止输入时页面频繁刷新
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username", placeholder="admin / user1 / user2")
            password = st.text_input("Password", type="password", placeholder="123456 / 111111 / 222222")
            login_btn = st.form_submit_button("Sign In", type="primary", use_container_width=True)

            if login_btn:
                if not username or not password:
                    st.warning("Please enter username and password!")
                else:
                    with st.spinner("Verifying account..."):
                        time.sleep(0.5)  # 稍微减小延迟，提升响应感
                        if user_login(username, password):
                            st.success("✅ Login successful, redirecting...")
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error("❌ Incorrect username or password")

    st.divider()
    st.info("""
    📌 Test Accounts:
    - Admin: admin  |  Password: 123456 (Full access)
    - User 1: user1 |  Password: 111111
    - User 2: user2 |  Password: 222222
    """)
    st.stop()  # 严格拦截，未登录不执行后续侧边栏与主页代码

# ---------------------- Public Sidebar (Logged In) ----------------------
st.sidebar.success(f"👤 Current User: {st.session_state.current_user}")
# 注意：此处 user_logout 内部不再包含 rerun
st.sidebar.button("🚪 Logout", on_click=user_logout, type="secondary", use_container_width=True)
st.sidebar.divider()
st.sidebar.info("💡 Select function from the menu above to start exploring.")

# ---------------------- Main Welcome Page ----------------------
st.title("🚗 Welcome to Traffic Prediction Platform")
st.markdown(f"Hello **{st.session_state.current_user}**, please use the sidebar to navigate through the platform features.")