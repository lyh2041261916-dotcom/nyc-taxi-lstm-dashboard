# auth.py
import streamlit as st

# Simulate user account
SIMULATED_USERS = {
    "admin": "123456",
    "user1": "111111",
    "user2": "222222"
}

def init_login_state():
    """Initialize login session state"""
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "current_user" not in st.session_state:
        st.session_state.current_user = None

def user_login(username: str, password: str) -> bool:
    """Simulate account verification"""
    if username in SIMULATED_USERS and SIMULATED_USERS[username] == password:
        st.session_state.logged_in = True
        st.session_state.current_user = username
        return True
    return False

def user_logout():
    """Logout and refresh page"""
    st.session_state.logged_in = False
    st.session_state.current_user = None
    # 🚨 移除了 st.rerun()。因为此函数挂载在按钮的 on_click 上，
    # Streamlit 会在回调结束后自动 rerun。这里写 rerun 会在最新版环境中报错。