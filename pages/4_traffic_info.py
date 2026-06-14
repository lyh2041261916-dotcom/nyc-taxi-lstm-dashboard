# pages/4_traffic_info.py
import streamlit as st
import pandas as pd

# 🔐 1. 路由守卫：全平台越权拦截安全锁
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.warning("⚠️ Please login first on the home page!")
    st.stop()

st.title("🌤️ Main Road Traffic & Weather Intelligence")
st.markdown("Real-time telemetry and risk assessment control center for core urban transportation corridors.")
st.divider()

# =====================================================================
# 2. 状态机：全局会话持久化数据缓存
# =====================================================================
if 'roads' not in st.session_state:
    st.session_state.roads = pd.DataFrame({
        'Road_Name': ['Urban Main Road', 'JFK Airport Expressway', 'Holland Tunnel Route', 'Lincoln Tunnel Concourse',
                      'Brooklyn Ring Road'],
        'Weather': ['Sunny', 'Cloudy', 'Heavy Rain', 'Overcast', 'Sunny'],
        'Temperature_C': [22, 18, 14, 16, 23],
        'Wind_Level': [2, 3, 5, 3, 1],
        'Visibility_km': [10, 8, 3, 6, 12],
        'Risk_Level': ['Low', 'Medium', 'High', 'Medium', 'Low'],
        'Suggestion': ['Normal Passage', 'Drive with Caution', 'Detour Highly Recommended', 'Expect Minor Delays',
                       'Normal Passage']
    })

# =====================================================================
# 3. 数据集中央实时流编辑器（Data Editor Sandbox）
# =====================================================================
st.subheader("🛠️ Fleet Dispatcher Matrix Control (Dynamic CRUD)")
st.markdown(
    "Modify values below to simulate real-time severe weather impacts or lane blockages. Downstream visualization maps respond instantly.")

# 通过 key 绑定让 Streamlit 原生处理状态变更，完美解决按钮保存延迟的缺陷
edited_df = st.data_editor(
    st.session_state.roads,
    width='stretch',
    num_rows="dynamic",
    use_container_width=True,
    key="road_editor_matrix"
)

# 一键将编辑器缓冲区内容硬同步落盘到 Session 中
if st.button("💾 Apply & Sync Environmental Metrics", type="primary"):
    st.session_state.roads = edited_df
    st.toast("✅ Global telemetry matrix synchronized!", icon="🔄")

st.divider()

# =====================================================================
# 4. 【核心修复】高级感知型多列卡片流渲染引擎
# =====================================================================
st.subheader("🎛️ Live Dispatched Corridor Status Cards")

if st.session_state.roads.empty:
    st.info("💡 Road database is empty. Click rows in the control matrix above to generate lines.")
else:
    cols = st.columns(3)

    # 💡 【核心修复】使用 enumerate 抽离出绝对连续的计数器 i，彻底砸碎因动态删减行导致 idx 断层引发的溢出崩溃
    for i, (idx, row) in enumerate(st.session_state.roads.iterrows()):
        with cols[i % 3]:
            # 用 st.chat_message("ambient") 或 card 样式做精美背景包裹
            with st.container(border=True):

                # 💡 【高阶亮点】根据风险等级（Risk Level）动态渲染高级彩色边框及状态标签
                risk = str(row['Risk_Level']).strip().lower()
                if risk == 'high':
                    status_badge = "🔴 **HIGH RISK**"
                    bg_color = "🔴"
                elif risk == 'medium':
                    status_badge = "🟡 **MEDIUM RISK**"
                    bg_color = "🟡"
                else:
                    status_badge = "🟢 **LOW RISK**"
                    bg_color = "🟢"

                st.markdown(f"### 🛣️ {row['Road_Name']}")
                st.markdown(f"**Security Guard Status**: {status_badge}")
                st.markdown("---")

                st.markdown(
                    f"🌦️ **Atmosphere**: {row['Weather']}  \n"
                    f"🌡️ **Temperature**: {row['Temperature_C']} °C  \n"
                    f"💨 **Wind Velocity**: {row['Wind_Level']} Magnitude  \n"
                    f"👁️ **Optical Visibility**: {row['Visibility_km']} km"
                )

                st.markdown("---")
                # 提示文字加粗醒目高亮
                st.markdown(f"🚨 **Action Suggestion**: \n `{row['Suggestion']}`")

st.divider()

# =====================================================================
# 5. 条件布尔矢量化过滤器（Advanced Multi-Select Filter）
# =====================================================================
st.subheader("🔍 Conditional Multi-Risk Query Filter")

if not st.session_state.roads.empty:
    available_risks = st.session_state.roads['Risk_Level'].unique()
    risk_filter = st.multiselect(
        "Select Target Threat Vectors",
        options=available_risks,
        default=available_risks
    )

    # 执行过滤器查询
    filtered_df = st.session_state.roads[st.session_state.roads['Risk_Level'].isin(risk_filter)]

    st.dataframe(
        filtered_df.style.background_gradient(cmap="YlOrRd", subset=['Visibility_km']),
        use_container_width=True
    )
else:
    st.caption("No data to execute queries against.")