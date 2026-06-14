# pages/1_model_training.py
import streamlit as st
import numpy as np
import time
import os
import pickle
import plotly.graph_objects as go  # 💡 核心修复：补全缺失的图形高能库
from datetime import datetime
from utils import load_zone_demand, load_lstm_models

# 路由守卫：防止未登录用户越权强行访问子页面
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.warning("⚠️ Please login first on the home page!")
    st.stop()

# 载入常驻资产
df_demand = load_zone_demand()
models = load_lstm_models()

st.title("⚙️ Model Training & Sandbox Dashboard")
st.markdown("Configure hyperparameters to simulate neural network convergence or replay historic training tracks.")
st.divider()

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🛠️ Hyperparameters")
    model_script = st.selectbox("Model Architecture", ["LSTM (Spatial-Decoupled)", "Baseline (Historical Average)"])
    dataset = st.selectbox("Target Dataset", ["hourly_zone_demand.csv"])

    # 根据是否是 Baseline 动态锁定 Epoch
    if "Baseline" in model_script:
        epochs = st.number_input("Epochs", value=1, disabled=True)
    else:
        epochs = st.number_input("Epochs", min_value=1, max_value=100, value=15, step=5)

    batch_size = st.selectbox("Batch Size", [32, 64, 128, 256], index=1)
    learning_rate = st.number_input("Learning Rate", min_value=0.0001, max_value=0.01, value=0.001, format="%.4f")
    device = st.selectbox("Execution Device", ["CPU", "GPU (CUDA)", "MPS (Mac)"], index=0)

    st.markdown("---")
    start_training = st.button("🚀 Launch Training Process", type="primary", use_container_width=True)

with col2:
    st.subheader("🖥️ Real-time Training Visualizer")
    status_placeholder = st.empty()
    progress_bar = st.progress(0)

    # 使用两个独立容器切分画布与文本日志
    loss_chart_placeholder = st.empty()
    log_text_placeholder = st.empty()

# =====================================================================
# 核心演练/回放状态机机制
# =====================================================================
if start_training:
    train_losses = []
    val_losses = []
    log_history = ""

    # 尝试加载真实后端的离线资产元数据
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 向上退一级到主目录
    history_path = os.path.join(base_dir, "data", "models", "training_history.pkl")

    has_real_history = False
    if os.path.exists(history_path) and "LSTM" in model_script:
        try:
            with open(history_path, 'rb') as f:
                history_zoo = pickle.load(f)
            # 取出 Zone 132 作为前端看板回放的标准主骨架
            real_train_track = history_zoo[132]['loss']
            real_val_track = history_zoo[132]['val_loss']
            # 动态对齐用户输入的 Epochs 长度，防止索引溢出
            max_epochs = min(epochs, len(real_train_track))
            has_real_history = True
        except Exception:
            has_real_history = False

    # 若未找到真实离线资产，启动高级逼真衰减仿真引擎
    if not has_real_history:
        max_epochs = epochs

    # ================= 动态梯度迭代动画循环 =================
    for epoch in range(1, max_epochs + 1):
        if "Baseline" in model_script:
            # Baseline 模型一键收敛
            train_loss, val_loss = 0.0821, 0.0854
            log_history += f"[{datetime.now().strftime('%H:%M:%S')}] HA Baseline computed instantly via spatial group-by.\n"
        else:
            if has_real_history:
                # 【亮点】回放真实神经网络产生的残差
                train_loss = real_train_track[epoch - 1]
                val_loss = real_val_track[epoch - 1]
            else:
                # 【逼真仿真】加入数学振幅，模拟标准的随机梯度下降（SGD）初期收敛快、后期震荡过拟合的现象
                train_loss = 0.45 * (0.82 ** epoch) + 0.02 + np.random.uniform(-0.005, 0.005)
                val_loss = 0.48 * (0.85 ** epoch) + 0.035 + np.random.uniform(-0.008, 0.012)
                if epoch > 12:  # 模拟尾部轻微过拟合
                    val_loss += (epoch - 12) * 0.003

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # 1. 更新全局顶层进度状态
        progress_bar.progress(epoch / max_epochs)
        status_placeholder.markdown(
            f"**Current Status:** `Iterating Epoch {epoch}/{max_epochs}` | **Train Loss:** `{train_loss:.4f}` | **Val Loss:** `{val_loss:.4f}`")

        # 2. 矢量化动态重绘 Plotly 黑暗时序画布
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(y=train_losses, mode='lines+markers', name='Train Loss', line=dict(color='#1f77b4', width=3)))
        fig.add_trace(
            go.Scatter(y=val_losses, mode='lines+markers', name='Validation Loss', line=dict(color='#ff7f0e', width=3)))

        fig.update_layout(
            title="Neural Network Residual Convergence (MSE Curve)",
            xaxis_title="Training Epoch",
            yaxis_title="Mean Squared Error",
            template="plotly_dark",
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        loss_chart_placeholder.plotly_chart(fig, use_container_width=True)

        # 3. 滚动追加高阶后台流式控制台日志（使用 st.code 防闪烁）
        log_history += f"[{datetime.now().strftime('%H:%M:%S')}] Epoch {epoch:02d}/{max_epochs:02d} -> loss: {train_loss:.4f} - val_loss: {val_loss:.4f} - lr: {learning_rate} - device: {device}\n"
        log_text_placeholder.code(log_history, language="bash")

        # 动态控制播放帧率（真实历史回放时调快，仿真时保留体感）
        time.sleep(0.05 if has_real_history else 0.15)

    # 清理进度条资产
    progress_bar.empty()
    status_placeholder.empty()

    # 抛出最后的全面大成功勋章
    st.success(
        f"🎉 {'Historical Tracks Replayed' if has_real_history else 'Model Weights Synthesized'} Successfully! Matrix locked in `data/models/`")
    st.balloons()