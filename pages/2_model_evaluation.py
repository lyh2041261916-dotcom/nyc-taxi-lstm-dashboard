# pages/2_model_evaluation.py
import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import timedelta
import plotly.graph_objects as go  # 💡 核心修复：补全缺失的 Plotly 拓扑画布引擎
from sklearn.metrics import mean_absolute_error, mean_squared_error

from utils import (
    load_zone_demand, load_zone_names, load_model_metrics,
    load_lstm_models, predict_next_hour, baseline_predict,
    df_to_download_link, TOP_ZONES
)

# 🔐 路由守卫：越权拦截安全锁
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.warning("⚠️ Please login first on the home page!")
    st.stop()

# 统一载入常驻资产
df_demand = load_zone_demand()
zone_names = load_zone_names()
models = load_lstm_models()
eval_metrics, baseline_metrics = load_model_metrics()

st.title("📈 Model Evaluation & Residual Analytics")
st.markdown("Validate neural network variance and contrast performance metrics against historical baseline benchmarks.")
st.divider()

col1, col2 = st.columns([1, 1.5])

with col1:
    selected_zone = st.selectbox(
        "🎯 Select Spatial Zone", TOP_ZONES,
        format_func=lambda x: f"Zone {x} - {zone_names.get(x, 'Unknown Hub')}"
    )

with col2:
    max_date = df_demand['pickup_hour'].max()
    min_date = max_date - timedelta(days=7)  # 默认看最近 7 天

    date_range = st.date_input(
        "📅 Evaluation Time Window",
        value=[min_date, max_date],
        min_value=df_demand['pickup_hour'].min(),
        max_value=max_date
    )

    # 💡 【核心修复】防御性防御：若用户尚未选定完整的 [开始, 结束] 区间，暂缓下游计算，防止解包溢出闪退
    if len(date_range) < 2:
        st.info("💡 Please select both start and end dates in the calendar picker.")
        st.stop()

    start_date, end_date = date_range[0], date_range[1]

# =====================================================================
# 时间网格切片与前向推理机流式计算
# =====================================================================
zone_data = df_demand[df_demand['PULocationID'] == selected_zone].copy()
zone_data = zone_data[
    (zone_data['pickup_hour'] >= pd.Timestamp(start_date)) &
    (zone_data['pickup_hour'] <= pd.Timestamp(end_date))
    ]
zone_data = zone_data.sort_values('pickup_hour')

true_values = zone_data['count'].values
timestamps = zone_data['pickup_hour'].values

if len(true_values) == 0:
    st.error("❌ No demand history detected within this slice. Please expand the time range.")
    st.stop()

with st.spinner("⏳ Run-time Inference Engine propagating predictions..."):
    lstm_predictions = []
    baseline_predictions = []
    model_lstm = models.get(selected_zone, None)

    for ts in timestamps:
        # 调用重构后的、带标准离线 Scaler 算子对齐的前向传播推理机
        pred_lstm = predict_next_hour(selected_zone, ts, df_demand, model_lstm)
        lstm_predictions.append(pred_lstm)

        # 调用基线均值推理
        pred_base = baseline_predict(selected_zone, ts, df_demand)
        baseline_predictions.append(pred_base)

# =====================================================================
# 核心指标量化看板（KPI Cards Layout）
# =====================================================================
mae_lstm = mean_absolute_error(true_values, lstm_predictions) if model_lstm else None
rmse_lstm = np.sqrt(mean_squared_error(true_values, lstm_predictions)) if model_lstm else None
mape_lstm = np.mean(np.abs((true_values - lstm_predictions) / (true_values + 1e-5))) * 100 if model_lstm else None

mae_base = mean_absolute_error(true_values, baseline_predictions)
rmse_base = np.sqrt(mean_squared_error(true_values, baseline_predictions))
mape_base = np.mean(np.abs((true_values - baseline_predictions) / (true_values + 1e-5))) * 100

st.subheader("📊 Slice KPI Dashboard (Current Window)")
kpi1, kpi2, kpi3 = st.columns(3)
with kpi1:
    st.metric("LSTM MAE", f"{mae_lstm:.2f}" if model_lstm else "N/A",
              delta=f"{(mae_lstm - mae_base):.2f} vs Base", delta_color="inverse")
with kpi2:
    st.metric("LSTM RMSE", f"{rmse_lstm:.2f}" if model_lstm else "N/A",
              delta=f"{(rmse_lstm - rmse_base):.2f} vs Base", delta_color="inverse")
with kpi3:
    st.metric("LSTM MAPE", f"{mape_lstm:.2f}%" if model_lstm else "N/A",
              delta=f"{(mape_lstm - mape_base):.2f}% vs Base", delta_color="inverse")

# =====================================================================
# 拟合度比对高级时序渲染图
# =====================================================================
st.subheader("📉 Time-Series Fitting Comparison")
fig_single = go.Figure()
fig_single.add_trace(
    go.Scatter(x=timestamps, y=true_values, mode='lines', name='Ground Truth (Actual)',
               line=dict(color='#ffffff', width=2))
)
if model_lstm:
    fig_single.add_trace(
        go.Scatter(x=timestamps, y=lstm_predictions, mode='lines+markers', name='Custom LSTM Net',
                   line=dict(color='#00ffff', width=2, size=4))
    )
fig_single.add_trace(
    go.Scatter(x=timestamps, y=lstm_predictions, mode='lines+markers',
               line=dict(color='#00ffff', width=2),  # ✅ 宽度控制
               marker=dict(size=4))                  # ✅ 大小控制放到 marker 里
)
fig_single.update_layout(
    title=dict(text=f"Dynamic Flow Tracking for Zone {selected_zone}", font=dict(size=16)),
    xaxis_title="Timeline Step", yaxis_title="Passenger Pickups / Hour",
    template="plotly_dark",
    margin=dict(l=20, r=20, t=50, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig_single, use_container_width=True)

# =====================================================================
# 资产导出中央工作台
# =====================================================================
st.subheader("📥 Export & Analytics Center")
metrics_df = pd.DataFrame({
    'Model_Type': ['Custom LSTM', 'HA Baseline'],
    'MAE': [f"{mae_lstm:.4f}" if model_lstm else "N/A", f"{mae_base:.4f}"],
    'RMSE': [f"{rmse_lstm:.4f}" if model_lstm else "N/A", f"{rmse_base:.4f}"],
    'MAPE_Percentage': [f"{mape_lstm:.2f}%" if model_lstm else "N/A", f"{mape_base:.2f}%"]
})

# 构造供完整下载的明细对账宽表
report_export_df = pd.DataFrame({
    'Timestamp': timestamps,
    'Actual_Demand': true_values,
    'LSTM_Predicted': lstm_predictions,
    'Baseline_Predicted': baseline_predictions
})

d1, d2 = st.columns(2)
with d1:
    st.markdown(df_to_download_link(metrics_df, "summary_metrics.csv"), unsafe_allow_html=True)
with d2:
    st.markdown(df_to_download_link(report_export_df, "hourly_predictions_matchcase.csv"), unsafe_allow_html=True)

# 全局大盘多区域联动报表（若离线端计算完毕则渲染）
if eval_metrics is not None and baseline_metrics is not None:
    st.markdown("---")
    st.subheader("🌐 Global Cross-Zone Evaluation Overview")

    # 动态匹配索引
    comparison_matrix = pd.DataFrame({
        'Zone Identification': [f"Zone {z}" for z in eval_metrics.index],
        'LSTM Architecture MAE': eval_metrics['MAE'].values,
        'Baseline Benchmark MAE': baseline_metrics['MAE'].values
    })

    # 渲染带有渐变色高亮的全局指标表，视觉极度高级
    st.dataframe(
        comparison_matrix.style.background_gradient(subset=['LSTM Architecture MAE', 'Baseline Benchmark MAE'],
                                                    cmap='Blues'),
        use_container_width=True
    )
