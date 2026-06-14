# utils.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import base64
from datetime import timedelta
import joblib

# 统一引入 load_data 中定义的规范化全局路径
from load_data import (
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    DATA_LSTM_DIR,
    load_processed_demand,
    load_zone_lookup
)

# 全局静态常量
KEEP_DAYS = 30
TOP_ZONES = [132, 161, 237, 236, 162]


# ---------------------- 1. 数据与模型资产高效加载层 ----------------------

@st.cache_data(show_spinner="📊 Loading platform time-series subset...")
def load_zone_demand() -> pd.DataFrame:
    """
    复用底层标准加载，并切分前端展示所需的最近 KEEP_DAYS (30天) 的轻量化数据
    """
    try:
        df = load_processed_demand()
        max_date = df['pickup_hour'].max()
        min_date = max_date - timedelta(days=KEEP_DAYS)
        # 切片，降低 Streamlit 前端渲染图表的内存压力
        return df[df['pickup_hour'] >= min_date].copy()
    except Exception as e:
        st.error(f"❌ Failed to load hourly demand: {str(e)}")
        return pd.DataFrame()


@st.cache_data(show_spinner="🗺️ Mapping spatial location names...")
def load_zone_names() -> dict:
    """
    加载纽约出租车分区字典，构建 ID -> 真实地名的映射
    """
    try:
        df = load_zone_lookup()
        return dict(zip(df['LocationID'], df['Zone']))
    except Exception:
        return {}


@st.cache_data(show_spinner="📈 Loading analytical metric reports...")
def load_model_metrics():
    """
    加载离线保存的 LSTM 与 Baseline 量化评估指标报表
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    eval_path = os.path.join(base_dir, "data", "models", "model_evaluation.csv")
    baseline_path = os.path.join(base_dir, "data", "models", "baseline_results.csv")

    eval_df = pd.read_csv(eval_path, index_col=0) if os.path.exists(eval_path) else None
    baseline_df = pd.read_csv(baseline_path, index_col=0) if os.path.exists(baseline_path) else None
    return eval_df, baseline_df


@st.cache_resource(show_spinner="🧠 Initializing Neural Network Runtime (Keras 3)...")
def load_lstm_models():
    """
    利用 cache_resource 保持 5 个区域的神经网络权重常驻内存，避免重复 I/O 导致网页卡死
    """
    os.environ["KERAS_BACKEND"] = "tensorflow"
    import tensorflow as tf  # 延迟加载，防止在某些环境中阻塞进程

    base_dir = os.path.dirname(os.path.abspath(__file__))
    models = {}
    progress_text = st.empty()

    for i, z in enumerate(TOP_ZONES):
        path = os.path.join(base_dir, "data", "models", f"lstm_zone_{z}.keras")
        if os.path.exists(path):
            try:
                progress_text.info(f"🔄 Loading Custom LSTM Framework for Zone {z}... ({i + 1}/{len(TOP_ZONES)})")
                models[z] = tf.keras.models.load_model(path)
            except Exception as e:
                st.warning(f"⚠️ Failed to load model for Zone {z}: {str(e)}")
    progress_text.empty()
    return models


# ---------------------- 2. 核心全栈预测推理机（Inference Engine） ----------------------

def predict_next_hour(zone_id: int, target_datetime, df_hourly: pd.DataFrame, model, time_steps: int = 24) -> int:
    """
    神经网络前向传播推理机：提取过去24小时序列 -> 调取离线算子归一化 -> 喂入模型 -> 反归一化
    """
    if not isinstance(target_datetime, pd.Timestamp):
        target_datetime = pd.Timestamp(target_datetime)

    # 提取当前区域的时序切片
    zone_data = df_hourly[df_hourly['PULocationID'] == zone_id].sort_values('pickup_hour')
    zone_data = zone_data.set_index('pickup_hour')

    # 构建滑动窗口截取边界
    end_time = target_datetime - timedelta(hours=1)
    start_time = end_time - timedelta(hours=time_steps - 1)

    hist = zone_data.loc[start_time:end_time]['count'].values

    # 健壮性防御：若前序样本不足，进行 Zero-Padding 填充
    if len(hist) < time_steps:
        hist = np.pad(hist, (time_steps - len(hist), 0), 'constant')

    if model is not None:
        try:
            # 🚨 【核心修复】拒绝使用自制临时归一化！统一从中央仓库加载离线算子
            base_dir = os.path.dirname(os.path.abspath(__file__))
            scalers_path = os.path.join(base_dir, "data", "lstm_data", "zone_scalers.pkl")
            scalers = joblib.load(scalers_path)
            current_scaler = scalers[zone_id]['scaler']

            # 1. 严格使用离线训练集的参数进行归一化
            hist_norm = current_scaler.transform(hist.reshape(-1, 1)).flatten()

            # 2. 变换形状以符合 Keras 3D 张量输入格式: [1, 24, 1]
            X = hist_norm.reshape(1, time_steps, 1)

            # 3. 前向传播预测
            pred_norm = model.predict(X, verbose=0)

            # 4. 反归一化还原真实物理量纲
            pred_real = current_scaler.inverse_transform(pred_norm).flatten()[0]
            return int(max(0, round(pred_real)))
        except Exception as e:
            # 算子缺失时的兜底：采用均值
            return int(hist.mean()) if len(hist) > 0 else 0
    else:
        # 无模型时降级为简易均值模型
        return int(hist.mean()) if len(hist) > 0 else 0


def baseline_predict(zone_id: int, target_datetime, df_hourly: pd.DataFrame) -> int:
    """
    计算经典的 Baseline 模型预测值（基于同区域、同星期、同小时的历史均值）
    """
    if not isinstance(target_datetime, pd.Timestamp):
        target_datetime = pd.Timestamp(target_datetime)

    zone_data = df_hourly[df_hourly['PULocationID'] == zone_id]
    weekday = target_datetime.weekday()
    hour = target_datetime.hour

    # 过滤出历史上同一星期几、同一小时的所有数据点
    matched_records = zone_data[
        (zone_data['pickup_hour'].dt.weekday == weekday) &
        (zone_data['pickup_hour'].dt.hour == hour)
        ]

    avg = matched_records['count'].mean()
    return int(round(avg)) if not np.isnan(avg) else 0


# ---------------------- 3. 高级空间特征分析工具 ----------------------

@st.cache_data
def get_demand_matrix_and_corr(top_zones: list, df_demand: pd.DataFrame):
    """
    【高性能重构】将一维长表通过透视矩阵转为高维宽表，秒级计算 5 大枢纽的空间相关性矩阵
    """
    # 筛选出 5 大枢纽的数据
    df_sub = df_demand[df_demand['PULocationID'].isin(top_zones)]

    # 优雅的矢量化透视转化：行索引为时间，列名变为区域 ID，值为交通流量
    demand_matrix = df_sub.pivot(index='pickup_hour', columns='PULocationID', values='count')

    # 处理可能的缺失值（虽然 preprocess 已经补 0，此处双重保险）
    demand_matrix = demand_matrix.fillna(0)

    # 保证列的排列顺序与传入的 top_zones 严格一致
    demand_matrix = demand_matrix.reindex(columns=top_zones)

    # 矢量化一键计算皮尔逊相关系数矩阵 (Pearson Correlation)
    corr_matrix = demand_matrix.corr().values
    return demand_matrix, corr_matrix


# ---------------------- 4. 生产级资产数据导出挂载器 ----------------------
def df_to_download_link(df: pd.DataFrame, filename: str) -> str:
    """
    将 pandas DataFrame 转换为二进制 base64 编码，生成纯前端 HTML 极速下载超链接
    """
    csv = df.to_csv(index=False).encode('utf-8')
    b64 = base64.b64encode(csv).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" style="text-decoration:none; color:#1f77b4; font-weight:bold;">📥 Click Here to Download {filename}</a>'
    return href