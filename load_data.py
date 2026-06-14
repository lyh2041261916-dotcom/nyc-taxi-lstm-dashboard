# load_data.py
import os
import pandas as pd

# 1. 绝对防御路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
DATA_LSTM_DIR = os.path.join(BASE_DIR, "data", "lstm_data")

# 💡 【核心修复】这是一个干净的、不带任何 st 装饰器的基础加载函数，供离线脚本直接调用
def load_raw_parquet_raw_accessible():
    parquet_path = os.path.join(DATA_RAW_DIR, "yellow_tripdata_2024_combined.parquet")
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"找不到原始 Parquet 文件，请确认路径: {parquet_path}")
    # 显式指定 pyarrow 引擎确保 Python 3.13 下的高性能
    return pd.read_parquet(parquet_path, engine='pyarrow')

def load_processed_demand_raw_accessible():
    csv_path = os.path.join(DATA_PROCESSED_DIR, "hourly_zone_demand.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到聚合时序表，请先运行 preprocess.py。路径: {csv_path}")
    return pd.read_csv(csv_path, parse_dates=['pickup_hour'])

def load_zone_lookup_raw_accessible():
    lookup_path = os.path.join(DATA_RAW_DIR, "taxi_zone_lookup.csv")
    if not os.path.exists(lookup_path):
        raise FileNotFoundError(f"找不到区域字典映射表，路径: {lookup_path}")
    return pd.read_csv(lookup_path)


# =====================================================================
# 2. 仅在 Streamlit 运行时环境下包装的缓存层（供 utils.py 或前端直接继承）
# =====================================================================
import streamlit as st

@st.cache_data
def load_raw_parquet():
    return load_raw_parquet_raw_accessible()

@st.cache_data
def load_processed_demand():
    return load_processed_demand_raw_accessible()

@st.cache_data
def load_zone_lookup():
    return load_zone_lookup_raw_accessible()