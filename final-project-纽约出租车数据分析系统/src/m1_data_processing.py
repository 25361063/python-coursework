# src/m1_data_processing.py

import os
import pandas as pd
import numpy as np
from datetime import datetime


def process_data(data_path="data/yellow_tripdata_2026-01.parquet",
                 output_dir="outputs"):
    """
    执行 M1 数据加载、质量报告、清洗、特征工程。

    参数:
        data_path: parquet 文件路径
        output_dir: 输出目录

    返回:
        df_clean: 清洗后的 DataFrame（含新特征）
    """
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # ==================== 1. 加载数据 ====================
    print("正在加载数据...")
    df = pd.read_parquet(data_path)
    print(f"原始数据量: {len(df)} 条记录")

    # ==================== 2. 数据质量报告 ====================
    print("生成数据质量报告...")

    # 2.1 缺失率统计（所有列）
    missing_rate = df.isnull().mean().reset_index()
    missing_rate.columns = ["column", "missing_rate"]

    # 2.2 异常值统计（针对关键数值列）
    # 定义异常判断函数，返回异常比例
    def abnormal_ratio(series, condition):
        """condition 为布尔 Series，计算异常比例"""
        return condition.mean()

    # 定义各列的异常条件（可根据业务常识调整阈值）
    abnormal_stats = {}

    # 车费异常：<=0 或 >500（极少见，可视为异常）
    if "fare_amount" in df.columns:
        cond = (df["fare_amount"] <= 0) | (df["fare_amount"] > 500)
        abnormal_stats["fare_amount_abnormal"] = abnormal_ratio(df["fare_amount"], cond)

    # 行程距离异常：<=0 或 >100 英里
    if "trip_distance" in df.columns:
        cond = (df["trip_distance"] <= 0) | (df["trip_distance"] > 100)
        abnormal_stats["trip_distance_abnormal"] = abnormal_ratio(df["trip_distance"], cond)

    # 乘客数异常：<1 或 >6（出租车通常最多6人）
    if "passenger_count" in df.columns:
        cond = (df["passenger_count"] < 1) | (df["passenger_count"] > 6)
        abnormal_stats["passenger_count_abnormal"] = abnormal_ratio(df["passenger_count"], cond)

    # 行程时间异常：起止时间差为负或超过24小时
    if "tpep_pickup_datetime" in df.columns and "tpep_dropoff_datetime" in df.columns:
        # 先转换为datetime（若尚未转换）
        pickup = pd.to_datetime(df["tpep_pickup_datetime"])
        dropoff = pd.to_datetime(df["tpep_dropoff_datetime"])
        duration = (dropoff - pickup).dt.total_seconds() / 60.0  # 分钟
        cond = (duration < 0) | (duration > 24 * 60)  # 负或超过24小时
        abnormal_stats["trip_duration_abnormal"] = abnormal_ratio(duration, cond)

    # 付款方式异常：不在有效范围 (1~6)
    if "payment_type" in df.columns:
        cond = ~df["payment_type"].isin([1, 2, 3, 4, 5, 6])
        abnormal_stats["payment_type_abnormal"] = abnormal_ratio(df["payment_type"], cond)

    # 合并缺失率和异常率到一个DataFrame
    # 先转异常统计为DataFrame
    abnormal_df = pd.DataFrame(list(abnormal_stats.items()),
                               columns=["column", "abnormal_rate"])

    # 合并（左连接，保留所有列）
    quality_report = pd.merge(missing_rate, abnormal_df, on="column", how="outer")
    quality_report.fillna(0.0, inplace=True)  # 未统计的异常率填0

    # 保存质量报告
    report_path = os.path.join(output_dir, "data_quality_report.csv")
    quality_report.to_csv(report_path, index=False)
    print(f"数据质量报告已保存至: {report_path}")

    # ==================== 3. 数据清洗 ====================
    print("开始数据清洗...")
    df_clean = df.copy()

    # 3.1 删除车费异常记录（<=0 或 >500）
    # 理由：车费应大于0，极高价可能是数据错误或特殊行程，但为保主流分析，剔除。
    df_clean = df_clean[(df_clean["fare_amount"] > 0) & (df_clean["fare_amount"] <= 500)]

    # 3.2 删除行程距离异常（<=0 或 >100英里）
    # 理由：距离为零不合理，超过100英里在纽约市出租车中极少，可能是GPS错误。
    df_clean = df_clean[(df_clean["trip_distance"] > 0) & (df_clean["trip_distance"] <= 100)]

    # 3.3 删除乘客数异常（<1 或 >6）
    # 理由：出租车最多允许6名乘客（含司机？通常乘客数0不合理，>6不合理）
    df_clean = df_clean[(df_clean["passenger_count"] >= 1) & (df_clean["passenger_count"] <= 6)]

    # 3.4 删除起止时间逻辑错误的记录（负时长或超过24小时）
    # 理由：时间倒流或过长行程不合理。
    pickup = pd.to_datetime(df_clean["tpep_pickup_datetime"])
    dropoff = pd.to_datetime(df_clean["tpep_dropoff_datetime"])
    duration_min = (dropoff - pickup).dt.total_seconds() / 60.0
    df_clean = df_clean[(duration_min >= 0) & (duration_min <= 24 * 60)]

    # 3.5 删除付款方式未知（不在1-6范围）
    # 理由：有效付款方式编码为1-6，其他值无效。
    df_clean = df_clean[df_clean["payment_type"].isin([1, 2, 3, 4, 5, 6])]

    # 可选：删除 store_and_fwd_flag 异常（通常为 N 或 Y，但影响不大，暂不处理）

    print(f"清洗后数据量: {len(df_clean)} 条记录")

    # ==================== 4. 基础时间特征提取 ====================
    print("提取时间特征...")
    # 确保 pickup 列为 datetime 类型
    pickup_dt = pd.to_datetime(df_clean["tpep_pickup_datetime"])

    df_clean["pickup_hour"] = pickup_dt.dt.hour
    df_clean["pickup_dayofweek"] = pickup_dt.dt.dayofweek  # 0=周一, 6=周日
    df_clean["pickup_is_weekend"] = (pickup_dt.dt.dayofweek >= 5).astype(int)  # 周六日=1

    # 自定义高峰时段：7:00-9:00 和 17:00-19:00（工作日早晚高峰）
    hour = pickup_dt.dt.hour
    is_weekday = pickup_dt.dt.dayofweek < 5
    is_rush = ((hour >= 7) & (hour < 9)) | ((hour >= 17) & (hour < 19))
    df_clean["pickup_is_rush_hour"] = (is_weekday & is_rush).astype(int)

    # ==================== 5. 衍生特征设计 ====================
    print("设计衍生特征...")

    # 5.1 行程时长（分钟）
    # 含义：反映出行耗时，可用于分析拥堵、用户等待时间等。
    dropoff_dt = pd.to_datetime(df_clean["tpep_dropoff_datetime"])
    df_clean["trip_duration_min"] = (dropoff_dt - pickup_dt).dt.total_seconds() / 60.0

    # 5.2 平均速度（英里/小时）= 距离 / 时长（小时）
    # 含义：反映道路交通状况，速度低可能表示拥堵，可用来辅助分析区域活跃度。
    df_clean["avg_speed_mph"] = df_clean["trip_distance"] / (df_clean["trip_duration_min"] / 60.0)
    # 避免除零（但我们已经删除时长为0的记录，所以安全），若有仍可过滤
    df_clean = df_clean[df_clean["avg_speed_mph"] > 0]  # 保证速度正数

    # 可选：5.3 人均车费（总费用/乘客数），作为补充衍生特征（超出两个，但鼓励）
    # df_clean["fare_per_passenger"] = df_clean["fare_amount"] / df_clean["passenger_count"]

    # 保存清洗后的数据（供后续模块使用，比如M2/M3）
    cleaned_path = os.path.join(output_dir, "cleaned_data.parquet")
    df_clean.to_parquet(cleaned_path, index=False)
    print(f"清洗后数据已保存至: {cleaned_path}")

    print("M1 数据处理完成。")
    return df_clean


# 若直接运行此脚本，可测试
if __name__ == "__main__":
    # 假设数据文件在 ../data/ 目录下（根据你的项目结构调整）
    df = process_data(data_path="../data/yellow_tripdata_2026-01.parquet",
                      output_dir="../outputs")
    print(df.head())
    print(df.columns.tolist())