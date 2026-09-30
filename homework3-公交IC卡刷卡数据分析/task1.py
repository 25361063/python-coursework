"""
任务1：数据预处理
"""
import pandas as pd
import numpy as np

# ===== 1. 读取数据 =====
df = pd.read_csv('ICData.csv')
print('===== 前5行 =====')
print(df.head())
print(f'\n行数: {df.shape[0]}, 列数: {df.shape[1]}')
print('\n各列数据类型:')
print(df.dtypes)

# ===== 2. 时间解析 =====
# 交易时间列当前是字符串(object)，需要转为 datetime 类型
df['交易时间'] = pd.to_datetime(df['交易时间'])
# 用 .dt 访问器提取小时(0~23的整数)，新增为 hour 列
df['hour'] = df['交易时间'].dt.hour

# ===== 3. 构造衍生字段 =====
# ride_stops = |下车站点 - 上车站点|，表示搭乘经过的站点数
df['ride_stops'] = abs(df['下车站点'] - df['上车站点'])
# 删除 ride_stops=0 的异常记录（上下车站点相同不合理）
before_count = len(df)
df = df[df['ride_stops'] != 0]
deleted_count = before_count - len(df)
print(f'\n删除 ride_stops=0 的异常记录: {deleted_count} 行')
df = df.reset_index(drop=True)

# ===== 4. 缺失值检查 =====
print('\n各列缺失值数量:')
missing = df.isnull().sum()
print(missing)
if missing.sum() > 0:
    df = df.dropna()
    print(f'已删除含缺失值的记录，剩余 {len(df)} 行')
else:
    print('无缺失值，无需删除')
print(f'\n预处理后数据量: {len(df)} 行')