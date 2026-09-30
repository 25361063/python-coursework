"""
任务2(a)：早晚时段刷卡量统计（必须使用 numpy）
"""
import pandas as pd
import numpy as np

# ===== 数据预处理（与 task1 相同） =====
df = pd.read_csv('ICData.csv')
df['交易时间'] = pd.to_datetime(df['交易时间'])
df['hour'] = df['交易时间'].dt.hour
df['ride_stops'] = abs(df['下车站点'] - df['上车站点'])
df = df[df['ride_stops'] != 0].reset_index(drop=True)

# 只保留刷卡类型=0 的正常记录
filtered = df[df['刷卡类型'] == 0]

# ===== 用 numpy 布尔索引统计 =====
# .values 将 pandas Series 转为 numpy array
# hours < 7 返回布尔数组 [False, True, ...]
# np.sum() 对布尔数组求和：True=1, False=0
hours = filtered['hour'].values
early_count = np.sum(hours < 7)     # 早峰前：hour 为 0~6
late_count = np.sum(hours >= 22)    # 深夜：hour 为 22~23
total_count = len(hours)

print(f'早峰前刷卡量（hour < 7）: {early_count} 次')
print(f'深夜刷卡量（hour >= 22）: {late_count} 次')
print(f'全天总刷卡量（刷卡类型=0）: {total_count} 次')
print(f'早峰前占比: {early_count / total_count * 100:.2f}%')
print(f'深夜占比: {late_count / total_count * 100:.2f}%')