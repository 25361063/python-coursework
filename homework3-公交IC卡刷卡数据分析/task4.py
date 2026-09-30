"""
任务4：高峰小时系数计算
"""
import pandas as pd
import numpy as np

# ===== 数据预处理 =====
df = pd.read_csv('ICData.csv')
df['交易时间'] = pd.to_datetime(df['交易时间'])
df['hour'] = df['交易时间'].dt.hour
df['ride_stops'] = abs(df['下车站点'] - df['上车站点'])
df = df[df['ride_stops'] != 0].reset_index(drop=True)
filtered = df[df['刷卡类型'] == 0].copy()

# ===== 步骤1：统计全天各小时刷卡量，自动识别高峰小时 =====
hourly = filtered.groupby('hour').size()
peak_hour = hourly.idxmax()    # 返回最大值对应的索引（高峰小时编号）
peak_count = hourly.max()      # 返回最大值（高峰小时刷卡量）
print(f'高峰小时: {peak_hour:02d}:00 ~ {peak_hour+1:02d}:00，刷卡量: {peak_count} 次')

# ===== 步骤2：提取高峰小时的所有记录 =====
peak_data = filtered[filtered['hour'] == peak_hour].copy()
# 计算从整点开始的偏移分钟数（带小数，包含秒）
# 例如 09:15:30 → minute_offset = 15 + 30/60 = 15.5
peak_data['minute'] = peak_data['交易时间'].dt.minute
peak_data['second'] = peak_data['交易时间'].dt.second
peak_data['minute_offset'] = peak_data['minute'] + peak_data['second'] / 60

# ===== 步骤3：5分钟粒度统计 =====
# // 整除：minute_offset // 5 → 窗口编号 0~11
peak_data['bin5'] = (peak_data['minute_offset'] // 5).astype(int)
# 边界处理：minute_offset=60 时 bin5=12，归入最后一个窗口
peak_data.loc[peak_data['bin5'] >= 12, 'bin5'] = 11

counts5 = peak_data.groupby('bin5').size()
max5_bin = counts5.idxmax()       # 窗口编号
max5_count = counts5.max()        # 该窗口的刷卡量
start5_min = max5_bin * 5
end5_min = start5_min + 5
print(f'最大5分钟刷卡量（{peak_hour:02d}:{start5_min:02d}~{peak_hour:02d}:{end5_min:02d}）: {max5_count} 次')

# PHF5 = 高峰小时刷卡量 / (12 × 最大5分钟刷卡量)
PHF5 = peak_count / (12 * max5_count)
print(f'PHF5  = {peak_count} / (12 × {max5_count}) = {PHF5:.4f}')

# ===== 步骤4：15分钟粒度统计 =====
# // 15 → 窗口编号 0~3
peak_data['bin15'] = (peak_data['minute_offset'] // 15).astype(int)
peak_data.loc[peak_data['bin15'] >= 4, 'bin15'] = 3

counts15 = peak_data.groupby('bin15').size()
max15_bin = counts15.idxmax()
max15_count = counts15.max()
start15_min = max15_bin * 15
end15_min = start15_min + 15
print(f'最大15分钟刷卡量（{peak_hour:02d}:{start15_min:02d}~{peak_hour:02d}:{end15_min:02d}）: {max15_count} 次')

# PHF15 = 高峰小时刷卡量 / (4 × 最大15分钟刷卡量)
PHF15 = peak_count / (4 * max15_count)
print(f'PHF15 = {peak_count} / (4 × {max15_count}) = {PHF15:.4f}')