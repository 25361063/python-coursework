"""
任务2(b)：24小时刷卡量分布可视化（必须使用 matplotlib）
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ===== 数据预处理 =====
df = pd.read_csv('ICData.csv')
df['交易时间'] = pd.to_datetime(df['交易时间'])
df['hour'] = df['交易时间'].dt.hour
df['ride_stops'] = abs(df['下车站点'] - df['上车站点'])
df = df[df['ride_stops'] != 0].reset_index(drop=True)
filtered = df[df['刷卡类型'] == 0]

# ===== 统计每小时刷卡量 =====
# reindex 确保 0~23 全部存在（某小时无数据时填 0）
hourly_counts = filtered.groupby('hour').size().reindex(range(24), fill_value=0)

# ===== 定义颜色 =====
colors = []
for h in range(24):
    if h < 7 or h >= 22:
        colors.append('salmon')      # 高亮：早峰前 + 深夜
    else:
        colors.append('steelblue')   # 正常：7:00~21:59

# ===== 画柱状图（必须用 matplotlib 的 ax.bar） =====
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(range(24), hourly_counts.values, color=colors)
ax.set_xticks(range(0, 24, 2))  # x 轴步长为 2
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Swipe Count')
ax.set_title('24-Hour IC Card Swipe Distribution')

# 图例：用 Patch 手动创建（ax.bar 没有 label 参数）
legend_elements = [
    Patch(facecolor='salmon', label='Early Morning (<7) & Late Night (>=22)'),
    Patch(facecolor='steelblue', label='Normal Hours (7-21)')
]
ax.legend(handles=legend_elements, loc='upper right')
ax.yaxis.grid(True, alpha=0.3, linestyle='--')  # 水平网格线

plt.tight_layout()
plt.savefig('hour_distribution.png', dpi=150)
plt.show()
print('已保存: hour_distribution.png')