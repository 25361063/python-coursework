"""
任务6：服务绩效排名与热力图
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ===== 数据预处理 =====
df = pd.read_csv('ICData.csv')
df['交易时间'] = pd.to_datetime(df['交易时间'])
df['hour'] = df['交易时间'].dt.hour
df['ride_stops'] = abs(df['下车站点'] - df['上车站点'])
df = df[df['ride_stops'] != 0].reset_index(drop=True)
filtered = df[df['刷卡类型'] == 0]

# ===== 统计各维度 Top 10 =====
top_drivers = filtered.groupby('驾驶员编号').size().nlargest(10)
top_routes = filtered.groupby('线路号').size().nlargest(10)
top_stations = filtered.groupby('上车站点').size().nlargest(10)
top_vehicles = filtered.groupby('车辆编号').size().nlargest(10)

print('===== Top 10 司机 =====')
print(top_drivers)
print('\n===== Top 10 线路 =====')
print(top_routes)
print('\n===== Top 10 上车站点 =====')
print(top_stations)
print('\n===== Top 10 车辆 =====')
print(top_vehicles)

# ===== 构造热力图矩阵 (4×10) =====
data_matrix = np.array([
    top_drivers.values,
    top_routes.values,
    top_stations.values,
    top_vehicles.values
])

row_labels = ['Driver', 'Route', 'Boarding Station', 'Vehicle']
col_labels = [f'Top{i+1}' for i in range(10)]

# ===== 画热力图 =====
fig, ax = plt.subplots(figsize=(12, 5))
sns.heatmap(
    data_matrix,
    annot=True,           # 每格标注数值
    fmt='d',              # 整数格式
    cmap='YlOrRd',        # 黄→橙→红渐变
    xticklabels=col_labels,
    yticklabels=row_labels,
    ax=ax
)
ax.set_title('Service Performance Heatmap: Top 10 by Passenger Trips')
ax.set_xlabel('Ranking')
ax.set_ylabel('Dimension')
ax.set_xticklabels(col_labels, rotation=0)  # x 轴标签水平显示

plt.tight_layout()
plt.savefig('performance_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print('已保存: performance_heatmap.png')

# ===== 结论分析（不少于50字） =====
print('\n===== 结论分析 =====')
analysis = """
从热力图中可以观察到以下服务绩效规律：

1. 头部集中效应显著：Top1 线路（线路46003）的客流量高达 7127 人次，
   远超 Top2 的 4893 人次，说明该线路是城市核心干线。

2. 驾驶员维度存在异常数据：Top1 驾驶员编号为 0，服务 6484 人次，
   这可能是数据录入时的默认值或缺失值标记，并非真实驾驶员。

3. 上车站点分布相对均匀：Top1 站点的 7776 人次与 Top10 的 5707 人次
   差距约 36%，说明城市公交站点的客流分布较为均衡。

4. 车辆维度同样存在异常：Top1 车辆编号为 0，服务 11774 人次，
   可能是数据质量问题，其余正常车辆的服务人次在 400~600 之间。

5. 线路维度的二八效应最为明显：Top1 线路的客流量是 Top10 的 2.2 倍，
   说明公交客流高度集中于少数主干线路。
"""
print(analysis)