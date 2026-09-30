"""
第三次作业 - 公交 IC 卡刷卡数据分析
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.patches import Patch

# ============================================================
# 任务 1：数据预处理
# ============================================================
# ...（task1.py 的代码）
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

# ============================================================
# 任务 2(a)：早晚时段刷卡量统计
# ============================================================
# ...（task2a.py 的代码）
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

# ============================================================
# 任务 2(b)：24小时刷卡量分布可视化
# ============================================================
# ...（task2b.py 的代码）
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

# ============================================================
# 任务 3：线路站点分析
# ============================================================
# ===== 数据预处理 =====
df = pd.read_csv('ICData.csv')
df['交易时间'] = pd.to_datetime(df['交易时间'])
df['hour'] = df['交易时间'].dt.hour
df['ride_stops'] = abs(df['下车站点'] - df['上车站点'])
df = df[df['ride_stops'] != 0].reset_index(drop=True)
filtered = df[df['刷卡类型'] == 0]


# ===== 函数定义（签名和 docstring 不可修改） =====
def analyze_route_stops(df, route_col='线路号', stops_col='ride_stops'):
    """
    计算各线路乘客的平均搭乘站点数及其标准差。
    Parameters
    ----------
    df : pd.DataFrame  预处理后的数据集
    route_col : str    线路号列名
    stops_col : str    搭乘站点数列名
    Returns
    -------
    pd.DataFrame  包含列：线路号、mean_stops、std_stops，按 mean_stops 降序排列
    """
    # 按线路号分组，计算搭乘站点数的均值和标准差
    result = df.groupby(route_col)[stops_col].agg(['mean', 'std']).reset_index()
    # 重命名为题目要求的列名
    result.columns = [route_col, 'mean_stops', 'std_stops']
    # 按均值降序排列
    result = result.sort_values('mean_stops', ascending=False).reset_index(drop=True)
    return result


# ===== 调用函数并打印前10行 =====
route_stats = analyze_route_stops(filtered)
print('===== 各线路平均搭乘站点数（前10） =====')
print(route_stats.head(10))

# ===== 取前15条线路 =====
top15 = route_stats.head(15)

# ===== seaborn 水平条形图 =====
fig, ax = plt.subplots(figsize=(10, 8))

# 用 order 参数显式指定 y 轴顺序（与 top15 的线路号顺序一致）
# 这样 set_yticklabels 的标签一定能和柱子对应上
route_order = top15['线路号'].tolist()
sns.barplot(data=top15, y='线路号', x='mean_stops', palette='Blues_d',
            orient='h', ax=ax, order=route_order)

# 线路号转为字符串标签
ax.set_yticklabels([str(int(v)) for v in route_order])

# ===== 误差棒（用 matplotlib 辅助实现） =====
ax.errorbar(
    top15['mean_stops'].values,    # x 位置（均值）
    range(len(top15)),              # y 位置（0, 1, 2, ...）
    xerr=top15['std_stops'].values,# x 方向误差（标准差）
    fmt='none',                     # 不画数据点
    capsize=0.3,                    # 端线长度
    color='black',
    linewidth=1.5
)

ax.set_xlabel('Mean Ride Stops')
ax.set_ylabel('Route Number')
ax.set_title('Top 15 Routes by Average Ride Stops')
ax.set_xlim(left=0)  # x 轴从 0 开始

plt.tight_layout()
plt.savefig('route_stops.png', dpi=150)
plt.show()
print('已保存: route_stops.png')

# 调用 + 可视化 ...

# ============================================================
# 任务 4：高峰小时系数 PHF
# ============================================================
# ...（task4.py 的代码）
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

# ============================================================
# 任务 5：线路驾驶员信息批量导出
# ============================================================
# ...（task5.py 的代码）
# ===== 数据预处理 =====
df = pd.read_csv('ICData.csv')
df['交易时间'] = pd.to_datetime(df['交易时间'])
df['hour'] = df['交易时间'].dt.hour
df['ride_stops'] = abs(df['下车站点'] - df['上车站点'])
df = df[df['ride_stops'] != 0].reset_index(drop=True)

# ===== 创建文件夹 =====
os.makedirs('线路驾驶员信息', exist_ok=True)

# ===== 遍历 20 条线路 =====
for route in range(1101, 1121):
    route_data = df[df['线路号'] == route]
    if route_data.empty:
        print(f'线路 {route} 无数据，跳过')
        continue
    # 提取车辆编号和驾驶员编号，去重
    pairs = route_data[['车辆编号', '驾驶员编号']].drop_duplicates()
    pairs = pairs.sort_values('车辆编号')

    filepath = f'线路驾驶员信息/{route}.txt'
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f'线路号: {route}\n')
        for _, row in pairs.iterrows():
            # int() 将 float64 转为整数，避免写入 1599.0
            f.write(f'{row["车辆编号"]}\t{int(row["驾驶员编号"])}\n')
    print(f'已生成: {filepath}（{len(pairs)} 组）')

print(f'\n全部 20 个文件生成完毕')

# ============================================================
# 任务 6：服务绩效热力图
# ============================================================
# ...（task6.py 的代码）
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

# ===== 结论分析 =====
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