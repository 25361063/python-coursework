"""
任务3：线路站点分析
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -*- coding: utf-8 -*-
"""
任务3：线路站点分析（20分）
要求：
  - 定义 analyze_route_stops 函数（签名不可修改）
  - 调用函数并打印前10行
  - 用 seaborn 水平条形图可视化前15条线路
  - 误差棒显示标准差，capsize=0.3
  - 保存为 route_stops.png（dpi=150）
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

# ===== 取前15条线路（均值最高的15条），再按线路号从小到大排序 =====
top15 = route_stats.head(15).sort_values('线路号', ascending=True).reset_index(drop=True)

# ===== seaborn 水平条形图 =====
fig, ax = plt.subplots(figsize=(10, 8))

# 用 order 参数显式指定 y 轴顺序：线路号从小到大（从上到下）
route_order = top15['线路号'].tolist()
sns.barplot(data=top15, y='线路号', x='mean_stops', palette='Blues_d',
            orient='h', ax=ax, order=route_order)

# 线路号转为字符串标签
ax.set_yticklabels([str(int(v)) for v in route_order])

# ===== 误差棒 =====
ax.errorbar(
    top15['mean_stops'].values,
    range(len(top15)),
    xerr=top15['std_stops'].values,
    fmt='none',
    capsize=0.3,
    color='black',
    linewidth=1.5
)

# ===== 标题和标签（英文） =====
ax.set_xlabel('Mean Ride Stops')
ax.set_ylabel('Route Number')
ax.set_title('Top 15 Routes by Average Ride Stops')

# ===== x 轴从 0 开始 =====
ax.set_xlim(left=0)

# ===== 保存并显示 =====
plt.tight_layout()
plt.savefig('route_stops.png', dpi=150)
plt.show()
print('已保存: route_stops.png')
