# src/m2_visualization.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import warnings

warnings.filterwarnings('ignore')


# ==================== 设置中文字体 ====================
def setup_chinese_font():
    """配置 matplotlib 中文字体"""
    try:
        # Windows 常用中文字体
        rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
        rcParams['axes.unicode_minus'] = False
    except:
        rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']


# ==================== 分析1：出行需求时间规律 ====================
def plot_demand_time_pattern(df, output_dir):
    """
    绘制工作日/周末分小时订单量对比折线图
    """
    df['hour'] = df['pickup_hour']
    df['is_weekend'] = df['pickup_is_weekend']

    demand_hourly = df.groupby(['hour', 'is_weekend']).size().reset_index(name='order_count')
    weekday = demand_hourly[demand_hourly['is_weekend'] == 0].sort_values('hour')
    weekend = demand_hourly[demand_hourly['is_weekend'] == 1].sort_values('hour')

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(weekday['hour'], weekday['order_count'], marker='o', linewidth=2,
            markersize=6, label='工作日', color='#1f77b4')
    ax.plot(weekend['hour'], weekend['order_count'], marker='s', linewidth=2,
            markersize=6, label='周末', color='#ff7f0e')

    ax.set_xlabel('小时', fontsize=12)
    ax.set_ylabel('订单数量', fontsize=12)
    ax.set_title('工作日 vs 周末分小时订单量对比', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(0, 24))
    ax.set_xticklabels([f'{i}:00' for i in range(24)], rotation=45)

    # 标注高峰时段
    ax.axvspan(7, 9, alpha=0.1, color='green', label='早高峰')
    ax.axvspan(17, 19, alpha=0.1, color='orange', label='晚高峰')

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'm2_1_demand_pattern.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✅ 图表已保存: {out_path}")


# ==================== 分析2：区域热度分析（基础） ====================
def plot_region_hotness(df, output_dir):
    """
    绘制 TOP10 上下车区域柱状图 + 热门区域小时热力图
    """
    # ---- TOP10 上车和下车区域 ----
    top_pickup = df['PULocationID'].value_counts().head(10).reset_index()
    top_pickup.columns = ['LocationID', 'count']
    top_dropoff = df['DOLocationID'].value_counts().head(10).reset_index()
    top_dropoff.columns = ['LocationID', 'count']

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # 上车区域
    ax1 = axes[0]
    bars1 = ax1.barh(range(len(top_pickup)), top_pickup['count'], color='#2ca02c')
    ax1.set_yticks(range(len(top_pickup)))
    ax1.set_yticklabels(top_pickup['LocationID'].astype(str), fontsize=10)
    ax1.set_xlabel('上车订单数量', fontsize=12)
    ax1.set_title('TOP10 热门上车区域', fontsize=13, fontweight='bold')
    for i, (bar, val) in enumerate(zip(bars1, top_pickup['count'])):
        ax1.text(val, bar.get_y() + bar.get_height() / 2, f'{val:,.0f}',
                 ha='left', va='center', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='x')

    # 下车区域
    ax2 = axes[1]
    bars2 = ax2.barh(range(len(top_dropoff)), top_dropoff['count'], color='#d62728')
    ax2.set_yticks(range(len(top_dropoff)))
    ax2.set_yticklabels(top_dropoff['LocationID'].astype(str), fontsize=10)
    ax2.set_xlabel('下车订单数量', fontsize=12)
    ax2.set_title('TOP10 热门下车区域', fontsize=13, fontweight='bold')
    for i, (bar, val) in enumerate(zip(bars2, top_dropoff['count'])):
        ax2.text(val, bar.get_y() + bar.get_height() / 2, f'{val:,.0f}',
                 ha='left', va='center', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'm2_2_region_top10.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✅ 图表已保存: {out_path}")

    # ---- 热力图：TOP5 上车区域的小时订单量 ----
    top5 = df['PULocationID'].value_counts().head(5).index.tolist()
    df_top5 = df[df['PULocationID'].isin(top5)]
    heatmap_data = df_top5.groupby(['PULocationID', 'pickup_hour']).size().unstack(fill_value=0)
    # 按总订单量排序
    region_order = df_top5['PULocationID'].value_counts().index
    heatmap_data = heatmap_data.loc[region_order]

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(heatmap_data.values, cmap='YlOrRd', aspect='auto', interpolation='nearest')
    ax.set_xticks(range(24))
    ax.set_xticklabels([f'{i}:00' for i in range(24)], rotation=45, ha='right')
    ax.set_yticks(range(len(heatmap_data.index)))
    ax.set_yticklabels(heatmap_data.index.astype(str), fontsize=10)
    ax.set_xlabel('小时', fontsize=12)
    ax.set_ylabel('区域ID', fontsize=12)
    ax.set_title('TOP5 热门上车区域的小时订单量热力图', fontsize=13, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('订单数量', fontsize=11)
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'm2_2_heatmap.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✅ 热力图已保存: {out_path}")


# ==================== 分析3：车费影响因素分析 ====================
def plot_fare_factors(df, output_dir):
    """
    绘制距离-车费散点、时段中位车费、乘客人数-车费关系
    """
    fig = plt.figure(figsize=(18, 6))

    # 子图1：距离-车费散点（采样）
    ax1 = fig.add_subplot(1, 3, 1)
    sample = df.sample(n=min(10000, len(df)), random_state=42)
    ax1.scatter(sample['trip_distance'], sample['fare_amount'], alpha=0.3, s=5, c='#1f77b4')
    # 趋势线
    z = np.polyfit(sample['trip_distance'], sample['fare_amount'], 1)
    p = np.poly1d(z)
    x_range = np.linspace(sample['trip_distance'].min(), sample['trip_distance'].max(), 100)
    ax1.plot(x_range, p(x_range), 'r-', linewidth=2, label='趋势线')
    ax1.set_xlabel('行程距离 (英里)', fontsize=12)
    ax1.set_ylabel('车费金额 (美元)', fontsize=12)
    ax1.set_title('行程距离与车费关系', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 子图2：不同时段中位车费
    ax2 = fig.add_subplot(1, 3, 2)
    df['time_period'] = pd.cut(df['pickup_hour'],
                               bins=[0, 6, 12, 18, 24],
                               labels=['凌晨(0-6)', '上午(6-12)', '下午(12-18)', '晚上(18-24)'],
                               right=False)
    median_fare = df.groupby('time_period')['fare_amount'].median().reset_index()
    bars = ax2.bar(range(len(median_fare)), median_fare['fare_amount'],
                   color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax2.set_xticks(range(len(median_fare)))
    ax2.set_xticklabels(median_fare['time_period'], fontsize=10)
    ax2.set_ylabel('中位车费 (美元)', fontsize=12)
    ax2.set_title('不同时段中位车费对比', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, median_fare['fare_amount']):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=10)

    # 子图3：乘客人数与车费
    ax3 = fig.add_subplot(1, 3, 3)
    passenger_stats = df.groupby('passenger_count')['fare_amount'].agg(['mean', 'std']).reset_index()
    x = passenger_stats['passenger_count']
    y = passenger_stats['mean']
    yerr = passenger_stats['std']
    bars = ax3.bar(x, y, yerr=yerr, capsize=5, color='#9467bd', alpha=0.7)
    ax3.set_xlabel('乘客人数', fontsize=12)
    ax3.set_ylabel('平均车费 (美元)', fontsize=12)
    ax3.set_title('乘客人数与平均车费关系', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, y):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'm2_3_fare_factors.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✅ 图表已保存: {out_path}")


# ==================== 分析4：自选分析（支付方式洞察） ====================
def plot_custom_analysis(df, output_dir):
    """
    自选分析：支付方式与时段、区域的关系
    """
    # 支付方式映射
    payment_map = {1: '现金', 2: '信用卡', 3: '借记卡', 4: '其他', 5: '数字支付', 6: '未知'}
    df['payment_name'] = df['payment_type'].map(payment_map)
    df['time_period'] = pd.cut(df['pickup_hour'],
                               bins=[0, 6, 12, 18, 24],
                               labels=['凌晨', '上午', '下午', '晚上'],
                               right=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 子图1：不同时段支付方式占比（堆叠）
    ax1 = axes[0]
    payment_by_period = pd.crosstab(df['time_period'], df['payment_name'], normalize='index')
    payment_by_period.plot(kind='bar', stacked=True, ax=ax1,
                           color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
    ax1.set_xlabel('时段', fontsize=12)
    ax1.set_ylabel('占比', fontsize=12)
    ax1.set_title('不同时段的支付方式分布', fontsize=13, fontweight='bold')
    ax1.legend(title='支付方式', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3, axis='y')

    # 子图2：TOP5 热门区域支付方式偏好
    ax2 = axes[1]
    top5 = df['PULocationID'].value_counts().head(5).index.tolist()
    df_top5 = df[df['PULocationID'].isin(top5)]
    payment_by_region = pd.crosstab(df_top5['PULocationID'], df_top5['payment_name'], normalize='index')
    payment_by_region.plot(kind='bar', stacked=True, ax=ax2,
                           color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
    ax2.set_xlabel('区域ID', fontsize=12)
    ax2.set_ylabel('占比', fontsize=12)
    ax2.set_title('TOP5热门区域的支付方式偏好', fontsize=13, fontweight='bold')
    ax2.legend(title='支付方式', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'm2_4_payment_insight.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✅ 图表已保存: {out_path}")

    # 额外图表：支付方式 vs 平均行程距离
    fig2, ax = plt.subplots(figsize=(12, 6))
    dist_by_payment = df.groupby('payment_name')['trip_distance'].mean().sort_values(ascending=False)
    bars = ax.bar(range(len(dist_by_payment)), dist_by_payment.values,
                  color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    ax.set_xticks(range(len(dist_by_payment)))
    ax.set_xticklabels(dist_by_payment.index, fontsize=10)
    ax.set_ylabel('平均行程距离 (英里)', fontsize=12)
    ax.set_xlabel('支付方式', fontsize=12)
    ax.set_title('不同支付方式的平均行程距离对比', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, dist_by_payment.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    out_path2 = os.path.join(output_dir, 'm2_4_payment_distance.png')
    plt.savefig(out_path2, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✅ 额外图表已保存: {out_path2}")


# ==================== 加分项：geopandas 区域分级设色地图 ====================
def plot_choropleth_map(df, shapefile_path, output_dir):
    """
    使用 geopandas 绘制纽约市区域分级设色地图（加分项）
    如果缺少依赖或 shapefile 不存在，则跳过并给出提示
    """
    try:
        import geopandas as gpd
    except ImportError:
        print("    ⚠️ 未安装 geopandas，跳过地图绘制（加分项）。")
        return

    if not os.path.exists(shapefile_path):
        print(f"    ⚠️ shapefile 不存在: {shapefile_path}，跳过地图绘制。")
        return

    print("    🗺️ 正在生成区域分级设色地图（加分项）...")

    try:
        # 加载 shapefile
        zones = gpd.read_file(shapefile_path)

        # 识别区域ID列
        id_candidates = ['LocationID', 'OBJECTID']
        id_col = None
        for col in id_candidates:
            if col in zones.columns:
                id_col = col
                break
        if id_col is None:
            # 尝试模糊匹配
            for col in zones.columns:
                if 'id' in col.lower():
                    id_col = col
                    break
        if id_col is None:
            print("    ❌ 无法识别 shapefile 中的区域ID列，跳过地图。")
            return

        # 统计各区域上车订单量
        pickup_counts = df['PULocationID'].value_counts().reset_index()
        pickup_counts.columns = ['LocationID', 'pickup_count']

        # 合并
        zones_merged = zones.merge(pickup_counts, left_on=id_col, right_on='LocationID', how='left')
        zones_merged['pickup_count'] = zones_merged['pickup_count'].fillna(0)

        # 绘制
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        zones_merged.plot(column='pickup_count',
                          cmap='YlOrRd',
                          linewidth=0.5,
                          edgecolor='black',
                          ax=ax,
                          legend=True,
                          legend_kwds={'label': '上车订单数量', 'shrink': 0.6},
                          missing_kwds={'color': 'lightgray', 'label': '无数据'})
        ax.set_title('纽约市出租车上车需求区域分布（2026年1月）', fontsize=16, fontweight='bold')
        ax.set_axis_off()
        plt.tight_layout()

        out_path = os.path.join(output_dir, 'm2_2_choropleth_map.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✅ 地图已保存: {out_path}")

    except Exception as e:
        print(f"    ❌ 绘制地图时出错: {e}，跳过。")


# ==================== 主入口函数 ====================
def visualize_all(df, output_dir="outputs", shapefile_path="data/taxi_zones.shp", enable_map=True):
    """
    执行 M2 全部可视化分析

    参数:
        df: 清洗后的 DataFrame
        output_dir: 输出目录
        shapefile_path: taxi_zones.shp 文件路径（用于地图加分项）
        enable_map: 是否启用地图绘制（默认 True）
    """
    os.makedirs(output_dir, exist_ok=True)
    setup_chinese_font()

    print("开始生成 M2 可视化图表...")

    # 1. 出行需求时间规律
    print("  1. 出行需求时间规律")
    plot_demand_time_pattern(df, output_dir)

    # 2. 区域热度分析（基础）
    print("  2. 区域热度分析（基础）")
    plot_region_hotness(df, output_dir)

    # 3. 车费影响因素
    print("  3. 车费影响因素分析")
    plot_fare_factors(df, output_dir)

    # 4. 自选分析
    print("  4. 自选分析（支付方式洞察）")
    plot_custom_analysis(df, output_dir)

    # 加分项：地图可视化
    if enable_map:
        print("  【加分项】区域分级设色地图")
        plot_choropleth_map(df, shapefile_path, output_dir)

    print("M2 所有图表生成完成！")


# ==================== 独立运行测试 ====================
if __name__ == "__main__":
    # 读取由 M1 生成的清洗数据
    data_path = "../outputs/cleaned_data.parquet"
    if not os.path.exists(data_path):
        print("请先运行 M1 生成清洗数据！")
    else:
        df = pd.read_parquet(data_path)
        print(f"加载数据完成: {len(df)} 条记录")
        visualize_all(df, output_dir="../outputs", shapefile_path="../data/taxi_zones.shp")