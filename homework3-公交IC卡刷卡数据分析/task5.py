"""
任务5：线路驾驶员信息批量导出
"""
import pandas as pd
import os

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