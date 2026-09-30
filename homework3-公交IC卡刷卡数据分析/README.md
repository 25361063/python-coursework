# 陈浚豪-25361063-第三次人工智能编程作业

仓库链接: https://gitee.com/vyt666/pythonhomework3

---

## 1. 任务拆解与 AI 协作策略

本次作业共 6 项任务，我采用了**"先探索数据，再逐任务拆解"**的策略：

1. **数据探索阶段**：先让 AI 读取 ICData.csv，了解数据规模（20 万行、10 列）、字段类型、缺失值情况、异常值分布等。这一步帮助我建立了对数据的整体认知，为后续任务打下基础。

2. **逐任务拆解**：按照任务 1→2(a)→2(b)→3→4→5→6 的顺序，每完成一个任务就运行验证，确认结果正确后再进入下一个。这样做的好处是：
   - 每步都能验证中间结果，避免错误累积
   - 可以针对每个任务的具体要求精准调整 Prompt
   - 遇到问题时更容易定位原因

3. **代码审查与修正**：AI 生成的代码进行阅读，重点检查：
   - 是否使用了指定的库（如任务 2 必须用 numpy）
   - 函数签名是否与题目要求完全一致
   - 可视化图表的标签、颜色、标题是否符合规范

4. **最终合并**：将 6 个独立的 task 文件合并为一个完整的 homework.py，确保从头到尾可以一次运行通过。

---

## 2. 核心 Prompt 迭代记录

### 迭代：任务 3 seaborn 柱状图 y 轴标签错位

**初代 Prompt**：
> 用 seaborn 画水平条形图，y 轴是线路号，x 轴是平均搭乘站点数，然后用 set_yticklabels 把线路号转成字符串。

**AI 生成的问题**：
AI 直接写了：
```python
sns.barplot(data=top15, y='线路号', x='mean_stops', palette='Blues_d', orient='h', ax=ax)
ax.set_yticklabels([str(int(v)) for v in top15['线路号']])
```
这里的问题是：seaborn 的 barplot 会自动对分类变量排序（按线路号数值大小），但 `set_yticklabels` 是按 `top15` 的行顺序（mean_stops 降序）赋值的。两者顺序不一致时，标签就会和柱子错位。

**优化后的 Prompt**：
> 给 sns.barplot 加 order 参数，显式指定 y 轴顺序与 top15 的线路号顺序一致，再用 set_yticklabels 转字符串。

**修正后的代码**：
```python
route_order = top15['线路号'].tolist()
sns.barplot(data=top15, y='线路号', x='mean_stops', palette='Blues_d',
            orient='h', ax=ax, order=route_order)
ax.set_yticklabels([str(int(v)) for v in route_order])
```

加上 `order=route_order` 后，seaborn 按我们指定的顺序排列柱子，标签就不会错位了。

---

## 3. Debug 记录

### 问题：ride_stops=0 导致平均搭乘站点数异常偏低

**报错现象**：
在任务 3 中，调用 `analyze_route_stops` 计算各线路的平均搭乘站点数时，代码给出的结果发现大部分线路的 mean_stops 只有 5~10，远低于预期。部分线路甚至只有 2~3 站。

**排查过程**：
1. 先检查了 `ride_stops` 的计算逻辑：`abs(df['下车站点'] - df['上车站点'])`，公式正确。
2. 统计了 `ride_stops` 的分布，发现有 40,886 条记录的 `ride_stops = 0`。
3. 进一步分析发现，这些记录的上车站点和下车站点完全相同，属于异常数据（可能是乘客刷卡后未乘车、系统重复记录等）。

**解决过程**：
让AI在任务 1 数据预处理阶段增加删除逻辑：
```python
before_count = len(df)
df = df[df['ride_stops'] != 0]
deleted_count = before_count - len(df)
print(f'删除 ride_stops=0 的异常记录: {deleted_count} 行')
```
删除 40,886 条异常记录后，各线路的平均搭乘站点数恢复正常。

**经验总结**：
数据预处理阶段的异常值检查非常重要。如果跳过这一步，后续所有统计结果都会被大量 0 值拉低，导致分析结论失真。

---

## 4. 人工代码审查（逐行中文注释）

以下是任务 4 高峰小时系数 PHF 计算的核心代码，附逐行中文注释：

```python
# ===== 步骤1：统计全天各小时刷卡量，自动识别高峰小时 =====
# 只保留刷卡类型=0 的正常刷卡记录，copy() 避免后续修改时触发 SettingWithCopyWarning
filtered = df[df['刷卡类型'] == 0].copy()

# 按小时分组计数，得到每小时的刷卡量
# hourly 是一个 Series，index 是 0~23，value 是该小时的刷卡量
hourly = filtered.groupby('hour').size()

# idxmax() 返回最大值对应的索引，即刷卡量最大的小时编号（如 9）
# max() 返回最大值，即该小时的刷卡量（如 14961）
peak_hour = hourly.idxmax()
peak_count = hourly.max()

# ===== 步骤2：提取高峰小时的所有记录 =====
# 筛选出 hour 等于高峰小时的所有记录
peak_data = filtered[filtered['hour'] == peak_hour].copy()

# 提取分钟和秒，计算从整点开始的偏移分钟数（带小数）
# 例如 09:15:30 → minute_offset = 15 + 30/60 = 15.5
# 这样可以用整除运算将记录精确分配到不同的时间窗口
peak_data['minute'] = peak_data['交易时间'].dt.minute
peak_data['second'] = peak_data['交易时间'].dt.second
peak_data['minute_offset'] = peak_data['minute'] + peak_data['second'] / 60

# ===== 步骤3：5分钟粒度统计 =====
# // 是整除运算符，minute_offset // 5 得到窗口编号（0~11）
# 窗口0: 00:00~04:59, 窗口1: 05:00~09:59, ..., 窗口11: 55:00~59:59
peak_data['bin5'] = (peak_data['minute_offset'] // 5).astype(int)

# 边界处理：如果 minute_offset 恰好等于 60（理论上不会，但防御性编程）
# bin5 会等于 12，需要归入最后一个窗口（编号 11）
peak_data.loc[peak_data['bin5'] >= 12, 'bin5'] = 11

# 按 bin5 分组计数，得到每个 5 分钟窗口的刷卡量
counts5 = peak_data.groupby('bin5').size()

# idxmax() 找到刷卡量最大的窗口编号，max() 得到该窗口的刷卡量
max5_bin = counts5.idxmax()
max5_count = counts5.max()

# 计算该窗口的起止时间：窗口编号 × 5 = 起始分钟
start5_min = max5_bin * 5
end5_min = start5_min + 5

# PHF5 = 高峰小时刷卡量 / (12 × 最大5分钟刷卡量)
# 1小时 = 12个5分钟窗口；PHF越接近1表示客流越均匀
PHF5 = peak_count / (12 * max5_count)

# ===== 步骤4：15分钟粒度统计 =====
# 逻辑与5分钟完全相同，只是窗口大小变为15分钟
# // 15 得到窗口编号（0~3）
# 窗口0: 00:00~14:59, 窗口1: 15:00~29:59, 窗口2: 30:00~44:59, 窗口3: 45:00~59:59
peak_data['bin15'] = (peak_data['minute_offset'] // 15).astype(int)

# 边界处理：bin15=4 归入最后一个窗口（编号 3）
peak_data.loc[peak_data['bin15'] >= 4, 'bin15'] = 3

counts15 = peak_data.groupby('bin15').size()
max15_bin = counts15.idxmax()
max15_count = counts15.max()

start15_min = max15_bin * 15
end15_min = start15_min + 15

# PHF15 = 高峰小时刷卡量 / (4 × 最大15分钟刷卡量)
# 1小时 = 4个15分钟窗口
PHF15 = peak_count / (4 * max15_count)
```
