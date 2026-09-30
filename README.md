# Python 课程作业合集（人工智能编程语言）

> 作者：陈浚豪　学号：25361063
>
> 本仓库为《人工智能编程语言》课程作业合集，由 Gitee 上的三个作业仓库合并迁移至 GitHub，
> 重新整理了目录结构，并补充了本总览说明。各子目录内保留原作业 README（含任务拆解、
> Prompt 迭代与 Debug 记录）。

## 仓库内容一览

| 目录 | 对应作业 | 主题 | 主要技术栈 |
| --- | --- | --- | --- |
| [homework2-学生信息管理系统](./homework2-学生信息管理系统) | 第二次作业 | 学生信息与考场管理系统 | Python 标准库（os、random） |
| [homework3-公交IC卡刷卡数据分析](./homework3-公交IC卡刷卡数据分析) | 第三次作业 | 公交 IC 卡刷卡数据分析（6 项任务） | pandas、numpy、matplotlib、seaborn |
| [final-project-纽约出租车数据分析系统](./final-project-纽约出租车数据分析系统) | 期末项目 | 数据处理 + 可视化 + 机器学习预测 + 智能问答 | pandas、scikit-learn、TensorFlow/Keras、Gradio、openai SDK |

```
python课程作业/
├── README.md
├── homework2-学生信息管理系统/
├── homework3-公交IC卡刷卡数据分析/
└── final-project-纽约出租车数据分析系统/
```

---

## 1. homework2-学生信息管理系统（第二次作业）

学生信息与考场管理系统，仅使用 Python 标准库实现，包含完整的异常处理与中文注释。

**功能**

- 从 txt 文件读取学生名单（文件不存在、格式错误等异常均有友好提示）
- 按学号查询学生信息
- 随机点名（校验人数范围，随机抽取不重复的学生）
- 随机打乱生成「考场安排表.txt」
- 批量生成准考证（`准考证/01.txt`、`02.txt` ...）

**关键文件**

| 文件 | 说明 |
| --- | --- |
| main.py | 程序入口 |
| student.py | Student 学生类 |
| exam_system.py | ExamSys 核心类（读取 / 查询 / 点名 / 排考场 / 准考证） |
| 人工智能编程语言学生名单.txt | 学生名单数据 |
| 考场安排表.txt、准考证/ | 程序生成的产物示例 |

**运行**

```bash
cd homework2-学生信息管理系统
python main.py
```

---

## 2. homework3-公交IC卡刷卡数据分析（第三次作业）

基于约 20 万行公交 IC 卡刷卡数据的六项分析任务。

**任务清单**

1. 数据预处理：时间解析、构造搭乘站点数 `ride_stops`、删除异常记录、缺失值检查
2. (a) 早晚时段刷卡量统计（numpy 布尔索引）　(b) 24 小时刷卡量分布图（高亮早峰前与深夜）
3. 线路站点分析：各线路平均搭乘站点数 + Top15 水平条形图（seaborn）
4. 高峰小时系数 PHF（5 分钟 / 15 分钟粒度，自动识别高峰小时）
5. 批量导出 20 条线路的「车辆-驾驶员」对应表（`线路驾驶员信息/`）
6. 服务绩效热力图（Top10 驾驶员 / 线路 / 上车站点 / 车辆）

**关键文件**

| 文件 | 说明 |
| --- | --- |
| homework.py | 六项任务合并版（从头到尾一次运行通过） |
| task1.py、task2a.py、task2b.py、task3.py、task4.py、task5.py、task6.py | 各任务独立脚本 |
| ICData.csv | 原始刷卡数据 |
| hour_distribution.png 等 *.png | 生成的可视化图表 |
| 线路驾驶员信息/ | 任务 5 导出的 20 个线路文件 |
| README.md | 作业文档（含 Debug 记录：删除 40,886 条 `ride_stops=0` 异常记录等） |

**运行**

```bash
cd homework3-公交IC卡刷卡数据分析
pip install pandas numpy matplotlib seaborn
python homework.py
```

---

## 3. final-project-纽约出租车数据分析系统（期末项目）

基于 2026 年 1 月纽约市黄色出租车行程数据，构建集数据处理、可视化分析、机器学习预测和智能问答于一体的数据分析系统。

**功能模块**

- **M1 数据处理**：加载 Parquet 原始数据、生成数据质量报告（缺失率 / 异常值率）、清洗（无效费用 / 距离 / 乘客数 / 行程时长 / 支付类型）、特征工程（小时、星期、是否周末、是否高峰时段、行程时长、平均速度）
- **M2 可视化分析**：工作日与周末逐小时订单量对比、Top10 上下车区域、Top5 区域需求热力图、费用影响因素分析、支付方式分布、NYC 出租车区域地理可视化
- **M3 机器学习建模**：构建时间序列面板数据集（滞后特征 + 滚动均值）、训练 Keras 神经网络与随机森林两个模型、按时间顺序 80/20 划分、MAE / RMSE 评估
- **M4 智能问答系统**：正则规则意图识别（时段查询 / 区域排名 / 需求预测 / 费用估算等）+ 小米 MiMo 大模型兜底 + Gradio Web 界面

**关键文件**

| 文件 | 说明 |
| --- | --- |
| main.py | 主入口（完整流程 / `--cli` / `--web`） |
| src/m1_data_processing.py | M1 数据处理 |
| src/m2_visualization.py | M2 可视化分析 |
| src/m3_modeling.py | M3 建模与评估 |
| src/m4_qa_system.py | M4 规则 + 大模型问答 |
| src/m4_web_ui.py | M4 Gradio Web 界面 |
| 人工智能作业.md | 期末作业文档 |
| data/ | NYC 出租车区域 Shapefile 及配套文件 |
| outputs/ | 运行产物（图表、数据质量报告、模型评估指标、小型模型文件） |

**运行**

```bash
cd final-project-纽约出租车数据分析系统
pip install -r requierments.txt
python main.py          # 完整流程（M1→M2→M3），最后启动 Web 界面 http://127.0.0.1:7860
python main.py --cli    # 完整流程后进入命令行问答模式
python main.py --web    # 仅启动 Web 界面（需已有 outputs 产物）
```

**注意事项**

- 大模型 API 密钥已从代码中移除，请通过环境变量配置：`set MIMO_API_KEY=你的密钥`（Windows）或 `export MIMO_API_KEY=你的密钥`（Linux/macOS）。未配置时规则问答仍可用，仅大模型兜底不可用。
- 依赖文件名沿用原仓库拼写 `requierments.txt`。

---

## 关于大文件（未包含在本仓库）

期末项目原仓库（Gitee）中以下大文件因体积超大或超出 GitHub 单文件 100MB 硬限制，未随本仓库迁移：

| 文件 | 大小 | 说明 |
| --- | --- | --- |
| random_forest.pkl | 396 MB | 随机森林模型（超 GitHub 硬限制，无法上传） |
| yellow_tripdata_2026-01.parquet | 64 MB | 原始行程数据 |
| cleaned_data.parquet | 76 MB | 清洗后数据 |

需要时请从 Gitee 原仓库获取，或自行重新运行 `python main.py` 生成（原始数据需放回 `data/` 目录）。

## 迁移说明

- 原 Gitee 仓库：[pythonhomework2](https://gitee.com/vyt666/pythonhomework2)｜[pythonhomework3](https://gitee.com/vyt666/pythonhomework3)｜[finalhomework](https://gitee.com/vyt666/finalhomework)
- 迁移时做了以下调整：三个仓库合并为单一仓库；目录改名为与作业内容对应的名称并去除多余嵌套；移除 finalhomework 的 main.py 中硬编码的 API 密钥；排除上述大文件。