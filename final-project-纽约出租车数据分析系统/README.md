# 纽约出租车数据分析与预测系统

基于2026年1月纽约市黄色出租车行程数据，构建集数据处理、可视化分析、机器学习预测和智能问答于一体的数据分析系统。

## 项目结构

```
final_project/
├── main.py                        # 主入口
├── requierments.txt               # Python 依赖
├── data/
│   ├── yellow_tripdata_2026-01.parquet   # 原始出租车行程数据
│   └── taxi_zones.*                       # NYC 出租车区域 Shapefile
├── outputs/
│   ├── cleaned_data.parquet               # 清洗后数据集
│   ├── data_quality_report.csv            # 数据质量报告
│   ├── m2_*.png                           # 可视化图表
│   ├── m3_neural_network_loss.png         # 神经网络训练损失曲线
│   ├── m3_model_metrics.csv               # 模型评估指标
│   └── models/
│       ├── neural_network.h5              # Keras 神经网络模型
│       ├── random_forest.pkl              # 随机森林模型
│       └── scaler.pkl                     # 标准化器
└── src/
    ├── m1_data_processing.py    # M1: 数据加载、质量报告、清洗、特征工程
    ├── m2_visualization.py      # M2: 可视化分析
    ├── m3_modeling.py           # M3: 机器学习建模与评估
    ├── m4_qa_system.py          # M4: 规则 + LLM 智能问答系统
    └── m4_web_ui.py             # M4: Gradio Web 界面
```

## 功能模块

### M1 - 数据处理
- 加载 Parquet 格式原始数据
- 生成数据质量报告（缺失率、异常值率）
- 数据清洗：过滤无效费用、距离、乘客数、行程时长、支付类型
- 特征工程：提取小时、星期、是否周末、是否高峰时段，计算行程时长和平均速度

### M2 - 可视化分析
- 工作日与周末逐小时订单量对比
- Top 10 上车/下车区域排名
- Top 5 区域逐小时需求热力图
- 费用因素分析（距离-费用散点图、时段中位费用、乘客数-费用关系）
- 支付方式分布分析
- NYC 出租车区域地理可视化（需安装 geopandas）

### M3 - 机器学习建模
- 构建时间序列面板数据集（含滞后特征和滚动均值）
- 训练两个模型：TensorFlow/Keras 神经网络 和 scikit-learn 随机森林
- 按时间顺序 80/20 划分训练集/测试集
- 使用 MAE 和 RMSE 评估模型性能

### M4 - 智能问答系统
- 基于正则表达式的规则意图识别（时段查询、区域排名、需求预测、费用估算等）
- 未识别意图自动调用小米 MiMo 大模型 API 进行回答
- 提供 Gradio Web 界面，支持图表展示

## 环境要求

- Python >= 3.10
- 主要依赖见 `requierments.txt`

## 安装与运行

### 1. 安装依赖

```bash
cd final_project
pip install -r requierments.txt
```

### 2. 准备数据

将数据文件放置到 `data/` 目录：
- `yellow_tripdata_2026-01.parquet` — 出租车行程数据
- `taxi_zones.shp` 及配套文件（`.dbf`, `.shx`, `.prj`, `.cpg`）— 可选，用于地理可视化

### 3. 运行

**完整流程**（数据处理 → 模型训练 → 启动 Web 界面）：
```bash
python main.py
```

**仅 CLI 模式**（数据处理 → 模型训练 → 终端问答）：
```bash
python main.py --cli
```

**仅启动 Web 界面**（需已有 outputs 产物）：
```bash
python main.py --web
```

启动后访问 `http://127.0.0.1:7860` 使用 Web 问答界面。

## 配置说明

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `MIMO_API_KEY` | 小米 MiMo API 密钥 | 见 `main.py` |
| `MIMO_MODEL` | LLM 模型名称 | `xiaomi/mimo-v2.5-pro` |

可通过环境变量覆盖默认配置：
```bash
export MIMO_API_KEY="your-api-key"
export MIMO_MODEL="xiaomi/mimo-v2.5-pro"
```

## 技术栈

- **数据处理**：pandas, numpy, pyarrow
- **可视化**：matplotlib, seaborn, geopandas
- **机器学习**：scikit-learn, tensorflow/keras
- **智能问答**：openai SDK, gradio
- **模型持久化**：joblib, h5