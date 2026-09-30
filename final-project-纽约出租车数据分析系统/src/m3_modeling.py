# src/m3_modeling.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')
import joblib

# 固定随机种子以保证可复现
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


def prepare_data(df):
    """
    从清洗数据构造时间序列面板数据，生成用于预测的特征和目标。

    特征设计思路：
    - 时间特征：小时、星期几、是否周末、是否高峰，用以捕捉出行需求的周期性规律。
    - 滞后特征（lag_1, lag_2, lag_3）：前1、2、3小时该区域的订单量，直接反映近期需求趋势。
    - 移动平均（rolling_mean_3）：过去3小时的平均订单量，平滑短期波动，提供更稳健的历史信息。

    选择这些特征的原因：
    - 出租车需求具有明显的日周期和星期效应，时间特征必须包含。
    - 需求具有强自相关性，滞后特征能有效预测当前值。
    - 移动平均相比原始滞后更稳定，可减少噪声影响。
    - 区域差异通过区域自身的时序历史信息（滞后和平均）间接体现，无需显式编码区域ID，避免高维稀疏问题。

    目标变量：当前小时该区域的订单量（count）。
    """
    print("正在构造样本...")

    # 确保时间列存在
    df['pickup_hour'] = df['pickup_hour'].astype(int)
    # 提取日期用于时间排序（假设日期已包含在数据中，若没有则从datetime提取）
    if 'pickup_date' not in df.columns:
        df['pickup_date'] = pd.to_datetime(df['tpep_pickup_datetime']).dt.date

    # 按区域、日期、小时聚合订单量（填充缺失小时为0）
    # 先创建完整的时间网格（所有区域 × 所有日期 × 所有小时）
    all_regions = df['PULocationID'].unique()
    all_dates = df['pickup_date'].unique()
    all_hours = range(24)

    # 构建完整索引（为了高效，用笛卡尔积再合并）
    idx = pd.MultiIndex.from_product([all_regions, all_dates, all_hours],
                                     names=['PULocationID', 'pickup_date', 'pickup_hour'])
    full_grid = pd.DataFrame(index=idx).reset_index()

    # 统计实际订单量
    grouped = df.groupby(['PULocationID', 'pickup_date', 'pickup_hour']).size().reset_index(name='count')
    full = full_grid.merge(grouped, on=['PULocationID', 'pickup_date', 'pickup_hour'], how='left')
    full['count'] = full['count'].fillna(0).astype(int)

    # 转换日期为datetime以便排序
    full['datetime'] = pd.to_datetime(full['pickup_date'].astype(str) + ' ' + full['pickup_hour'].astype(str) + ':00:00')

    # 按区域和时间排序
    full = full.sort_values(['PULocationID', 'datetime']).reset_index(drop=True)

    # 提取时间特征
    full['hour'] = full['pickup_hour']
    full['dayofweek'] = full['datetime'].dt.dayofweek
    full['is_weekend'] = (full['dayofweek'] >= 5).astype(int)
    # 高峰：工作日 7-9, 17-19
    full['is_rush'] = ((full['dayofweek'] < 5) &
                       ((full['hour'] >= 7) & (full['hour'] < 9) |
                        (full['hour'] >= 17) & (full['hour'] < 19))).astype(int)

    # 按区域分组生成滞后特征
    def add_lag_features(group):
        group = group.sort_values('datetime')
        group['lag_1'] = group['count'].shift(1)
        group['lag_2'] = group['count'].shift(2)
        group['lag_3'] = group['count'].shift(3)
        group['rolling_mean_3'] = group['count'].rolling(window=3, min_periods=1).mean().shift(1)
        return group

    full = full.groupby('PULocationID', group_keys=False).apply(add_lag_features)

    # 删除缺失滞后值的行（前几行）
    full = full.dropna(subset=['lag_1', 'lag_2', 'lag_3', 'rolling_mean_3'])

    # 选择特征列
    feature_cols = ['hour', 'dayofweek', 'is_weekend', 'is_rush',
                    'lag_1', 'lag_2', 'lag_3', 'rolling_mean_3']
    target_col = 'count'

    X = full[feature_cols].values
    y = full[target_col].values

    print(f"样本数: {len(X)}")
    return X, y, feature_cols, full


def train_and_evaluate(X, y, feature_cols, output_dir="outputs"):
    """
    训练神经网络和随机森林，对比评估并保存结果
    """
    os.makedirs(output_dir, exist_ok=True)

    # 按时间顺序划分训练集和测试集（前80%时间，后20%）
    # 由于数据已按时间排序，按样本顺序切分
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"训练集: {len(X_train)} 样本, 测试集: {len(X_test)} 样本")

    # 特征标准化（神经网络需要，随机森林不需要但也做，保证一致性）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ---------- 神经网络 (TensorFlow) ----------
    print("训练神经网络...")
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)  # 回归输出
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # 训练并记录历史
    history = model.fit(
        X_train_scaled, y_train,
        validation_split=0.2,  # 从训练集中再分20%作为验证集
        epochs=50,
        batch_size=64,
        verbose=0
    )

    # 绘制loss曲线
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history.history['loss'], label='训练损失')
    ax.plot(history.history['val_loss'], label='验证损失')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('神经网络训练损失曲线')
    ax.legend()
    ax.grid(True, alpha=0.3)
    loss_path = os.path.join(output_dir, 'm3_neural_network_loss.png')
    plt.savefig(loss_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Loss曲线已保存: {loss_path}")

    # 预测测试集
    y_pred_nn = model.predict(X_test_scaled, verbose=0).flatten()

    # 评估
    mae_nn = mean_absolute_error(y_test, y_pred_nn)
    rmse_nn = np.sqrt(mean_squared_error(y_test, y_pred_nn))

    # ---------- 随机森林 ----------
    print("训练随机森林...")
    rf = RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)
    rf.fit(X_train, y_train)  # 使用未标准化数据（随机森林不受影响）
    y_pred_rf = rf.predict(X_test)

    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

    # 保存指标
    metrics_df = pd.DataFrame({
        'Model': ['Neural Network', 'Random Forest'],
        'MAE': [mae_nn, mae_rf],
        'RMSE': [rmse_nn, rmse_rf]
    })
    metrics_path = os.path.join(output_dir, 'm3_model_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"    指标已保存: {metrics_path}")

    # 输出结果
    print("\n=== 模型性能对比 ===")
    print(metrics_df.to_string(index=False))

    # 在 train_and_evaluate 函数末尾添加
    model_dir = os.path.join(output_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, "neural_network.h5"))
    joblib.dump(rf, os.path.join(model_dir, "random_forest.pkl"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))

    # ===== 代码注释中分析优劣 =====
    # 分析：
    # 1. 神经网络：能够捕捉复杂的非线性关系，并且通过多层结构可以学习高阶特征交互。
    #    在本任务中，特征主要包括时间特征和需求滞后特征，这些特征间可能存在复杂的相互作用（例如高峰时段滞后效应更强），
    #    神经网络理论上能更好拟合。但训练需要更多数据，调参较复杂，且容易过拟合（需正则化）。
    #    从结果看，MAE和RMSE通常略优于或略逊于随机森林，取决于数据量和调参。
    #
    # 2. 随机森林：对异常值不敏感，训练快，不需要特征缩放，且能给出特征重要性。
    #    对于中小型表格数据，随机森林往往表现优秀，且不易过拟合。本任务特征较少，随机森林可能表现稳定。
    #    但随机森林无法外推训练集范围外的特征值（如新的小时/区域），不过在本场景中不存在此问题。
    #
    # 综合：在特征工程合理的情况下，两种方法性能接近。神经网络可能在数据量极大时更有优势，
    # 但随机森林因其简单、解释性强，常被选为基线模型。本任务中，两者均可，建议根据实际部署需求选择。

    # 返回模型和结果（便于后续使用）
    return model, rf, scaler, history, metrics_df


def run_modeling(df, output_dir="outputs"):
    """
    对外接口：从清洗数据开始，执行建模全流程
    """
    X, y, feature_cols, full_df = prepare_data(df)
    model, rf, scaler, history, metrics = train_and_evaluate(X, y, feature_cols, output_dir)
    return model, rf, scaler


if __name__ == "__main__":
    # 独立测试时，读取清洗数据
    data_path = "../outputs/cleaned_data.parquet"
    if not os.path.exists(data_path):
        print("请先运行 M1 生成清洗数据！")
    else:
        df = pd.read_parquet(data_path)
        print(f"加载清洗数据: {len(df)} 条记录")
        run_modeling(df, output_dir="../outputs")