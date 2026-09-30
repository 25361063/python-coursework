# src/m4_qa_system.py

import os
import re
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# 尝试导入大模型库（如果没有安装openai，则跳过）
try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class QASystem:
    """
    出租车数据问答系统
    支持规则匹配 + 大模型增强（小米 MiMo）
    """

    def __init__(self, data_path="outputs/cleaned_data.parquet",
                 model_path="outputs/models/",
                 shapefile_path="data/taxi_zones.shp"):
        """
        初始化问答系统，加载清洗数据、模型、大模型客户端
        """
        self.data_path = data_path
        self.model_path = model_path
        self.shapefile_path = shapefile_path

        # 加载清洗后的数据
        if os.path.exists(data_path):
            self.df = pd.read_parquet(data_path)
            print(f"✅ 加载数据成功: {len(self.df)} 条记录")
        else:
            print("⚠️ 清洗数据不存在，请先运行 M1 生成数据。")
            self.df = None

        # 加载训练好的模型（如果存在）
        self.nn_model = None
        self.rf_model = None
        self.scaler = None
        self.feature_cols = None

        # 尝试加载模型
        nn_path = os.path.join(model_path, "neural_network.h5")
        rf_path = os.path.join(model_path, "random_forest.pkl")
        scaler_path = os.path.join(model_path, "scaler.pkl")
        if os.path.exists(nn_path) and os.path.exists(rf_path):
            try:
                import tensorflow as tf
                import joblib
                self.nn_model = tf.keras.models.load_model(nn_path)
                self.rf_model = joblib.load(rf_path)
                self.scaler = joblib.load(scaler_path)
                print("✅ 加载预测模型成功")
            except Exception as e:
                print(f"⚠️ 加载模型失败: {e}")
        else:
            print("⚠️ 预测模型未找到，需求预测功能可能不可用。")

        # ----- 初始化小米 MiMo 大模型客户端（从环境变量读取 API Key） -----
        self.llm_client = None
        self.llm_model = None
        api_key = os.environ.get("MIMO_API_KEY")
        if api_key and OPENAI_AVAILABLE:
            try:
                self.llm_client = OpenAI(
                    api_key=api_key,
                    base_url="https://api.xiaomimimo.com/v1"  # 小米 MiMo 固定 Base URL
                )
                self.llm_model = os.environ.get("MIMO_MODEL", "xiaomi/mimo-v2.5-pro")
                print("✅ 小米 MiMo 大模型 API 已初始化")
            except Exception as e:
                print(f"⚠️ 初始化小米 MiMo API 失败: {e}")
        elif not OPENAI_AVAILABLE:
            print("⚠️ openai 库未安装，大模型功能不可用。请执行: pip install openai")
        else:
            print("⚠️ 未设置环境变量 MIMO_API_KEY，大模型功能不可用。")

        # 预计算一些统计数据
        if self.df is not None:
            self.total_orders = len(self.df)
            self.avg_fare = self.df['fare_amount'].mean()
        else:
            self.total_orders = 0
            self.avg_fare = 0

    # ==================== 意图识别（规则匹配） ====================
    def parse_intent(self, query):
        """
        基于规则和关键词匹配识别用户意图
        返回意图类型和提取的参数（字典）
        """
        query_lower = query.lower().strip()
        params = {}

        # 1. 时段查询
        pattern_time = r'(\d{4}-\d{2}-\d{2}).*?(\d{1,2})\s*点'
        match = re.search(pattern_time, query_lower)
        if match:
            date_str = match.group(1)
            hour = int(match.group(2))
            try:
                datetime.strptime(date_str, '%Y-%m-%d')
                params['date'] = date_str
                params['hour'] = hour
                return 'time_query', params
            except:
                pass
        if ('订单' in query or '多少' in query) and ('点' in query or '小时' in query):
            hour_match = re.search(r'(\d{1,2})\s*点', query_lower)
            if hour_match:
                params['hour'] = int(hour_match.group(1))
                date_match = re.search(r'(\d{4}-\d{2}-\d{2})', query_lower)
                if date_match:
                    params['date'] = date_match.group(1)
                return 'time_query', params

        # 2. 区域排名
        if '热门' in query or 'top' in query_lower or '排名' in query:
            if '上车' in query or '出发' in query:
                params['type'] = 'pickup'
            elif '下车' in query or '到达' in query:
                params['type'] = 'dropoff'
            else:
                params['type'] = 'both'
            num_match = re.search(r'top\s*(\d+)|前\s*(\d+)', query_lower)
            if num_match:
                params['top_n'] = int(num_match.group(1) or num_match.group(2))
            else:
                params['top_n'] = 10
            return 'ranking', params

        # 3. 需求预测
        if '预测' in query and ('订单' in query or '需求' in query):
            area_match = re.search(r'区域\s*(\d+)', query_lower)
            if area_match:
                params['area'] = int(area_match.group(1))
            else:
                params['area'] = None
            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', query_lower)
            if date_match:
                params['date'] = date_match.group(1)
            else:
                params['date'] = None
            hour_match = re.search(r'(\d{1,2})\s*点', query_lower)
            if hour_match:
                params['hour'] = int(hour_match.group(1))
            else:
                params['hour'] = None
            if params['area'] is not None and params['date'] is not None and params['hour'] is not None:
                return 'predict', params
            else:
                return 'predict', params  # 缺少信息也返回预测意图，后面会提示

        # 4. 费用估算
        if '费用' in query or '车费' in query or '多少钱' in query:
            pattern_from_to = r'从\s*区域\s*(\d+)\s*到\s*区域\s*(\d+)'
            match = re.search(pattern_from_to, query_lower)
            if match:
                params['from_area'] = int(match.group(1))
                params['to_area'] = int(match.group(2))
                return 'fare_estimate', params
            else:
                nums = re.findall(r'\d+', query_lower)
                if len(nums) >= 2:
                    params['from_area'] = int(nums[0])
                    params['to_area'] = int(nums[1])
                    return 'fare_estimate', params
                else:
                    return 'fare_estimate', {'from_area': None, 'to_area': None}

        # 5. 数据质量
        if '缺失' in query or '质量' in query or '报告' in query:
            return 'quality', {}

        # 6. 总体统计
        if '总订单' in query or '平均车费' in query or '统计' in query:
            return 'statistics', {}

        # 默认：无法识别
        return 'unknown', {}

    # ==================== 各意图回答函数 ====================
    def answer_time_query(self, params):
        """时段查询：返回某日某小时的订单量"""
        date = params.get('date')
        hour = params.get('hour')
        if date is None or hour is None:
            return "请提供具体的日期和时间（例如：2026-01-15 8点）", None, None

        df = self.df
        df['pickup_date'] = pd.to_datetime(df['tpep_pickup_datetime']).dt.date
        mask = (df['pickup_date'] == datetime.strptime(date, '%Y-%m-%d').date()) & (df['pickup_hour'] == hour)
        orders = df[mask]
        count = len(orders)
        if count == 0:
            text = f"在 {date} {hour}:00 没有找到订单记录。"
        else:
            text = f"在 {date} {hour}:00 共有 {count} 笔订单。"
        return text, count, None

    def answer_ranking(self, params):
        """区域排名：返回热门区域"""
        top_n = params.get('top_n', 10)
        region_type = params.get('type', 'both')
        df = self.df
        results = {}
        if region_type in ['pickup', 'both']:
            top_pickup = df['PULocationID'].value_counts().head(top_n).reset_index()
            top_pickup.columns = ['区域ID', '订单数']
            results['上车'] = top_pickup
        if region_type in ['dropoff', 'both']:
            top_dropoff = df['DOLocationID'].value_counts().head(top_n).reset_index()
            top_dropoff.columns = ['区域ID', '订单数']
            results['下车'] = top_dropoff

        text_lines = []
        for key, df_top in results.items():
            text_lines.append(f"热门{key}区域 TOP{top_n}:")
            for i, row in df_top.iterrows():
                text_lines.append(f"  {i + 1}. 区域 {row['区域ID']} - {row['订单数']} 订单")
        text = "\n".join(text_lines)
        chart_path = "outputs/m2_2_region_top10.png" if os.path.exists("outputs/m2_2_region_top10.png") else None
        return text, results, chart_path

    def answer_predict(self, params):
        """需求预测：使用已训练的模型"""
        if self.nn_model is None:
            return "预测模型未加载，请先运行 M3 训练模型。", None, None

        area = params.get('area')
        date = params.get('date')
        hour = params.get('hour')
        if area is None or date is None or hour is None:
            return "请提供完整信息：区域ID、日期、小时（例如：预测区域100在2026-01-15 8点）", None, None

        dt = datetime.strptime(date, '%Y-%m-%d')
        dayofweek = dt.weekday()
        is_weekend = 1 if dayofweek >= 5 else 0
        is_rush = 1 if (dayofweek < 5 and ((7 <= hour < 9) or (17 <= hour < 19))) else 0

        # 使用全局平均滞后（简化）
        df = self.df
        lag_1 = df['lag_1'].mean() if 'lag_1' in df.columns else 5
        lag_2 = df['lag_2'].mean() if 'lag_2' in df.columns else 5
        lag_3 = df['lag_3'].mean() if 'lag_3' in df.columns else 5
        rolling_mean_3 = df['rolling_mean_3'].mean() if 'rolling_mean_3' in df.columns else 5

        features = np.array([[hour, dayofweek, is_weekend, is_rush, lag_1, lag_2, lag_3, rolling_mean_3]])
        features_scaled = self.scaler.transform(features)
        pred = self.nn_model.predict(features_scaled, verbose=0)[0][0]
        pred = max(0, pred)
        text = f"预测区域 {area} 在 {date} {hour}:00 的订单量为: {pred:.1f} 笔"
        return text, pred, None

    def answer_fare_estimate(self, params):
        """费用估算：根据两区域历史平均车费"""
        from_area = params.get('from_area')
        to_area = params.get('to_area')
        if from_area is None or to_area is None:
            return "请提供出发区域和到达区域（例如：从区域100到区域200的费用）", None, None

        df = self.df
        mask = (df['PULocationID'] == from_area) & (df['DOLocationID'] == to_area)
        trips = df[mask]
        if len(trips) == 0:
            text = f"没有找到从区域 {from_area} 到区域 {to_area} 的行程记录，无法估算费用。"
            return text, None, None
        avg_fare = trips['fare_amount'].mean()
        median_fare = trips['fare_amount'].median()
        text = f"从区域 {from_area} 到区域 {to_area} 的平均车费为 ${avg_fare:.2f}，中位车费为 ${median_fare:.2f}（基于 {len(trips)} 笔行程）"
        return text, {'avg': avg_fare, 'median': median_fare, 'count': len(trips)}, None

    def answer_quality(self, params):
        """数据质量报告"""
        report_path = "outputs/data_quality_report.csv"
        if os.path.exists(report_path):
            report_df = pd.read_csv(report_path)
            text = "数据质量报告摘要：\n"
            for _, row in report_df.iterrows():
                col = row['column']
                missing = row['missing_rate'] * 100
                abnormal = row.get('abnormal_rate', 0) * 100
                text += f"  {col}: 缺失率 {missing:.2f}%, 异常率 {abnormal:.2f}%\n"
            return text, report_df, report_path
        else:
            return "质量报告文件未找到，请先运行 M1 生成。", None, None

    def answer_statistics(self, params):
        """总体统计"""
        df = self.df
        total = len(df)
        avg_fare = df['fare_amount'].mean()
        avg_dist = df['trip_distance'].mean()
        avg_dur = df['trip_duration_min'].mean()
        text = f"数据总体统计：\n  总订单数: {total:,}\n  平均车费: ${avg_fare:.2f}\n  平均行程距离: {avg_dist:.2f} 英里\n  平均行程时长: {avg_dur:.1f} 分钟"
        return text, {'total': total, 'avg_fare': avg_fare, 'avg_dist': avg_dist, 'avg_dur': avg_dur}, None

    # ==================== 大模型增强（Unknown 处理） ====================
    def answer_unknown(self, query):
        """无法识别意图时，调用小米 MiMo 大模型生成回复，或返回默认提示"""
        if self.llm_client:
            try:
                response = self.llm_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {"role": "system", "content": self._get_system_prompt()},
                        {"role": "user", "content": query}
                    ],
                    temperature=0.7,
                    max_tokens=500
                )
                reply = response.choices[0].message.content
                return reply, None, None
            except Exception as e:
                print(f"大模型调用失败: {e}")
                return self._default_unknown_response(), None, None
        else:
            return self._default_unknown_response(), None, None

    def _get_system_prompt(self):
        """系统提示词（可根据实际需求迭代优化）"""
        return """
你是一个纽约出租车出行数据分析助手。用户可能会询问以下信息：
- 某天某个小时的订单量（例如：2026-01-15 8点有多少订单？）
- 热门上下车区域排名（例如：热门上车区域TOP10）
- 预测某个区域在某个时间的订单量（例如：预测区域100在2026-01-15 8点的订单量）
- 估算两个区域之间的费用（例如：从区域100到区域200的费用）
- 数据质量报告
- 总体统计信息（总订单数、平均车费等）

如果用户的问题与上述主题相关但表述不够清晰，请友好地引导用户提供更具体的信息。
如果问题完全不相关，请礼貌地告知您只能回答出租车数据相关的问题。
回答要简洁、准确，不要编造数据，可以建议用户尝试系统支持的问题类型。
"""

    def _default_unknown_response(self):
        """默认的无法识别回复"""
        return "抱歉，我无法理解您的问题。请尝试以下类型的问题：\n" \
               "  - 时段查询：'2026-01-15 8点有多少订单？'\n" \
               "  - 区域排名：'热门上车区域TOP10'\n" \
               "  - 需求预测：'预测区域100在2026-01-15 8点的订单量'\n" \
               "  - 费用估算：'从区域100到区域200的费用'\n" \
               "  - 数据质量：'数据质量报告'\n" \
               "  - 总体统计：'总订单数和平均车费'"

    # ==================== 对外接口 ====================
    def process_query(self, query):
        """
        处理用户输入，返回（文本答案，数据对象，图表路径）
        """
        intent, params = self.parse_intent(query)
        # 调试时打印意图识别结果
        # print(f"[意图识别] {intent}  参数: {params}")

        if intent == 'time_query':
            return self.answer_time_query(params)
        elif intent == 'ranking':
            return self.answer_ranking(params)
        elif intent == 'predict':
            return self.answer_predict(params)
        elif intent == 'fare_estimate':
            return self.answer_fare_estimate(params)
        elif intent == 'quality':
            return self.answer_quality(params)
        elif intent == 'statistics':
            return self.answer_statistics(params)
        else:
            return self.answer_unknown(query)

    def run_interactive(self):
        """启动命令行交互循环"""
        print("\n" + "=" * 60)
        print("🚕 纽约出租车数据问答系统")
        print("=" * 60)
        print("输入您的问题，或输入 'exit' 退出。")
        print("支持的问题类型示例：")
        print("  - 2026-01-15 8点有多少订单？")
        print("  - 热门上车区域TOP10")
        print("  - 预测区域100在2026-01-15 8点的订单量")
        print("  - 从区域100到区域200的费用")
        print("  - 数据质量报告")
        print("  - 总订单数和平均车费")
        if self.llm_client:
            print("  - 其他问题（将由大模型智能回答）")
        print("-" * 60)

        while True:
            try:
                query = input("\n您的问题: ").strip()
                if query.lower() in ['exit', 'quit', '退出']:
                    print("感谢使用，再见！")
                    break
                if not query:
                    continue

                text, data, chart_path = self.process_query(query)
                print("\n" + "=" * 60)
                print("📌 回答:")
                print(text)
                if chart_path:
                    print(f"📊 相关图表: {chart_path}")
                print("=" * 60)

            except KeyboardInterrupt:
                print("\n\n退出程序。")
                break
            except Exception as e:
                print(f"❌ 处理问题时出错: {e}")


# ==================== 外部调用接口 ====================
def start_qa_system(data_path="outputs/cleaned_data.parquet",
                    model_dir="outputs/models/"):
    """
    启动问答系统的入口函数
    """
    qa = QASystem(data_path=data_path, model_path=model_dir)
    qa.run_interactive()


if __name__ == "__main__":
    # 独立测试
    start_qa_system()