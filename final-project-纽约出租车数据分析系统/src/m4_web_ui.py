# src/m4_web_ui.py

import os
import gradio as gr
from .m4_qa_system import QASystem  # 导入已有的问答系统

# 全局初始化问答系统（只加载一次）
qa_system = None


def get_qa_system():
    """懒加载问答系统，避免启动时过慢"""
    global qa_system
    if qa_system is None:
        qa_system = QASystem(
            data_path="outputs/cleaned_data.parquet",
            model_path="outputs/models/"
        )
    return qa_system


def answer_question(query):
    """
    Gradio 调用的处理函数
    输入：用户问题字符串
    输出：(文本答案, 图片路径或None)
    """
    if not query or query.strip() == "":
        return "请输入您的问题。", None

    qa = get_qa_system()
    text, data, chart_path = qa.process_query(query)

    # 如果 chart_path 存在且文件可读，返回图片
    if chart_path and os.path.exists(chart_path):
        return text, chart_path
    else:
        return text, None


# 定义示例问题（方便用户一键点击）
examples = [
    "2026-01-15 8点有多少订单？",
    "热门上车区域TOP10",
    "预测区域100在2026-01-15 8点的订单量",
    "从区域100到区域200的费用",
    "数据质量报告",
    "总订单数和平均车费",
]


# 创建 Gradio 界面
def create_interface():
    with gr.Blocks(title="🚕 纽约出租车数据问答系统", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🚕 纽约出租车出行数据智能问答系统
        **基于 2026年1月 黄色出租车公开数据**

        输入您的问题，系统将自动理解并返回答案及相关图表。
        """)

        with gr.Row():
            with gr.Column(scale=4):
                # 输入框
                query_input = gr.Textbox(
                    label="请输入您的问题",
                    placeholder="例如：2026-01-15 8点有多少订单？",
                    lines=2
                )
                submit_btn = gr.Button("提交问题", variant="primary")
                # 示例按钮
                gr.Examples(
                    examples=examples,
                    inputs=query_input,
                    label="💡 试试这些问题"
                )
            with gr.Column(scale=6):
                # 输出：文本答案
                answer_output = gr.Textbox(
                    label="📌 回答",
                    lines=10,
                    interactive=False
                )
                # 输出：图片（如果存在）
                image_output = gr.Image(
                    label="📊 相关图表",
                    type="filepath",
                    visible=True
                )

        # 绑定事件
        submit_btn.click(
            fn=answer_question,
            inputs=query_input,
            outputs=[answer_output, image_output]
        )
        query_input.submit(
            fn=answer_question,
            inputs=query_input,
            outputs=[answer_output, image_output]
        )

        gr.Markdown("""
        ---
        **支持的问题类型**：
        - **时段查询**：`2026-01-15 8点有多少订单？`
        - **区域排名**：`热门上车区域TOP10`
        - **需求预测**：`预测区域100在2026-01-15 8点的订单量`
        - **费用估算**：`从区域100到区域200的费用`
        - **数据质量**：`数据质量报告`
        - **总体统计**：`总订单数和平均车费`
        """)

    return demo


# 启动函数
def launch_ui(share=False):
    demo = create_interface()
    demo.launch(share=share, server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    launch_ui(share=False)