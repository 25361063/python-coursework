# main.py
import os
# 小米 MiMo 大模型 API 密钥请通过环境变量配置（不要把真实密钥写入代码）：
#   Windows:      set MIMO_API_KEY=你的密钥
#   Linux/macOS:  export MIMO_API_KEY=你的密钥
import sys
import warnings
warnings.filterwarnings('ignore')

# 导入四个模块
from src.m1_data_processing import process_data
from src.m2_visualization import visualize_all
from src.m3_modeling import run_modeling
from src.m4_qa_system import start_qa_system

# 尝试导入 Web UI（如果 gradio 未安装，则不可用）
try:
    from src.m4_web_ui import launch_ui
    GRADIO_AVAILABLE = True
except ImportError:
    launch_ui = None
    GRADIO_AVAILABLE = False
    print("⚠️ Gradio 未安装，Web 界面不可用。如需使用，请执行: pip install gradio")


def main():
    """主入口函数"""
    # 解析命令行参数
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg == "--web":
            # 仅启动 Web 界面（跳过 M1-M3，假设已处理完成）
            if not GRADIO_AVAILABLE:
                print("❌ Gradio 未安装，无法启动 Web 界面。")
                return
            print("🚀 启动 Web 可视化界面（跳过数据处理与训练）...")
            launch_ui(share=False)
            return
        elif arg == "--cli":
            # 执行完整流程后进入命令行问答
            print("📟 执行完整流程，然后进入命令行问答模式...")
        else:
            print(f"⚠️ 未知参数: {arg}，将执行默认流程（M1→M2→M3→Web界面）")
    else:
        # 默认：执行完整流程后启动 Web 界面
        print("🚕 执行完整流程（数据处理→可视化→模型训练），然后启动 Web 界面...")

    # 如果是 --cli 或 默认，都需要先运行 M1-M3
    if len(sys.argv) > 1 and sys.argv[1].lower() == "--cli":
        print("="*60)
        print("🚕 纽约出租车数据分析与预测系统（命令行模式）")
        print("="*60)
    else:
        print("="*60)
        print("🚕 纽约出租车数据分析与预测系统（Web 界面模式）")
        print("="*60)

    # ----- M1: 数据处理 -----
    try:
        df = process_data(data_path="data/yellow_tripdata_2026-01.parquet",
                          output_dir="outputs")
        print("✅ M1 数据处理完成\n")
    except FileNotFoundError:
        print("❌ 数据文件不存在，请将 yellow_tripdata_2026-01.parquet 放入 data/ 目录。")
        return

    # ----- M2: 可视化分析 -----
    try:
        visualize_all(df, output_dir="outputs",
                      shapefile_path="data/taxi_zones.shp",
                      enable_map=True)
        print("✅ M2 可视化完成\n")
    except Exception as e:
        print(f"⚠️ M2 可视化部分出错（继续执行）: {e}")

    # ----- M3: 模型训练 -----
    try:
        model_nn, model_rf, scaler = run_modeling(df, output_dir="outputs")
        print("✅ M3 模型训练完成\n")
    except Exception as e:
        print(f"⚠️ M3 模型训练出错（继续执行）: {e}")

    # ----- 根据模式选择后续 -----
    if len(sys.argv) > 1 and sys.argv[1].lower() == "--cli":
        # 命令行问答
        print("="*60)
        print("💬 进入命令行问答系统")
        print("="*60)
        try:
            start_qa_system(data_path="outputs/cleaned_data.parquet",
                            model_dir="outputs/models/")
        except KeyboardInterrupt:
            print("\n用户中断退出。")
        except Exception as e:
            print(f"❌ 问答系统出错: {e}")
    else:
        # 默认启动 Web 界面
        if not GRADIO_AVAILABLE:
            print("⚠️ Gradio 未安装，无法启动 Web 界面。回退到命令行问答模式。")
            start_qa_system(data_path="outputs/cleaned_data.parquet",
                            model_dir="outputs/models/")
        else:
            print("="*60)
            print("🌐 启动 Web 可视化界面...")
            print("请打开浏览器访问: http://127.0.0.1:7860")
            print("按 Ctrl+C 可停止服务")
            print("="*60)
            try:
                launch_ui(share=False)
            except KeyboardInterrupt:
                print("\n用户中断，Web 服务已停止。")
            except Exception as e:
                print(f"❌ Web 界面启动失败: {e}")


if __name__ == "__main__":
    main()