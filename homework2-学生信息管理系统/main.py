# 程序入口文件，负责创建 ExamSys 对象并启动系统

from exam_system import ExamSys

def main():
    # 创建 ExamSys 对象，传入学生名单文件名
    system = ExamSys("人工智能编程语言学生名单.txt")

    # 启动系统菜单
    system.run()

# Python 的惯用写法，确保只有直接运行此文件时才执行 main()
if __name__ == "__main__":
    main()