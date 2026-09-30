"""
该部分是核心，负责
1.从文件读取学生信息（load_students）
2.查询学生（find_student）
3.随机点名（random_roll_call）
4.生成考场安排表（generate_exam_arrangement）
5.生成准考证文件（generate_admission_tickets）
6.显示菜单并控制流程（run）
"""
import os
import random
from student import Student

class ExamSys:

    #初始化，加载学生信息
    def __init__(self, filename):
        self.students = []  # 用列表存储所有 Student 对象
        self.exam_arrangement = []  # 存储考场安排信息
        self.load_students(filename)  # 启动时自动读取学生信息


    def load_students(self,filename):
        # 读取学生信息
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                lines = f.readlines()  # 读取所有行

        # 跳过第一行（表头），从第二行开始解析
            for line in lines[1:]:
                line = line.strip()  # 去掉首尾空白字符（包括换行符）
                if not line:  # 跳过空行
                    continue
            # 按制表符分割每行数据
                parts = line.split('\t')
                seat_no = int(parts[0])  # 序号转为整数
                name = parts[1]
                gender = parts[2]
                class_no = int(parts[3])  # 班级转为整数
                stu_id = parts[4]
                college = parts[5]

                # 创建 Student 对象并添加到列表
                student = Student(seat_no, name, gender, class_no, stu_id, college)
                self.students.append(student)

            print(f"成功加载 {len(self.students)} 名学生信息。")

        except FileNotFoundError:
            # 文件不存在时的友好提示
            print(f"错误：找不到文件 '{filename}'，请确认文件路径是否正确。")
        except Exception as e:
            # 其他异常
            print(f"读取文件时发生错误：{e}")

    def find_student(self):
        stu_id = input("请输入要查询的学号：").strip()

        # 遍历所有学生
        for student in self.students:
            if student.stu_id == stu_id:
                print("\n--- 查询结果 ---")
                print(student)  # 这里会调用 Student 的 __str__ 方法
                print("---------------\n")
                return

        # 如果遍历完都没有找到，说明学号不存在
        print("未找到该学号对应的学生，请检查输入是否正确。\n")

    def random_roll_call(self):
        # 随机点名功能

        try:
            num_input = input("请输入需要点名的学生数量：").strip()
            num = int(num_input)  # 将输入转为整数

            # 检查人数是否小于等于0
            if num <= 0:
                print("点名人数必须大于0，请重新输入。\n")
                return

            # 检查人数是否超过学生总数
            if num > len(self.students):
                print(f"点名人数不能超过学生总数（{len(self.students)}人），请重新输入。\n")
                return

            # 使用 random.sample 随机抽取不重复的学生
            selected = random.sample(self.students, num)

            # 输出结果
            print("\n本次随机点名结果：")
            for i, student in enumerate(selected, 1):
                print(f"{i}.{student.name} {student.stu_id}")
            print()

        except ValueError:
            # 输入的不是数字时触发此异常
            print("输入无效，请输入一个正整数。\n")

    def generate_exam_arrangement(self):
        #生成考场安排表

        # 使用 random.shuffle 原地打乱学生列表顺序
        shuffled = self.students.copy()  # 复制一份，不改变原始顺序
        random.shuffle(shuffled)

        # 保存安排结果，供生成使用
        self.exam_arrangement = shuffled

        # 写入文件
        with open('考场安排表.txt', 'w', encoding='utf-8') as f:
            for i, student in enumerate(shuffled, 1):
                f.write(f"{i},{student.name},{student.stu_id}\n")

        print("考场安排表已生成：考场安排表.txt\n")

    def generate_admission_tickets(self):
        # 生成准考证文件

        # 检查是否已生成考场安排
        if not self.exam_arrangement:
            print("请先执行功能3（生成考场安排表），再生成准考证。\n")
            return

        folder_name = "准考证"
        os.makedirs(folder_name, exist_ok=True)
        # exist_ok=True 表示：如果文件夹已存在，不抛出异常

        # 遍历考场安排，生成准考证文件
        for i, student in enumerate(self.exam_arrangement, 1):
            # 文件名格式：01.txt, 02.txt, 03.txt ...
            # 使用 zfill(2) 将数字补零为两位数
            filename = os.path.join(folder_name, str(i).zfill(2) + ".txt")

            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"考场座位号:{i}\n")
                f.write(f"姓名:{student.name}\n")
                f.write(f"学号:{student.stu_id}\n")

        print(f"准考证文件已生成，共 {len(self.exam_arrangement)} 份，"
              f"保存在 '{folder_name}' 文件夹中。\n")

    def run(self):
        # 主运行函数，循环显示菜单，调用功能

        while True:
            # 打印功能菜单
            print("===== 学生信息与考场管理系统 =====")
            print("1. 查询学生信息")
            print("2. 随机点名")
            print("3. 生成考场安排表")
            print("4. 生成准考证文件")
            print("+--------------------------------------------------------------------------")
            print("0. 退出系统")

            choice = input("请输入功能编号：").strip()

            # 根据用户输入调用对应功能
            if choice == '1':
                self.find_student()
            elif choice == '2':
                self.random_roll_call()
            elif choice == '3':
                self.generate_exam_arrangement()
            elif choice == '4':
                self.generate_admission_tickets()
            elif choice == '0':
                print("感谢使用，再见！")
                break  # 退出 while 循环，程序结束
            else:
                # 输入的功能编号不在 0~4 范围内
                print("功能编号不存在，请正确输入功能编号（0~4）：\n")