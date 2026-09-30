# 陈浚豪-25361063-第二次人工智能编程作业

仓库链接:https://gitee.com/vyt666/pythonhomework2

## 1. 任务拆解与 AI 协作策略

在编写代码前，我将这个大任务拆解为以下几个步骤与 AI 协作完成：

步骤1：先让 AI 帮我理解作业要求，梳理出需要实现的功能列表和类的设计思路，了解项目总体思路。

步骤2：让 AI 生成定义 Student 类代码，这部分内容相对独立。

步骤3：在 Student 类完成的基础上，一步步让 AI 生成 ExamSys 类的核心功能，进行测试和debug，根据报错信息尝试修改。

步骤4：了解每段代码的功能，进行异常处理，增加中文注释。

步骤5：最后整合测试。

## 2. 核心 Prompt 迭代记录

**初代 Prompt：**
"写一个学生信息管理系统函数，实现从 txt 文件读取学生信息"

**AI 生成的问题/缺陷：**
- 使用了没学过的第三方库（如 pandas）
- 没有异常处理

**优化后的 Prompt (追问)：**
"只能使用 os 和 random 标准库。需要对用户输入进行异常处理，包括文件不存在、输入非数字、人数越界等情况。"

**说明：** 通过逐步增加约束条件（只用标准库、异常处理），AI 生成的代码逐渐符合规范。

## 3. Debug 与异常处理记录

**报错类型：** FileNotFoundError

**现象：** 当学生名单文件路径不正确时，程序直接崩溃报错。

**解决过程：** 我将 Traceback 错误信息发给 AI，AI 建议在 load_students 函数中使用 try-except 捕获 FileNotFoundError，并给出友好的提示信息。修改后，即使文件不存在，程序也不会崩溃，而是提示用户检查文件路径。

**最终修改：** 在文件读取代码外层包裹了 try-except 块。

## 4. 人工代码审查 (Code Review)

以下是 AI 生成的随机点名功能核心代码，我逐行添加了中文注释：

```python
def random_roll_call(self):
    """随机点名功能"""
    try:
        # 获取用户输入并去除首尾空格
        num_input = input("请输入需要点名的学生数量：").strip()
        # 将字符串转换为整数，如果输入非数字会触发 ValueError
        num = int(num_input)

        # 检查输入人数是否合法（必须大于0）
        if num <= 0:
            print("点名人数必须大于0")
            return  # 提前结束函数

        # 检查输入人数是否超过学生总数
        if num > len(self.students):
            print("点名人数不能超过学生总数")
            return

        # 使用 random.sample 从学生列表中随机抽取不重复的学生
        selected = random.sample(self.students, num)

        # 遍历选中的学生，打印序号、姓名和学号
        for i, student in enumerate(selected, 1):
            print(f"{i}.{student.name} {student.stu_id}")

    except ValueError:
        # 当用户输入的不是数字时，捕获异常并提示
        print("输入无效，请输入一个正整数。")
