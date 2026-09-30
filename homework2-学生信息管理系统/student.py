# 定义student类

class Student:

# 构造方法
    def __init__(self,seat_no,name,gender,class_no,stu_id,college):
        # 初始化学生对象
        self.seat_no = seat_no
        self.name = name
        self.gender = gender
        self.class_no = class_no
        self.stu_id = stu_id
        self.college = college
        # 传入的参数保存为对象的属性

# 返回字符串表示
    def __str__(self):
        # 返回学生信息字符串表示，方便打印
        return(
            f"序号：{self.seat_no}\n"
            f"姓名：{self.name}\n"
            f"性别：{self.gender}\n"
            f"班级：{self.class_no}\n"
            f"学号：{self.stu_id}\n"
            f"学院：{self.college}"
        )
