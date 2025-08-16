import time  # 导入time模块，用于在循环中模拟耗时操作
import sys  # 导入sys模块，用于操作与Python解释器交互的一些变量和函数


# 定义一个函数，用于显示自定义形状的进度条
def custom_shape_progress_bar(total, progress,name='加载'):
    bar_length = 50  # 设置进度条的总长度为50个字符
    filled_length = int(round(bar_length * progress / float(total)))  # 计算已完成的进度条长度，并四舍五入为整数
    percents = round(100.0 * progress / float(total), 1)  # 计算进度百分比，保留一位小数
    bar = '▋' * filled_length + ' ' * (bar_length - filled_length)  # 构造进度条字符串，使用'▋'表示已完成部分，空格表示未完成部分
    sys.stdout.write(f'\r[{bar}] {percents}% '+name+'进度')  # 使用sys.stdout.write打印进度条，\r表示回到当前行的开头，这样进度条会在同一行更新
    sys.stdout.flush()  # 强制将缓冲区的内容输出到标准输出设备，确保进度条即时更新


# 模拟进度
total = 100  # 设置总进度为100
for i in range(total + 1):  # 循环从0到100（包括100），共101次迭代，以模拟进度从0%到100%
    custom_shape_progress_bar(total, i)  # 调用自定义进度条函数，传入总进度和当前进度
    time.sleep(0.1)  # 暂停0.1秒，模拟耗时操作
print("\n完成!")  # 循环结束后，打印"完成!"，并换行


