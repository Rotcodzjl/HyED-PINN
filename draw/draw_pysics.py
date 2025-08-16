import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D

def min_max_scale(sequence):
    """
    将序列线性缩放到[0, 1]范围

    参数:
        sequence (list/np.array): 输入数据序列

    返回:
        np.array: 缩放后的序列
    """
    arr = np.array(sequence)
    min_val = np.min(arr)
    max_val = np.max(arr)

    # 处理全等序列（避免除以0）
    if min_val == max_val:
        return np.zeros_like(arr)

    scaled = (arr - min_val) / (max_val - min_val)
    return scaled
def smooth_sequence(data, method='moving_avg', window_size=30, alpha=0.3):
    smoothed = np.zeros_like(data)
    if method == 'moving_avg':
        # 移动平均法
        for i in range(len(data)):
            start = max(0, i - window_size // 2)
            end = min(len(data), i + window_size // 2 + 1)
            smoothed[i] = np.mean(data[start:end])

    elif method == 'exponential':
        # 指数平滑法
        smoothed[0] = data[0]  # 初始值
        for i in range(1, len(data)):
            smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i - 1]

    else:
        raise ValueError("Method must be 'moving_avg' or 'exponential'")
    return smoothed

cacle_e=[]
cacle_s=[]
cacle_dirs=os.listdir('../results/cacle/')
for cacle_dir in cacle_dirs:
    dir='../results/cacle/'+cacle_dir
    cacle_e_i,cacle_s_i=min_max_scale(smooth_sequence(np.load(dir)['array1'],window_size=100)),min_max_scale(smooth_sequence(np.load(dir)['array2']))
    cacle_e.append(cacle_e_i)
    cacle_s.append(cacle_s_i)
    break
"""-----------------------------------hust------------------------------------"""
hust_e=[]
hust_s=[]
hust_dirs=os.listdir('../results/hust/')
for hust_dir in hust_dirs:
    dir='../results/hust/'+hust_dir
    hust_e_i,hust_s_i=min_max_scale(smooth_sequence(np.load(dir)['array1'])),min_max_scale(np.load(dir)['array2'])
    hust_e.append(hust_e_i)
    hust_s.append(hust_s_i)
    break
"""-----------------------------------ox----------------------------------------------"""
ox_e=[]
ox_s=[]
ox_dirs=os.listdir('../results/oxford/')
for ox_dir in ox_dirs:
    dir='../results/oxford/'+ox_dir
    ox_e_i,ox_s_i=min_max_scale(np.load(dir)['array1']),min_max_scale(np.load(dir)['array2'])
    ox_e.append(ox_e_i)
    ox_s.append(ox_s_i)
    break
"""-----------------------------------------xjtu-------------------------------------------"""
xjtu_e=[]
xjtu_s=[]
xjtu_dirs=os.listdir('../results/xjtu/')
for xjtu_dir in xjtu_dirs:
    dir='../results/xjtu/'+xjtu_dir
    xjtu_e_i,xjtu_s_i=min_max_scale(smooth_sequence(np.load(dir)['array1'])),min_max_scale(smooth_sequence(np.load(dir)['array2']))
    xjtu_e.append(xjtu_e_i)
    xjtu_s.append(xjtu_s_i)
    break
"""-----------------------------------------xjtu2-------------------------------------------"""
xjtu2_e=[]
xjtu2_s=[]
xjtu2_dirs=os.listdir('../new/results/xjtu2/')
for xjtu2_dir in xjtu2_dirs:
    dir='../new/results/xjtu2/'+xjtu2_dir
    xjtu2_e_i,xjtu2_s_i=min_max_scale(smooth_sequence(np.load(dir)['array1'])),min_max_scale(smooth_sequence(np.load(dir)['array2']))
    xjtu2_e.append(xjtu2_e_i)
    xjtu2_s.append(xjtu2_s_i)
    break
"""-----------------------------------------xjtu3-------------------------------------------"""
xjtu3_e=[]
xjtu3_s=[]
xjtu3_dirs=os.listdir('../new/results/xjtu3/')
for xjtu3_dir in xjtu3_dirs:
    dir='../new/results/xjtu3/'+xjtu3_dir
    xjtu3_e_i,xjtu3_s_i=min_max_scale(smooth_sequence(np.load(dir)['array1'])),min_max_scale(smooth_sequence(np.load(dir)['array2']))
    xjtu3_e.append(xjtu3_e_i)
    xjtu3_s.append(xjtu3_s_i)
    break
# plt.plot(smooth_sequence(min_max_scale(xjtu_e[0])))
# plt.plot(smooth_sequence(min_max_scale(xjtu_e[1])))
# plt.plot(smooth_sequence(min_max_scale(xjtu_e[2])))
# plt.plot(smooth_sequence(min_max_scale(xjtu_e[3])))
# plt.show()


def plot_four_groups_combined_legend(group_data_list, group_names=None):
    """
    绘制四组曲线（每组同色），并合并为一个图例

    参数:
        group_data_list (list): 包含四组曲线的列表，每组是4个一维数组
        group_names (list): 每组名称（可选）
    """
    # 设置默认参数
    if group_names is None:
        group_names = [f'Group {i + 1}' for i in range(5)]

    # 颜色和线型设置
    colors = plt.cm.tab10(np.arange(5))  # 4种不同颜色
    line_styles = ['-', '-', '-', '-']  # 4种线型
    markers = ['^', 's', 'p', 'o','D']
    plt.figure(figsize=(8, 6.5))
    plt.yticks([0, 0.25, 0.5, 0.75, 1],fontsize=14)
    plt.xticks(fontsize=14)
    # 绘制所有曲线（不自动生成图例）
    i=0
    for group_idx, (group_data, color) in enumerate(zip(group_data_list, colors)):
        for curve_idx, curve_data in enumerate(group_data):
            x = np.arange(len(curve_data))
            print(i)
            if i%5 == 0 or i==19:
                plt.plot(x, curve_data,
                         color=color,
                         lw=2,
                         marker=markers[group_idx],
                         markersize=8,
                         markevery=40,
                         linestyle=line_styles[curve_idx],
                         alpha=0.5,
                         label=group_names[group_idx]
                         )
            else:
                plt.plot(x, curve_data,
                         color=color,
                         lw=2,
                         marker=markers[group_idx],
                         markersize=8,
                         markevery=40,
                         linestyle=line_styles[curve_idx],
                         alpha=0.5,
                         )
            i += 1
    # 手动创建合并图例
    legend_elements = []
    # 添加颜色代表的组别
    for color, name in zip(colors, group_names):
        legend_elements.append(
            Line2D([0], [0],
                   color=color,
                   lw=2,
                   label=name)
        )

    # plt.legend(fontsize=16)
    # 标签和网格
    plt.xlabel('Cycle Index',fontsize=20)
    plt.ylabel('Proportion of Remaining Active Material',fontsize=22)
    # plt.title('Evolution of Electrochemical  Field During Cycling',fontsize=24)
    # plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()


# 示例数据（四组，每组4条不同长度的曲线）
group1 = cacle_e
group2 = ox_e  # 第2组（15个点）
group3 = hust_e # 第3组（8个点）
group4 = xjtu_e # 第4组（12个点）
group5 = xjtu2_e  # 第4组（12个点）
group6 = xjtu3_e # 第4组（12个点）
# 调用函数
plot_four_groups_combined_legend([group1,group3,group4,group5,group6],
                                 group_names=['CACLE dataset', 'HUST dataset', 'XJTU batch-1','XJTU batch-2','XJTU batch-3'])