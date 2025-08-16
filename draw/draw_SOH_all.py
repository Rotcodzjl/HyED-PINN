import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid", palette="husl", font_scale=1.2)
def smooth_data(sequence, window_size):
    """数据平滑"""
    if window_size < 1:
        raise ValueError("窗口大小必须大于等于1")
    # 初始化平滑后的数据列表
    smoothed_sequence = []
    # 计算窗口内的平均值
    for i in range(len(sequence)):
        # 计算窗口的起始和结束索引
        start_index = max(0, i - window_size + 1)
        end_index = i + 1
        # 计算窗口内的数据平均值
        window_average = sum(sequence[start_index:end_index]) / (end_index - start_index)
        # 将平均值添加到平滑后的数据列表中
        smoothed_sequence.append(window_average)
    return smoothed_sequence
def drop_outlier(array,count,bins):
    """离群值提取--用3sigma方法"""
    index = []
    range_n = np.arange(1,count,bins)
    for i in range_n[:-1]:
        array_lim = array[i:i+bins]
        sigma = np.std(array_lim)
        mean = np.mean(array_lim)
        th_max,th_min = mean + sigma*2, mean - sigma*2
        idx = np.where((array_lim < th_max) & (array_lim > th_min))
        idx = idx[0] + i
        index.extend(list(idx))
    return np.array(index)
def clean_data(array_figs,array_labels):
    index_keep=drop_outlier(array_labels,len(array_labels),35)
    array_figs,array_labels=array_figs[index_keep],array_labels[index_keep]
    array_figs,array_labels=array_figs[drop_outlier(array_labels,len(array_labels),10)],array_labels[drop_outlier(array_labels,len(array_labels),10)]
    return array_figs,array_labels
def plot_battery_soh(soh_data):
    """
    绘制电池SOH随循环次数变化的图表，支持每组数据包含多条SOH序列。

    参数:
    soh_data (dict): 每批次电池的SOH数据序列，键为批次名称，值为包含多条SOH序列的列表。
    """
    # 定义每批次电池对应的颜色
    colors = {
        'CACLE dataset': '#DE66C2',
        'Oxford dataset':'#DE6E66',
        'HUST dataset': '#DEA13A',
        'XJTU batch-1': '#61DE45',
        'XJTU batch-2': '#5096DE',
        'XJTU batch-3': '#CBDE3A'
    }
    marks ={
        'CACLE dataset': 'o',
        'Oxford dataset': 'v',
        'HUST dataset': 's',
        'XJTU batch-1': '^',
        'XJTU batch-2': 'p',
        'XJTU batch-3': 'D'
    }
    # 创建图表
    plt.figure(figsize=(10, 6))

    # 绘制每批次电池的SOH数据
    for batch, soh_sequences in soh_data.items():
        for i, soh in enumerate(soh_sequences):
            print(i)
            cycles = range(len(soh))  # 横坐标为SOH数据序列的序号
            plt.plot(cycles, soh, color=colors.get(batch, 'black'),lw=1,label=f'{batch}' if i == 0 else "")

    # 设置图表标题和坐标轴标签
    # plt.title('Battery SOH vs Cycle Count')
    plt.xlabel('Cycle')
    plt.ylabel('SOH')

    # 设置横轴和纵轴范围
    plt.xlim(0, 2600)
    plt.ylim(0.75, 1)

    # 添加网格线
    plt.grid(True, linestyle='--', alpha=0.6)

    # 添加图例
    plt.legend()

    # 显示图表
    plt.show()

def read_datas(path):
    files=os.listdir(path)
    data=[]
    for i,file in enumerate(files):
        if i<=10:
            file_name=os.path.join(path, file)
            arrays = np.load(file_name)
            _, SOHs =clean_data(arrays['array1'], arrays['array2'])
            a=SOHs.tolist()
            data.append(a)
    return data
def read_datas_xjtu(path):
    files=os.listdir(path)
    data=[]
    for i, file in enumerate(files):
        if i <= 7:
            file_name=os.path.join(path, file)
            arrays = np.load(file_name)
            _, SOHs =arrays['array1'], arrays['array2']/2
            a=SOHs.tolist()
            data.append(a)
    return data
# 数据
soh_data = {
    'CACLE dataset': read_datas('../data/CACLE_data/pinn_path'),
    'Oxford dataset': read_datas('../data/OXFORD_data/pinn_path'),

    'HUST dataset':read_datas('../data/HUST_data/pinn_path'),
    'XJTU batch-1':read_datas_xjtu('../data/XJTU_data/Batch-1/all'),
    'XJTU batch-2':read_datas_xjtu('../data/XJTU_data/Batch-2/all') ,
    'XJTU batch-3':read_datas_xjtu('../data/XJTU_data/Batch-3/all')
}

# 调用函数绘制图表
plot_battery_soh(soh_data)
def read_datas(path):
    files=os.listdir(path)
    data=[]
    for file in files:
        file_name=os.path.join(path, file)
        arrays = np.load(file_name)
        _, SOHs =arrays['array1'], arrays['array2']
        a=SOHs.tolist()
        print(a)

        data.append(a)
    return data

print(len(read_datas('../data/CACLE_data/pinn_path')))