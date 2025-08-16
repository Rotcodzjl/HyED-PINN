import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from scipy import signal

def smooth_sequence(data, method='moving_avg', window_size=10, alpha=0.3):
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

def min_max_normalize(arr):
    """
    将数组映射到 [0, 1] 之间。

    Parameters:
    arr (numpy.ndarray): 输入数组

    Returns:
    numpy.ndarray: 归一化后的数组
    """
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    if arr_max == arr_min:  # 避免除零错误
        return np.zeros_like(arr)
    return (arr - arr_min) / (arr_max - arr_min)
"""----------------cacle-------------------------------"""
cacle_data = np.load("../data/CACLE_data/all_path/CS2_35.npz")
cacle_data2 = np.load("../data/CACLE_data/all_path/CS2_36.npz")
cacle_data3 = np.load("../data/CACLE_data/all_path/CS2_37.npz")
cacle_data4 = np.load("../data/CACLE_data/all_path/CS2_38.npz")

# D:\pyworks\bysj\new\data\CACLE_data\all_path\CS2_35.npz
feature_cacle = cacle_data["array1"]
soh_cacle = cacle_data["array2"]
feature_cacle2 = cacle_data2["array1"]
soh_cacle2 = cacle_data2["array2"]
feature_cacle3 = cacle_data3["array1"]
soh_cacle3 = cacle_data3["array2"]
feature_cacle4 = cacle_data4["array1"]
soh_cacle4 = cacle_data4["array2"]

feature_cacle=np.concatenate((feature_cacle,feature_cacle2,feature_cacle3,feature_cacle4),axis=0)
soh_cacle=np.concatenate((soh_cacle,soh_cacle2,soh_cacle3,soh_cacle4),axis=0)

feature_cacle[:,1]*=-1
feature_cacle[:,2]*=-1
"""-----------------------------ox-----------------------------------------------"""
file_paths = [
    "../data/OXFORD_data/all_path/Cell1.npz",
    "../data/OXFORD_data/all_path/Cell3.npz",
    "../data/OXFORD_data/all_path/Cell7.npz",
    "../data/OXFORD_data/all_path/Cell8.npz"
]
# 初始化列表来存储加载的数据
feature_cacles = []
soh_cacles = []

# 循环遍历文件路径，加载并合并数据
for path in file_paths:
    data = np.load(path)
    feature_cacles.append(data["array1"])
    soh_cacles.append(data["array2"])

feature_ox = np.concatenate(feature_cacles, axis=0)
feature_ox=np.concatenate([feature_ox,feature_ox],axis=0)
soh_ox = np.concatenate(soh_cacles, axis=0)
soh_ox=np.concatenate([soh_ox,soh_ox],axis=0)
feature_ox[:,1]*=-1
feature_ox[:,9]*=-1
"---------------------------------hust--------------------------------------"
file_paths = [
    # "D:/pyworks/bysj/new/data/HUST_data/all_path/1-5.npz",
    # "D:/pyworks/bysj/new/data/HUST_data/all_path/1-2.npz",
    "../data/HUST_data/all_path/1-3.npz",
    # "D:/pyworks/bysj/new/data/HUST_data/all_path/1-4.npz",
    # "D:/pyworks/bysj/new/data/HUST_data/all_path/1-5.npz",
    # "D:/pyworks/bysj/new/data/HUST_data/all_path/1-6.npz",
    # "D:/pyworks/bysj/new/data/HUST_data/all_path/1-7.npz",
    # "D:/pyworks/bysj/new/data/HUST_data/all_path/1-8.npz",
]
# 初始化列表来存储加载的数据
feature_cacles = []
soh_cacles = []
# 循环遍历文件路径，加载并合并数据
for path in file_paths:
    data = np.load(path)
    feature_cacles.append(data["array1"])
    soh_cacles.append(data["array2"])
# 将列表中的数组沿指定轴连接
feature_hust = np.concatenate(feature_cacles, axis=0)
soh_hust = np.concatenate(soh_cacles, axis=0)
feature_hust[:,4]*=-1
feature_hust[:,9]*=-1
"----------------------------------xjtu-------------------------------------"

# hust_data = np.load("D:/pyworks/bysj/new/draw/HUST.npz")
# feature_hust = hust_data["arr1"]
# soh_hust = min_max_normalize(hust_data["arr2"])
file_paths = [
    "../data/XJTU_data/Batch-1/all/2C_battery-1.npz",
    "../XJTU_data/Batch-1/all/2C_battery-2.npz",
    "../XJTU_data/Batch-1/all/2C_battery-3.npz",
    "../XJTU_data/Batch-1/all/2C_battery-4.npz",
    "../data/XJTU_data/Batch-1/all/2C_battery-5.npz",
    "../data/XJTU_data/Batch-1/all/2C_battery-6.npz",
]

# 初始化列表来存储加载的数据
feature_cacles = []
soh_cacles = []

# 循环遍历文件路径，加载并合并数据
for path in file_paths:
    data = np.load(path)
    feature_cacles.append(data["array1"])
    soh_cacles.append(data["array2"])

# 将列表中的数组沿指定轴连接
feature_xjtu = np.concatenate(feature_cacles, axis=0)
# feature_xjtu[:,0]*=-1
soh_xjtu = np.concatenate(soh_cacles, axis=0)

print(soh_xjtu)
def add_random_noise(arr):
    """
    对输入的 NumPy 数组的每个元素添加随机噪声，并确保结果仍在 [0, 1] 范围内
    
    参数:
    arr -- 输入的 NumPy 数组
    
    返回:
    添加了随机噪声后的 NumPy 数组
    """
    # 生成随机噪声，范围在 [-0.1, 0.1] 之间
    noise = np.random.uniform(-0.045, 0.045, size=arr.shape)
    
    # 添加噪声
    noisy_arr = arr + noise
    
    # 确保结果在 [0, 1] 范围内
    noisy_arr = np.clip(noisy_arr, 0.0, 1.0)
    
    return noisy_arr
def smooth_and_normalize(df,
                         smooth_method='moving_avg',
                         window_size=5,
                         alpha=0.3,
                         normalize=True):
    processed_df = pd.DataFrame()

    for col in df.columns:
        # 平滑处理
        if smooth_method == 'moving_avg':
            smoothed = df[col].rolling(window=window_size,
                                       min_periods=1,
                                       center=True).mean()
        elif smooth_method == 'savgol':
            smoothed = signal.savgol_filter(df[col],
                                            window_length=window_size,
                                            polyorder=2)
        elif smooth_method == 'exponential':
            smoothed = df[col].ewm(alpha=alpha).mean()
        else:
            raise ValueError("smooth_method必须是'moving_avg', 'savgol'或'exponential'")
        smoothed=df[col]
        # 归一化到[0,1]
        if normalize:
            col_min = smoothed.min()
            col_max = smoothed.max()
            if col_max == col_min:  # 处理全等值列
                normalized = pd.Series(0.5, index=df.index)
            else:
                normalized = (smoothed - col_min) / (col_max - col_min)
            processed_df[col] = normalized
        else:
            processed_df[col] = smoothed
        processed_df=add_random_noise(processed_df)
    return processed_df

from scipy import interpolate
def upsample_curve_spline(x, y, multiplier=2, kind='cubic'):
    """
    使用样条插值增加点数

    参数:
        kind: 插值类型 ('linear', 'cubic', 'quadratic'等)
    """
    f = interpolate.interp1d(x, y, kind=kind)
    new_x = np.linspace(x[0], x[-1], (len(x) - 1) * multiplier + 1)
    new_y = f(new_x)
    return new_x, new_y
def upsample_curve(x, y, multiplier=2):
    if multiplier < 1:
        raise ValueError("倍数必须≥1")

    if len(x) != len(y):
        raise ValueError("x和y长度必须相同")

    if multiplier == 1:
        return x.copy(), y.copy()

    # 计算新数组长度
    n = len(x)
    new_length = (n - 1) * multiplier + 1

    # 生成新的x坐标（线性插值）
    new_x = np.linspace(x[0], x[-1], new_length)

    # 线性插值y坐标
    new_y = np.interp(new_x, x, y)

    return new_x, new_y
def plot_features_vs_soh(features_list, SOH_list):
    """
    绘制16个特征与SOH的相关性散点图矩阵图，每组数据用不同颜色表示。

    Parameters:
    features_list (list of numpy.ndarray): 包含四组特征数组的列表，每组形状为(n, 10)
    SOH_list (list of numpy.ndarray): 包含四组SOH数组的列表，每组形状为(n, 1)
    """
    # 设置Seaborn主题
    sns.set_theme(style="whitegrid", palette="bright")
    plt.rcParams['axes.linewidth'] = 2.0  # 默认是 0.8
    plt.rcParams['axes.edgecolor'] = 'black'  # 边框颜色
    # sns.set_theme(style="bright", palette="deep")
    # 创建2x5的子图布局
    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    axes = axes.ravel()  # 将二维数组转换为一维数组，方便遍历

    # 定义特征名称
    feature_names = ['Voltage Mean', 'Voltage Std', 'Voltage Kurt', 'Voltage Skew', 'Voltage Sampen', 'IC Kurt', 'IC Skew', 'IC PSD', 'CCT', 'CVT']

    # 定义每组数据的颜色
    colors = ['#FADD85', '#F36F43', '#3E398F', '#7FA5E1']
    colors=['']
    # colors = ['#DE66C2', '#5096DE', '#DEA13A', '#61DE45']
    # colors = ['#DE66C2', '#5096DE',  'orange', '#61DE45']
    colors = plt.cm.tab10(np.array([0,1,2,4]))  # 4种不同颜色
    # colors = ['#1F3A93', '#3B6BB0', '#5C9BD5', '#8FC7E8']
    # 初始化存储每组数据的Pearson相关系数
    pearson_list = []
    labels=['CACLE', 'OXFORD', 'HUST','XJTU']
    # 遍历每个特征
    # 初始化存储图例的句柄和标签
    legend_handles = []
    legend_labels = []
    for i in range(10):
        # 遍历每组数据
        for j in range(len(features_list)):

            
            df = pd.DataFrame({
                    'SOH': SOH_list[j].flatten(),
                    feature_names[i]: features_list[j][:, i]
                })

            df=smooth_and_normalize(df)
            # print(df)
            # 使用Seaborn绘制散点图
            scatter=sns.scatterplot(
                x="SOH",
                y=feature_names[i],
                data=df,
                ax=axes[i],
                color=colors[j],  # 每组数据使用不同颜色
                alpha=0.6,
                s=20,
                label=labels[j]  # 添加图例标签
            )
            # # 收集图例的句柄和标签
            # if i == 0:
            #     legend_handles.append(scatter.legend_elements()[0][0])
            #     legend_labels.append(labels[j])
            # 计算Pearson相关系数
            r, _ = pearsonr(df['SOH'], df[feature_names[i]])
            if j == 0:
                pearson_list.append([])
            pearson_list[i].append(r)

        # 设置子图标题和标签
        axes[i].set_title(feature_names[i], fontsize=20)
        axes[i].set_xlabel("SOH", fontsize=20)
        # axes[i].set_ylabel(feature_names[i], fontsize=35)
        axes[i].set_ylabel('', fontsize=20)
        # 设置 x 轴刻度标签在图表内部
        if 5<=i<=9:
            axes[i].set_xticks([0, 0.5, 1])
            axes[i].set_xticklabels([0, 0.5, 1], fontsize=20)
        else:
            axes[i].set_xticks([])
        if i==0 or i==5:
            axes[i].set_yticks([0, 0.5, 1])
            axes[i].set_yticklabels([0, 0.5, 1], fontsize=20)
        else:
            axes[i].set_yticks([])
        # 设置刻度位置
        axes[i].tick_params(axis='both', which='both', direction='in', length=6, width=1, color='black')
        axes[i].grid(False)
        axes[i].legend().set_visible(False)
        # fig.legend(legend_handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=4, fontsize=14)
        # axes[i].legend(loc='upper left')  # 显示图例
        # handles, labels = axes[0, 0].get_legend_handles_labels()  # 获取图例的句柄和标签
        # # fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=2)
    # 调整子图布局
    plt.tight_layout()
    plt.show()
    return pearson_list

# 示例数据
# features_list = [
#     np.random.rand(100, 10),  # 第一组特征
#     np.random.rand(100, 10),  # 第二组特征
#     np.random.rand(100, 10),  # 第三组特征
#     np.random.rand(100, 10)  # 第四组特征
# ]
# SOH_list = [
#     np.random.rand(100, 1),  # 第一组SOH
#     np.random.rand(100, 1),  # 第二组SOH
#     np.random.rand(100, 1),  # 第三组SOH
#     np.random.rand(100, 1)  # 第四组SOH
# ]
features_list=[feature_cacle,feature_ox,feature_hust,feature_xjtu]
SOH_list=[soh_cacle,soh_ox,soh_hust,soh_xjtu]
# 调用函数绘制图形
pearson_list = plot_features_vs_soh(features_list, SOH_list)
print("Pearson相关系数：", pearson_list)