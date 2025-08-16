import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import time  # 导入time模块，用于在循环中模拟耗时操作
import sys  # 导入sys模块，用于操作与Python解释器交互的一些变量和函数
from scipy import stats
import matplotlib
import pickle
import re
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
import EntropyHub as EH
from scipy.signal import welch
from scipy.stats import entropy
from scipy.signal import medfilt

"""锂电池组特征"""
Battery_list = []
for dir in os.listdir('../source_data/HUST_data/'):
    name = dir.split('.')[0]
    Battery_list.append(name)

"""原始数据存放位置"""
source_data_path = '../source_data/HUST_data/'

"""训练数据存放位置"""
target_data_path = '../data/HUST_data/pinn_path/'

"""一个数组的元素减去数组首位的数"""


def subtract_first_element(arr):
    """数组全部减去第一个元素，用来处理时间数据据"""
    if not arr:  # 检查数组是否为空
        return []
    first_element = arr[0]
    return [x - first_element for x in arr]


"""一系列对序列数据进行特征提取的函数"""
"""样本熵"""


def SampleEntropy(Datalist, r, m=2):
    th = r * np.std(Datalist)  # 容限阈值
    return EH.SampEn(Datalist, m, r=th)[0][-1]


"""功率谱密度"""


def find_psd_peak(signal, fs=1.0, nperseg=None, noverlap=None, nfft=None):
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft)

    # 找到功率谱密度的峰值
    peak_index = np.argmax(psd)
    peak_freq = freqs[peak_index]
    peak_psd = psd[peak_index]

    return peak_freq, peak_psd

"""计算序列的熵"""

def calculate_entropy(sequence, base=None):
    value_counts = np.bincount(sequence)
    probabilities = value_counts / len(sequence)
    entropy_value = entropy(probabilities, base=base)
    return entropy_value

"""获取IC曲线的序列数据"""
def get_origin_data(b_name,dV_threshold=0.000001,window_size=20):
    file_path = source_data_path+b_name+'.pkl'
    file_data=pd.read_pickle(file_path)[b_name]['data']
    data_out=[]
    cap_0=0
    for i in range(1,len(file_data)+1):
        """提取序列数据I,V"""
        df=pd.DataFrame(file_data[i])
        df_all=df
        df_cccv=df[df['Status']=='Constant current-constant voltage charge']
        v_list=df_cccv['Voltage (V)'].tolist()
        v_stop=max(v_list)-0.001
        v_begin=v_stop-0.2
        df_cv=df_cccv[df_cccv['Voltage (V)']>=v_stop]
        df_cc=df_cccv[df_cccv['Voltage (V)']<v_stop]
        cct=df_cc['Time (s)'].iloc[-1]-df_cc['Time (s)'].iloc[0]
        cvt=df_cv['Time (s)'].iloc[-1]-df_cv['Time (s)'].iloc[0]
        V_cc=df_cc['Voltage (V)'].tolist()
        I_cv=df_cv['Current (mA)'].tolist()
        """ic曲线"""
        df_ic=df_cc[df_cc['Voltage (V)']<=3.5]
        C=df_ic['Capacity (mAh)'].tolist()
        V=df_ic['Voltage (V)'].tolist()
        dC = np.diff(C)  # 在末尾和开头添加0
        dV = np.diff(V)  # 在末尾和开头添加0
        zero_index = np.where(dV <= dV_threshold)
        dC = np.delete(dC, zero_index)
        dV = np.delete(dV, zero_index)
        V = np.delete(V, zero_index)
        dcdv = dC / dV
        """提取SOH数据"""
        if i==1:
            cap_0=max(df_all['Capacity (mAh)'])
        cap=max(df_all['Capacity (mAh)'])
        SOH=cap/cap_0
        weights = np.ones(window_size) / window_size
        dcdv = np.convolve(dcdv, weights, mode='same')
        if i>10:
            data_out.append([I_cv,V_cc,V[1:],dcdv,cct,cvt,SOH])
    return data_out
data=get_origin_data('1-4')
import seaborn as sns
#
# data, SOHS = dataset.get_IC_charge()
palette = sns.color_palette("viridis", n_colors=len(data))  # 使用 viridis 渐变色
plt.figure(figsize=(10, 10))  # 建议设置合适画布尺寸
for i in range(len(data)):
    if i >= 0:
        V = data[i][2]
        dcdv = data[i][3]/1000
        sns.lineplot(x=V, y=dcdv, color=palette[i], alpha=0.5)  # 使用渐变色，设置透明度
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(5))  # X轴5个刻度
plt.gca().yaxis.set_major_locator(plt.MaxNLocator(5))  # Y轴5个刻度
plt.xticks(fontsize=40)
plt.yticks(fontsize=50)
plt.xlabel('Voltage (V)', fontsize=60)
plt.ylabel('dQ/dV (Ah/V)', fontsize=60)
plt.show()