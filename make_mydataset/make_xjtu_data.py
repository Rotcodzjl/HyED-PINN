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
import scipy
from datetime import datetime, timedelta
from scipy.signal import medfilt

"""锂电池组特征"""
Batch_list = ['Batch-1', 'Batch-2', 'Batch-3', 'Batch-4', 'Batch-5', 'Batch-6']


def choose_batch(Batch_name):
    Battery_list = []
    dir = '../source_data/XJTU_data/' + Batch_name
    for name in os.listdir(dir):
        Battery_list.append(name.strip('.mat'))
    source_data_path = '../source_data/XJTU_data/' + Batch_name + '/'
    target_data_path = '../data/XJTU_data/' + Batch_name + '/'
    return Battery_list, source_data_path, target_data_path


Battery_list, source_data_path, target_data_path = choose_batch(Batch_list[0])
print(Battery_list)
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


def subtract_datetimes(datetime_str1, datetime_str2, format='%Y-%m-%d,%H:%M:%S'):
    """
    计算两个日期时间字符串的差值（datetime_str2 - datetime_str1）。

    参数:
        datetime_str1 (str): 第一个日期时间字符串。
        datetime_str2 (str): 第二个日期时间字符串。
        format (str): 日期时间字符串的格式，默认为 '%Y-%m-%d,%H:%M:%S'。

    返回:
        timedelta: 两个日期时间的差值。
    """
    # 将字符串转换为 datetime 对象
    datetime1 = datetime.strptime(datetime_str1, format)
    datetime2 = datetime.strptime(datetime_str2, format)

    # 计算差值
    time_difference = datetime2 - datetime1
    return time_difference.total_seconds()

def get_origin_data(batch_name,dV_threshold=0.000001,window_size=20):
    Battery_list,source_data_path,target_data_path=choose_batch(batch_name)
    data_batch=[]
    for Battery_name in Battery_list:
        print(Battery_name)
        source_path=os.path.join(source_data_path,Battery_name)
        data=scipy.io.loadmat(source_path)
        data_out=[]
        data_all=data['data'][0]
        cap_0=1
        for i in range(60,len(data_all)):
            v_all=data_all[i][2]
            time_all=data_all[i][0]
            times,vs,cs,qs=[],[],[],[]
            c_all=data_all[i][4]
            i_all=data_all[i][3]
            for j in range(len(time_all)):
                times.append(time_all[j][0][0])
                vs.append(v_all[j][0])
                cs.append(i_all[j][0])
                qs.append(c_all[j][0])
            df=pd.DataFrame({'v':vs,'i':cs,'c':qs,'t':times})
            df_cccv=df[df['i']>0]
            df_cc=df_cccv[df_cccv['i']>3.999]
            df_cv=df_cccv[df_cccv['i']<3.999]
            """cvt 和 cct"""
            cvt=subtract_datetimes(df_cv['t'].iloc[0],df_cv['t'].iloc[-1])
            cct=subtract_datetimes(df_cc['t'].iloc[0],df_cc['t'].iloc[-1])
            """锂电池部分电流和电压"""
            v_stop=df_cc['v'].iloc[-1]
            v_begin=v_stop-0.2
            V_cc=df_cc[(df_cc['v']<=v_stop+0.1) & (df_cc['v']>=v_begin)]['v'].tolist()
            I_cv=df_cv[(df_cv['i']<=1) & (df_cv['i']>=0.5)]['i'].tolist()
            """cccv图"""
            C=df_cc['c'].tolist()
            V=df_cc['v'].tolist()
            dC = np.diff(C)  # 在末尾和开头添加0
            dV = np.diff(V)  # 在末尾和开头添加0
            zero_index = np.where(dV <= dV_threshold)
            dC = np.delete(dC, zero_index)
            dV = np.delete(dV, zero_index)
            V = np.delete(V, zero_index)
            dcdv = dC / dV
            weights = np.ones(window_size) / window_size
            dcdv = np.convolve(dcdv, weights, mode='same')
            """提取SOH数据"""
            if i==0:
                cap_0=max(df['c'])
            cap=max(df['c'])
            SOH=cap/cap_0
            data_out.append([I_cv,V_cc,V[1:],dcdv,cct,cvt,SOH])
        data_batch.append(data_out)
        break
    return data_batch
data_batch=get_origin_data(Batch_list[0],dV_threshold=0.000001,window_size=20)
data=data_batch[0]
import seaborn as sns
plt.figure(figsize=(10, 10))
palette = sns.color_palette("viridis", n_colors=len(data))  # 使用 viridis 渐变色
for i in range(len(data)):
    if i>=0:
        V = data[i][2]
        dcdv = data[i][3]
        sns.lineplot(x=V, y=dcdv, color=palette[i], alpha=0.5)  # 使用渐变色，设置透明度
plt.xticks(fontsize=40)
plt.yticks(fontsize=50)
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(5))  # X轴5个刻度
plt.gca().yaxis.set_major_locator(plt.MaxNLocator(5))  # Y轴5个刻度
plt.xlabel('Voltage (V)', fontsize=60)
plt.ylabel('dQ/dV (Ah/V)', fontsize=60)
plt.show()