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
import scipy.io
from d2l import torch as d2l
import seaborn as sns
from math import exp
d2l.use_svg_display()

class OXFORD_data(object):
    def __init__(self,target_data_path='../data/OXFORD_data', file_name='Cell1'):
        """待处理的问价你路径"""
        self.file_name = file_name
        self.target_data_path = target_data_path
        self.source_data_path = '../source_data/OXFORD_data/Oxford_Battery_Degradation_Dataset_1.mat'
        self.csv_path = self.target_data_path + 'csv_path'
        self.pkl_path = self.target_data_path + 'npz_path'
    def get_IC_charge(self,dV_threshold=0.0001,window_size=300):
        file_path = self.source_data_path
        data_all = scipy.io.loadmat(file_path)[self.file_name][0][0]
        cap0=0
        data_out,SOH=[],[]
        for i_all in tqdm(range(len(data_all))):
            cycle=i_all+1
            data_i=data_all[i_all][0][0][0]
            V=np.array(data_i['v'][0][0]).reshape(-1)
            t=data_i['t'][0][0].reshape(-1)
            C=data_i['q'][0][0].reshape(-1)
            dC = np.diff(C)  # 在末尾和开头添加0
            dV = np.diff(V)  # 在末尾和开头添加0
            zero_index = np.where(dV <= dV_threshold)
            dC = np.delete(dC, zero_index)
            dV = np.delete(dV, zero_index)
            V = np.delete(V, zero_index)
            dcdv = dC / dV
            weights = np.ones(window_size) / window_size
            dcdv = np.convolve(dcdv, weights, mode='same')
            capacity=C[-1]
            if i_all==0:
                cap0=capacity
            soh=capacity/cap0
            data_out.append([V[:-1], dcdv])
            SOH.append(soh)
        return data_out,SOH
    def make_wb_figure(self,x,y,xlim=None,ylim=None,if_scatter=False):
        """画黑白图像并转化为数组"""
        plt.figure(figsize=(2.56,2.56),dpi=100)
        if if_scatter:
            plt.scatter(x,y,color='black',s=0.8,marker='s')
        else:
            plt.plot(x,y,color='black')
        if xlim:
            plt.xlim(xlim)
        if ylim:
            plt.ylim(ylim)
        plt.axis('off')
        fig = plt.gcf()  # 获取当前图形
        canvas = fig.canvas
        canvas.draw()  # 绘制图形
        # 获取RGBA图像数据
        image_data_rgba = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        image_data_rgba = image_data_rgba.reshape(canvas.get_width_height()[::-1] + (3,))
        # 转换为单通道图像数据（灰度）
        image_data_gray = image_data_rgba.mean(axis=2)
        # 清除图形，避免重复绘制
        plt.close(fig)
        return image_data_gray
    def get_figs_IC(self):
        """获取标准归一化退化图组"""
        data,SOHS=self.get_IC_charge()
        figs=[]
        print("\n构建" + self.file_name + "电池充电IC图像特征数据集\n")
        matplotlib.use("Agg")
        for i in tqdm(range(len(data))):
            data_i = data[i]
            """IC图像"""
            V = data[i][0]
            dcdv = data[i][1]
            fig = self.make_wb_figure(data_i[0], data_i[1], xlim=[2.6,4.2],ylim=[0,1200],if_scatter=False)
            """整合"""
            fig = fig.astype(np.float32)/255
            figs.append(fig)
        figs = np.stack(figs, axis=0)
        sohs = np.stack(SOHS, axis=0)
        np.savez('../data/OXFORD_data/npz_path/' + self.file_name + '.npz', array1=figs, array2=sohs)
    def read_data(self):
        path='../data/OXFORD_data/npz_path/'+self.file_name+'.npz'
        arrays = np.load(path)
        return arrays['array1'], arrays['array2']
def get_heat_map( ):
    data = np.random.rand(16, 16)
    # 设置seaborn样式
    sns.set()
    # 创建热图
    plt.figure(figsize=(8, 8))  # 设置图像大小
    sns.heatmap(data, annot=False, cmap='viridis')  # 绘制热图，annot=False表示不显示数值
    # 显示图形
    plt.show()
def get_point(data):
    row_indices = np.zeros(256, dtype=int)
    # 遍历每一列
    for i in range(256):
        # 使用np.argmax来找到第一个小于0.5的数的索引
        # np.argwhere返回的是布尔数组中为True的元素的索引
        # 我们使用负号来反转数组，这样我们可以从下往上查找
        # [-1]表示我们只关心第一个小于0.5的元素
        condition = data[::-1, i] < 1
        if np.any(condition):
            row_indices[i] = np.argmax(condition)
        return row_indices

if __name__ == '__main__':
    dataset = OXFORD_data(file_name='Cell8')
    """画图"""
    import seaborn as sns

    plt.figure(figsize=(8, 8))
    data, SOHS = dataset.get_IC_charge()
    palette = sns.color_palette("viridis", n_colors=len(data))  # 使用 viridis 渐变色
    for i in range(len(data)):
        if i >= 0:
            V = data[i][0]
            dcdv = data[i][1]/1000
            sns.lineplot(x=V, y=dcdv, color=palette[i], alpha=0.5)  # 使用渐变色，设置透明度
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(5))  # X轴5个刻度
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(6))  # Y轴5个刻度
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=50)
    plt.xlabel('Voltage (V)', fontsize=60)
    plt.ylabel('dQ/dV (Ah/V)', fontsize=60)
    plt.show()
    """数据整合"""
    # file_names = ['Cell1','Cell2', 'Cell3','Cell4','Cell5','Cell6','Cell7','Cell8']
    # f_figs = []
    # f_sohs = []
    # for i in range(8):
    #     file_name = file_names[i]
    #     dataset = OXFORD_data(file_name=file_name)
    #     dataset.get_figs_IC()
    # """查看数据"""
    # file_names = ['Cell1', 'Cell2', 'Cell3', 'Cell4', 'Cell5', 'Cell6', 'Cell7', 'Cell8']
    # f_figs = []
    # f_sohs = []
    # for i in range(1):
    #     file_name = file_names[i]
    #     dataset = OXFORD_data(file_name=file_name)
    #     figs, sohs = dataset.read_data()
    #     f_figs.append(figs[76]-figs[50])
    #     f_sohs.append(sohs)
    #     plt.plot(sohs)
    # plt.imshow(f_figs[0])
    # plt.show()
    # print(f_sohs)