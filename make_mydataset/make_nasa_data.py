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
d2l.use_svg_display()
class NASA_data(object):
    def __init__(self, target_data_path='../data/NASA_data', file_name='B0007'):
        """待处理的问价你路径"""
        self.file_name = file_name
        self.target_data_path = target_data_path
        self.source_data_path = '../source_data/NASA_data/NASA_5/1_BatteryAgingARC-FY08Q4/'
        self.csv_path = self.target_data_path + 'csv_path'
        self.pkl_path = self.target_data_path + 'npz_path'
    def subtract_first_element(self,arr):
        """数组全部减去第一个元素，用来处理时间数据据"""
        if not arr:  # 检查数组是否为空
            return []
        first_element = arr[0]
        return [x - first_element for x in arr]
    def get_data(self):
        """获取特征数据点组"""
        file_path= self.source_data_path + self.file_name + '.mat'
        data=scipy.io.loadmat(file_path)[self.file_name]
        cols=data[0][0][0][0]
        n_cols=cols.shape[0]
        cycles=0
        charges_cycles=[]
        discharges_cycles=[]
        print('整合循环数据：\n')
        for i in (range(n_cols)):
            if str(cols[i][0][0]) == 'charge':
                charge_data=cols[i][3][0][0]
                charges_cycles.append(charge_data)
            if str(cols[i][0][0]) == 'discharge':
                discharge_data=cols[i][3][0][0]
                discharges_cycles.append(discharge_data)
                cycles+=1
        all_cycles=[]
        for i in range(cycles):
            all_cycles.append([charges_cycles[i],discharges_cycles[i]])
        """对循环数据进行处理"""
        print("整合完毕\n数据处理：\n")
        data_out=[]
        for i in tqdm(range(len(all_cycles))):
            cycle=i+1
            charge_data,discharge_data=all_cycles[i][0],all_cycles[i][1]
            """充电过程"""
            charge_i=charge_data['Current_measured'][0]
            charge_v=charge_data['Voltage_measured'][0]
            charge_t=charge_data['Time'][0]
            df_charge=pd.DataFrame({'charge_i':charge_i,'charge_v':charge_v,'charge_t':charge_t})
            df_cc=df_charge[(df_charge['charge_i']>=1.5)&(df_charge['charge_i']<=1.8)]
            df_cv=df_charge[(df_charge['charge_v']>=4.2)&(df_charge['charge_v']<=4.3)]
            """获取恒流时的电压区间【4.-4.2】和恒压时的电流区间【1.-0.5】"""
            V_vt=df_cc[(df_cc['charge_v']>=4.0 )& (df_cc['charge_v']<=4.2)]['charge_v'].tolist()
            T_vt=df_cc[(df_cc['charge_v']>=4.0 )& (df_cc['charge_v']<=4.2)]['charge_t'].tolist()
            T_vt=self.subtract_first_element(T_vt)
            I_it=df_cv[(df_cv['charge_i']>=0.5)&(df_cv['charge_i']<=1)]['charge_i'].tolist()
            T_it=df_cv[(df_cv['charge_i']>=0.5)&(df_cv['charge_i']<=1)]['charge_t'].tolist()
            T_it=self.subtract_first_element(T_it)
            """对放电的数据进行提取"""
            discharge_i=discharge_data['Current_load'][0]
            discharge_v=discharge_data['Voltage_load'][0]
            discharge_t=discharge_data['Time'][0]
            df_disc=pd.DataFrame({'disc_i':discharge_i,'disc_v':discharge_v,'disc_t':discharge_t})
            """获取放电的电压电流和时间"""
            if df_disc['disc_i'].tolist()[0]<=0:
                df_need=df_disc[(df_disc['disc_i']<=-1.9)&(df_disc['disc_i']>=-2.1)]
            else:
                df_need = df_disc[(df_disc['disc_i'] <= 2.1) & (df_disc['disc_i'] >= 1.9)]
            df_last=df_need[(df_need['disc_v']>=2.0)]
            V_vc=df_last['disc_v'].tolist()
            T_vc=df_last['disc_t'].tolist()
            T_vt=self.subtract_first_element(T_vt)
            I_vc=df_last['disc_i'].tolist()
            C_vc=T_vc
            data_out.append([cycle,I_it,T_it,V_vt,T_vt,V_vc,C_vc])
        return data_out
    def get_IC_charge(self,dV_threshold=0.00005,window_size=30):
        """获取充电的IC曲线数据"""
        file_path = self.source_data_path + self.file_name + '.mat'
        data = scipy.io.loadmat(file_path)[self.file_name]
        cols = data[0][0][0][0]
        n_cols = cols.shape[0]
        cycles = 0
        charges_cycles = []
        discharges_cycles = []
        print('整合循环数据：\n')
        for i in tqdm((range(n_cols))):
            if str(cols[i][0][0]) == 'charge':
                charge_data = cols[i][3][0][0]
                charges_cycles.append(charge_data)
            if str(cols[i][0][0]) == 'discharge':
                discharge_data = cols[i][3][0][0]
                discharges_cycles.append(discharge_data)
                cycles += 1
        all_cycles = []
        for i in range(cycles):
            all_cycles.append([charges_cycles[i], discharges_cycles[i]])
        """对循环数据进行处理"""
        print("整合完毕\n数据处理：\n")
        data_out,SOHS= [],[]
        cap_0=1
        max_height=0
        for all_i in tqdm(range(len(all_cycles))):
            cycle = all_i + 1
            charge_data, discharge_data = all_cycles[all_i][0], all_cycles[all_i][1]
            c = charge_data['Current_measured'][0].T
            v = charge_data['Voltage_measured'][0].T
            t = charge_data['Time'][0].T
            v, c, t = np.squeeze(v), np.squeeze(c), np.squeeze(t)
            v, c, t = v.reshape(-1), c.reshape(-1), t.reshape(-1)
            step = 0.004  # 电压微分最小值,防止dV过小导致的噪声阶跃
            numV = len(v)  # 索引
            i = 0
            # 补充一些边缘点的值，防止数值过大
            dV = [1000]
            Vind = [3.10]
            dQ = [0]
            while i < (numV - 2):
                diff = 0
                j = 0
                # 寻找符合电压微分最小值的间隔点
                while (diff < step and (i + j) < (numV - 1)):
                    j += 1
                    diff = abs(v[i + j] - v[i])
                # 检查是否到达下限,末尾反复波动，截止电压为3.4V
                # 选择结束末尾的插值点
                if (diff < step or v[i] > 4.194):
                    i += numV - 2
                    continue
                else:
                    dt = np.diff(t[i:i + j + 1]) / 3600  # 单位h小时
                    ch_c1 = c[i:i + j + 1]
                    dc = ch_c1[:-1] + np.diff(ch_c1) * 0.5  # 电流微分，每两个点取一个均值，单位A
                    dQ.append(np.sum(dc * dt))
                    dV.append(diff)
                    Vind.append(v[i] * 0.5 + v[i + j] * 0.5)  # 取两个点的均值
                    i += 1
                    continue
            dV.append(dV[-1])
            Vind.append(4.20)
            dQ.append(dQ[-1])
            if (min(Vind) < 3):
                continue
            dQdV = np.array(dQ) / np.array(dV)
            # 去除相同V值对应的dQdV
            unique_V, unique_indices = np.unique(Vind, return_index=True)
            unique_dQdV = []
            for i in range(len(unique_V)):
                indices = np.where(Vind == unique_V[i])[0]
                mean_dQdV = np.mean(dQdV[indices])
                unique_dQdV.append(mean_dQdV)
            if len(unique_dQdV) <= window_size:
                continue
            weights = np.ones(window_size) / window_size
            unique_dQdV = np.convolve(unique_dQdV, weights, mode='same')
            data_out.append([unique_V, unique_dQdV])
            """将capacity,soh放入"""
            capacity = discharge_data['Capacity'][0]
            if all_i == 0:
                cap_0 = capacity
            soh = capacity / cap_0
            SOHS.append(capacity)
            if max(unique_dQdV) > max_height:
                max_height = max(unique_dQdV)
        return data_out,SOHS

    def get_IC_discharge(self,dV_threshold=-0.0005,window_size=50):
        """获取放电的ic曲线"""
        data = self.get_data()
        data_out=[]
        for i in range(len(data)):
            V = data[i][5]
            C = data[i][6]
            dC = np.diff(C)
            dV = np.diff(V)
            zero_index = np.where(dV >= dV_threshold)
            dC = np.delete(dC, zero_index)
            dV = np.delete(dV, zero_index)
            V = np.delete(V, zero_index)
            dcdv = dC / dV
            weights = np.ones(window_size) / window_size
            dcdv = np.convolve(dcdv, weights, mode='same')
            data_out.append([V[1:],dcdv])
        return data_out
#--------------------------归一图形化进行去噪神经网络---------------------------------------------------------------------
    def make_wb_figure(self,x,y,xlim=None,ylim=None,if_scatter=False):
        """画黑白图像并转化为数组"""
        plt.figure(figsize=(2.56,2.56),dpi=100)
        if if_scatter:
            plt.scatter(x,y,color='black',s=0.8,marker='s')
        else:
            plt.plot(x,y,color='black',linewidth=3)
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
        max_h=0
        for i in range(len(data)):
            cycle_1=data[i][1]
            m=max(cycle_1)
            if max_h<m:
                max_h=m
        for i in tqdm(range(len(data))):
            if i == 0:
                continue
            data_i = data[i]
            """IC图像"""
            V = data[i][0]
            dcdv = data[i][1]
            fig = self.make_wb_figure(data_i[0], data_i[1], xlim=[3.8,4.2],ylim=[0,7],if_scatter=False)
            """整合"""
            fig = fig.astype(np.float32)/255
            figs.append(fig)
        figs = np.stack(figs, axis=0)
        sohs = np.stack(SOHS[1:], axis=0)
        np.savez('../data/NASA_data/npz_path/' + self.file_name + '.npz', array1=figs, array2=sohs)
    def read_data(self):
        path='../data/NASA_data/npz_path/'+self.file_name+'.npz'
        arrays = np.load(path)
        return arrays['array1'], arrays['array2']

#------------------------------------测试--------------------------------------
if __name__ == '__main__':
    dataset=NASA_data(file_name='B0005')
    # data=dataset.get_data()
    """测试1"""
    # data = dataset.get_IC_discharge()
    # for i in range(len(data)):
    #     V = data[i][0]
    #     dcdv = data[i][1]
    #     plt.scatter(V, dcdv,s=0.5)
    # plt.show()
    """测试2"""
    # data,_,_=dataset.get_IC_charge()
    # for i in range(len(data)):
    #     if i%10==0:
    #         V = data[i][0]
    #         dcdv = data[i][1]
    #         plt.plot(V, dcdv)
    # plt.show()
    """测试3"""
    # dataset.get_figs_IC()
    """test4"""
    # figs,sohs=dataset.read_data()
    # plt.imshow(figs[60]-figs[100])
    # plt.show()
    """整理数据"""
    # file_names = ['B0005', 'B0006', 'B0007', 'B0018']
    # for i in range(4):
    #     file_name = file_names[i]
    #     dataset=NASA_data(file_name=file_name)
    #     dataset.get_figs_IC()
    """查看数据"""
    file_names = ['B0005', 'B0006', 'B0007', 'B0018']
    f_figs=[]
    f_sohs=[]
    for i in range(4):
        file_name = file_names[i]
        dataset=NASA_data(file_name=file_name)
        figs,sohs=dataset.read_data()
        f_figs.append(figs[0])
        f_figs.append(figs[100])
        f_sohs.append(sohs)
        plt.plot(sohs)
        print(len(figs))
        print(len(sohs))
    plt.imshow(f_figs[0]-f_figs[2]+f_figs[4]-f_figs[6])
    plt.show()





