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
from scipy.signal import medfilt



#--------------------------合并单个电池的测试数据-------------------------------------------------------
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]
def get_data(name,root_dir,sheet_name):
    new_rootdir=root_dir+'/'+name
    all_files=os.listdir(new_rootdir)
    all_files = sorted(all_files, key=natural_sort_key)
    """从零开始累计所有的循环"""
    cycle=0
    dataframes=[]
    count=0
    for file in all_files:
        print('READING'+':'+file)
        data=pd.read_excel(new_rootdir+'/'+file,sheet_name=sheet_name)
        df=pd.DataFrame(data)
        df['Cycle_Index']=df['Cycle_Index']+cycle
        cycle=max(i for i in df['Cycle_Index'].values)
        dataframes.append(df)
    df = pd.concat(dataframes)
    file_name='../source_data/CACLE_data/CS2_combine'+'/'+name+'_combine.xlsx'
    df.to_excel(file_name,index=False)
#-----------------------------------处理数据的类-----------------------------------------------------------------------
"""定义用于处理马里兰大学的cacle类"""
class CACLE_data(object):
    def __init__(self, target_data_path='../data/CACLE_data', file_name='CS2_38'):
        """待处理的问价你路径"""
        self.file_name = file_name
        self.target_data_path = target_data_path
        self.source_data_path = '../source_data/CACLE_data/CS2_combine/'
        self.csv_path = self.target_data_path + 'csv_path'
        self.pkl_path = self.target_data_path + 'npz_path'
    def subtract_first_element(self,arr):
        """数组全部减去第一个元素，用来处理时间数据据"""
        if not arr:  # 检查数组是否为空
            return []
        first_element = arr[0]
        return [x - first_element for x in arr]
    def get_data(self):
        file_path = self.source_data_path + self.file_name + '_combine.xlsx'
        df = pd.read_excel(file_path, sheet_name='Sheet1')
        df = pd.DataFrame(df)
        cycles = set(df['Cycle_Index'].tolist())
        data_out = []
        cap_0 = 0
        SOH = 1
        max_Ti = 0
        max_Tv = 0
        max_Q = 0
        min_Q = 10000000
        print("锂电池数据整合：\n")
        for cycle in tqdm(cycles):
            df_n = df[df['Cycle_Index'] == cycle]
            "恒压阶段-计算最后的电流"
            df_I = df_n[df_n['Step_Index'] == 4]
            I_it = df_I[(df_I['Current(A)'] <= 0.5) & (df_I['Current(A)'] >= 0.1)]['Current(A)'].tolist()
            T_it=df_I[(df_I['Current(A)'] <= 0.5) & (df_I['Current(A)'] >= 0.1)]['Step_Time(s)'].tolist()
            T_it = self.subtract_first_element(T_it)
            """恒流阶段-计算最后的电压"""
            df_V = df_n[df_n['Step_Index'] == 2]
            V_stop = df_V['Voltage(V)'].tolist()[-1]
            V_begin = V_stop - 0.2
            V_vtq = df_V[(df_V['Voltage(V)'] >= V_begin) & (df_V['Voltage(V)'] <= V_stop)]['Voltage(V)'].tolist()
            T_vtq=df_V[(df_V['Voltage(V)'] >= V_begin) & (df_V['Voltage(V)'] <= V_stop)]['Step_Time(s)'].tolist()
            T_vtq = self.subtract_first_element(T_vtq)
            Q_vtq=df_V[(df_V['Voltage(V)'] >= V_begin) & (df_V['Voltage(V)'] <= V_stop)]['Charge_Capacity(Ah)'].tolist()
            """放电的电压容量曲线"""
            df_discharge = df_n[df_n['Step_Index'] == 7]
            if len(df_discharge['Voltage(V)'].tolist())==0:
                continue
            V_disbegin=df_discharge['Voltage(V)'].tolist()[0]
            V_distop =3
            V_vc=df_discharge[(df_discharge['Voltage(V)'] >= V_distop) & (df_discharge['Voltage(V)'] <= V_disbegin)]['Voltage(V)'].tolist()
            C_max=df_discharge['Charge_Capacity(Ah)'].tolist()[0]
            C_vc=df_discharge[(df_discharge['Voltage(V)'] >= V_distop) & (df_discharge['Voltage(V)'] <= V_disbegin)]['Discharge_Capacity(Ah)'].tolist()
            C_vc = self.subtract_first_element(C_vc)
            """计算SOH"""
            df_SOH = df_n[df_n['Step_Index'] == 7]
            df_cap = df_SOH[(df_SOH['Voltage(V)'] <= 3.8) & (df_SOH['Voltage(V)'] >= 3.4)]
            if cycle == 1:
                cap_0 = df_cap['Discharge_Energy(Wh)'].tolist()[0] - df_cap['Discharge_Energy(Wh)'].tolist()[-1]
            if len(df_cap['Discharge_Energy(Wh)'].tolist()) == 0:
                SOH = SOH
            else:
                cap_i = df_cap['Discharge_Energy(Wh)'].tolist()[0] - df_cap['Discharge_Energy(Wh)'].tolist()[-1]
                SOH = cap_i / cap_0
            """最大时间数据---用来归一化"""

            data_out.append([cycle, T_it, I_it, max_Ti, T_vtq, V_vtq, Q_vtq, max_Tv, min_Q, max_Q,V_vc,C_vc, SOH])
            # if cycle == 20:
            #     break
        return data_out
    def get_IC_charge(self,dV_threshold=0.000001,window_size=5):
        file_path = self.source_data_path + self.file_name + '_combine.xlsx'
        df = pd.read_excel(file_path, sheet_name='Sheet1')
        df = pd.DataFrame(df)
        cycles = set(df['Cycle_Index'].tolist())
        cap_0 = 0
        SOH = 1
        data_out,SOHS = [],[]
        for cycle in tqdm(cycles):
            df_n = df[df['Cycle_Index'] == cycle]
            """恒流阶段-计算ic曲线"""
            df_IC = df_n[df_n['Step_Index'] == 2]
            df_c = df_n[(df_n['Step_Index'] == 2) | (df_n['Step_Index'] == 4)]
            C=df_IC['Charge_Capacity(Ah)'].tolist()
            V=df_IC['Voltage(V)'].tolist()
            dC = np.diff(C)  # 在末尾和开头添加0
            dV = np.diff(V)  # 在末尾和开头添加0
            zero_index = np.where(dV <= dV_threshold)
            dC = np.delete(dC, zero_index)
            dV = np.delete(dV, zero_index)
            V = np.delete(V, zero_index)
            dcdv = dC / dV
            if len(dcdv)==0 or len(dcdv)<window_size:
                continue
            """计算SOH"""
            df_SOH = df_n[df_n['Step_Index'] == 7]
            df_cap = df_SOH[(df_SOH['Voltage(V)'] <= 3.8) & (df_SOH['Voltage(V)'] >= 3.4)]
            if cycle == 1:
                # cap_0 = df_cap['Discharge_Energy(Wh)'].tolist()[0] - df_cap['Discharge_Energy(Wh)'].tolist()[-1]
                cap_0=max(df_c['Charge_Capacity(Ah)']) - min(df_c['Charge_Capacity(Ah)'])
            # if len(df_cap['Discharge_Energy(Wh)'].tolist()) == 0:
            #     SOH = SOH
            else:
                # cap_i = df_cap['Discharge_Energy(Wh)'].tolist()[0] - df_cap['Discharge_Energy(Wh)'].tolist()[-1]
                cap_i = max(df_c['Charge_Capacity(Ah)']) - min(df_c['Charge_Capacity(Ah)'])
                SOH = cap_i / cap_0
            weights = np.ones(window_size) / window_size
            dcdv = np.convolve(dcdv, weights, mode='same')
            data_out.append([V[:-1], dcdv])
            SOHS.append(SOH)
        return data_out,SOHS
    def get_IC_discharge(self,dV_threshold=-0.001,window_size=20):
        data = self.get_data()
        data_out,SOH=[],[]
        for i in range(len(data)):
            C = data[i][11]
            V = data[i][10]
            dC = np.diff(C)  # 在末尾和开头添加0
            dV = np.diff(V)  # 在末尾和开头添加0
            zero_index = np.where(dV >= dV_threshold)
            dC = np.delete(dC, zero_index)
            dV = np.delete(dV, zero_index)
            V = np.delete(V, zero_index)
            dcdv = dC / dV
            weights = np.ones(window_size) / window_size
            dcdv = np.convolve(dcdv, weights, mode='same')
            data_out.append([V[:-1],dcdv])
        return data_out
    #--------------------------归一图形化进行去噪神经网络---------------------------------------------------------------------
    def make_wb_figure(self,x,y,xlim=None,ylim=None,if_scatter=False):
        """画黑白图像并转化为数组"""
        plt.figure(figsize=(2.56,2.56),dpi=100)
        if if_scatter:
            plt.scatter(x,y,color='black',s=5,marker='s')
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
            fig = self.make_wb_figure(data_i[0], data_i[1], xlim=[3.4,4.3],ylim=[0,10],if_scatter=False)
            """整合"""
            fig = fig.astype(np.uint8)/255
            figs.append(fig)
        figs = np.stack(figs, axis=0)
        sohs = np.stack(SOHS, axis=0)
        np.savez('../data/CACLE_data/npz_path/' + self.file_name + '.npz', array1=figs, array2=sohs)
    def read_data(self):
        path='../data/CACLE_data/npz_path/'+self.file_name+'.npz'
        arrays = np.load(path)
        return arrays['array1'], arrays['array2']

#---------------------------------测试--------------------------------------------
if __name__ == '__main__':
    dataset=CACLE_data(file_name='CS2_38')
    """test_1"""
    # data=dataset.get_IC_discharge()
    # for i in range(len(data)):
    #     V=data[i][0]
    #     dcdv=data[i][1]
    #     plt.plot(V,dcdv)
    # plt.show()
    """test_2"""
    import seaborn as sns
    data,SOHS = dataset.get_IC_charge()
    palette = sns.color_palette("viridis", n_colors=len(data))  # 使用 viridis 渐变色
    plt.figure(figsize=(10, 10))  # 建议设置合适画布尺寸
    for i in range(len(data)):
        if i>=0:
            V = data[i][0]
            dcdv = data[i][1]
            sns.lineplot(x=V, y=dcdv, color=palette[i], alpha=0.5)  # 使用渐变色，设置透明度
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(5))  # X轴5个刻度
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(5))  # Y轴5个刻度
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=50)
    plt.xlabel('Voltage (V)', fontsize=60)
    plt.ylabel('dQ/dV (Ah/V)', fontsize=60)
    plt.show()
    """test_3"""
    palette = sns.color_palette("viridis", n_colors=len(data))  # 使用 viridis 渐变色
    plt.figure(figsize=(10, 10))  # 建议设置合适画布尺寸
    for i in range(len(data)):
        if i != 0:
            sns.lineplot(data=data[i][0], color=palette[i], alpha=0.5)
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(5))  # X轴5个刻度
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(5))  # Y轴5个刻度
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=50)
    plt.xlabel('Sample points', fontsize=60)
    plt.ylabel('Voltage(V)', fontsize=60)
    plt.show()
    # dataset.get_figs_IC()
    """test4"""
    # figs,sohs=dataset.read_data()
    # plt.imshow(figs[2])
    # plt.show()
    """test_5"""
    # for i in range(4):
    #     file_name='CS2_3'+str(i+5)
    #     dataset=CACLE_data(file_name=file_name)
    #     dataset.get_figs_IC()
