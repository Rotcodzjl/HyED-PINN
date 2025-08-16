import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# plt.rcParams['xtick.labelsize'] = 12
# plt.rcParams['ytick.labelsize'] = 12
# 假设 RMSE 是一个 4x4 的矩阵，表示预训练数据集（行）和微调数据集（列）的 RMSE 值
RMSE = np.array([
    [0, 2.7315, 1.1462,1.2543 ],
    [1.7121, 0, 1.2234, 1.3224],
    [1.6431, 2.7145, 0, 1.6212],
    [1.3724, 1.9653, 1.4572, 0]
])

# # 数据集名称
pretrain_datasets = ['CACLE dataset', 'HUST dataset', 'XJTU dataset', 'Oxford dataset']
finetune_datasets = ['CACLE dataset', 'HUST dataset', 'XJTU dataset', 'Oxford dataset']
#
# 绘制热图
sns.heatmap(RMSE, annot=True, fmt=".2f", cmap='YlGnBu', xticklabels=finetune_datasets, yticklabels=pretrain_datasets)
plt.xlabel('Finetune Datasets')
plt.ylabel('Pretrain Datasets')
plt.title('RMSE of Transfer Learning', fontsize=16)
plt.show()
from matplotlib.ticker import MaxNLocator

# 设置全局字体（基础字号从10→20）
plt.rcParams.update({
    'font.size': 20,          # 全局基础字号
    'axes.titlesize': 24,     # 标题字号
    'axes.labelsize': 22      # 坐标轴标签字号
})

# 创建热力图
fig, ax = plt.subplots(figsize=(10,8))
heatmap = sns.heatmap(
    RMSE,
    annot=True,
    annot_kws={'size': 20},  # 注释文字字号
    fmt=".2f",
    cmap='YlGnBu',
    xticklabels=finetune_datasets,
    yticklabels=pretrain_datasets,
    cbar_kws={
        'label': 'RMSE',
        'ticks': MaxNLocator(nbins=4)  # 关键参数：强制最多5个刻度（nbins+1）
    },
    ax=ax
)

# 设置刻度标签字号
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', fontsize=18)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=18)

# 设置颜色条字号
cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(labelsize=18)  # 刻度数字号
cbar.set_label('RMSE (%)', fontsize=20, labelpad=15)  # 颜色条标题

plt.title('RMSE of Transfer Learning')
plt.tight_layout()
plt.show()
# 假设 RMSE 是一个字典，表示不同预训练模型在微调数据集上的 RMSE 值


#----------------------------------------------------------------------------------------------------
RMSE = {
    'CACLE dataset': [0, 2.7315, 1.1462,1.2543 ],
    'Oxford dataset': [1.3724, 1.9653, 1.4572, 0],
    'HUST dataset':  [1.7121, 0, 1.2234, 1.3224],
    'XJTU dataset':   [1.6431, 2.7145, 0, 1.6212],

}
# # 微调数据集名称
# finetune_datasets =  ['CACLE dataset', 'HUST dataset', 'XJTU batch-1', 'Oxford dataset']
#
# # 绘制雷达图
angles = np.linspace(0, 2 * np.pi, len(finetune_datasets), endpoint=False).tolist()
angles += angles[:1]  # 闭合图形


# 设置全局字体大小
plt.rcParams.update({'font.size': 25})  # 基础字号从默认10增加到14

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'polar': True}, facecolor='white')

# 绘制线条 (线宽增加到2.5)
line_styles = ['-', '--', '-.', ':']  # 添加不同线型增强区分度
for (model, values), ls in zip(RMSE.items(), line_styles):
    closed_values = values + values[:1]
    ax.plot(angles, closed_values,
            linewidth=2.5,  # 线条加粗
            linestyle=ls,
            marker='o',     # 添加数据点标记
            markersize=6,
            label=model)

# 坐标轴标签设置
ax.set_xticks(angles[:-1])
ax.set_xticklabels(finetune_datasets,
                  fontsize=14,  # X轴标签字号
                  rotation=20,  # 旋转角度避免重叠
                  ha='center')  # 水平居中

# 径向坐标设置
ax.set_rlabel_position(30)  # 调整数值标签位置
ax.tick_params(axis='y', labelsize=16)  # Y轴刻度字号

# 标题和图例
ax.set_title('RMSE of Transfer Learning on Different Finetune Datasets',
            fontsize=24, pad=20)
ax.legend(loc='upper right',
         bbox_to_anchor=(1.25, 1.05),  # 调整图例位置
         fontsize=16,
         framealpha=0.9)

# 网格线增强
ax.grid(linewidth=0.4, alpha=0.7)

plt.tight_layout()
plt.show()

